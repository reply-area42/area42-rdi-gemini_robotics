import argparse
import logging_mp

# Configura il logging PRIMA di altri import
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

from wcwidth import width
from camera.image_client_depth import ImageClient_depth
import cv2
import numpy as np
import json
from secrets import API_KEY
from google import genai
from google.genai import types
from robot_visualizer import visualize_robot_on_meshcat
import position_calculator as pc
from call_gemini import call_model
import threading
import time
from multiprocessing import shared_memory




from robot_control.robot_arm import G1_29_ArmController
from robot_control.robot_arm_ik import G1_29_ArmIK
#from robot_control.robot_hand_inspire_dfx import Inspire_Controller

# from teleop.utils.episode_writer import EpisodeWriter
# from teleop.utils.ipc import IPC_Server
from sshkeyboard import listen_keyboard, stop_listening
# import config_mocked_avp

# filepath: /home/area42-user/area42-rdi-gemini_robotics/gemini_solve_ik.py

# Define the prompt for Gemini to return a list of trajectory points
PROMPT = """
Analyze the provided RGB and depth images (horizontally stacked) to detect 
the cylinder and the basket, then generate a trajectory of 15 points moving the cylinder
from its current position to the basket.
 Return a list of points following this JSON format:
 [{"u": <u_pixel>, "v": <v_pixel>, "depth_mm": <depth_in_mm>}, ...].
Coordinates should be in pixel format (u, v) normalized to 0-1000 range,
and depth in millimeters.
"""

def main():
    parser = argparse.ArgumentParser(description="Script to initialize CameraIntrinsics, stream RGB and depth, and use Gemini to detect trajectory points.")
    parser.add_argument('--camera_id', type=str, required=True, help='Camera device ID/serial number')
    # ---------------------------  if we need to add more arguments, we can do it here ---------------------------
    args = parser.parse_args()

    tv_img_shm = None
    tv_depth_shm = None

    
    arm_ik = G1_29_ArmIK(Visualization=False)
    arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
    print("G1_29 Initialized")
            

    try: #blocco di try per assicurarsi che la memoria condivisa venga pulita anche in caso di errori
        # Initialize CameraIntrinsics
        try:
            intrinsics = pc.CameraIntrinsics(args.camera_id)
            print(f"CameraIntrinsics initialized with ID: {args.camera_id}")
            print(f"fx={intrinsics.fx}, fy={intrinsics.fy}, cx={intrinsics.cx}, cy={intrinsics.cy}")
        except Exception as e:
            print(f"Error initializing CameraIntrinsics: {e}")
            return

        # Set up image client for RGB and depth streaming 
        img_config = {
            'fps': 15,
            'head_camera_type': 'realsense',
            'head_camera_image_shape': [720, 1280],  # Head camera resolution
            'depth_image_shape': [720, 1280],
            'head_camera_id_numbers': [args.camera_id],  # Use the provided camera ID
        }

        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)
        tv_depth_shape = tuple(img_config['depth_image_shape'])

        # Create shared memory for RGB and depth
        tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
        tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)

        tv_depth_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_depth_shape) * np.uint16().itemsize)
        tv_depth_array = np.ndarray(tv_depth_shape, dtype=np.uint16, buffer=tv_depth_shm.buf)

        # Initialize ImageClient_depth
        img_client = ImageClient_depth(
            tv_img_shape=tv_img_shape,
            tv_img_shm_name=tv_img_shm.name,
            tv_img_shape_resize=None,  # Not resizing for this script
            tv_img_resized_shm_name=None,
            server_address="192.168.123.164",  # Assuming same as teleop.py
            tv_depth_shape=tv_depth_shape,
            tv_depth_shm_name=tv_depth_shm.name,
        )

        # Start image receiving thread
        image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
        image_receive_thread.start()

        # Wait a bit for images to start streaming
        time.sleep(2)

        # Capture current RGB and depth images
        rgb_image = tv_img_array.copy()
        depth_image = tv_depth_array.copy()

        # Combine RGB and depth for Gemini input 
        # Normalize depth for visualization
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        combined = np.hstack((rgb_image, depth_colormap)).astype(np.uint8)

        # Encode the combined image
        success, encoded_image = cv2.imencode('.jpg', combined)
        if not success:
            print("Failed to encode combined image")
            return
        image_bytes = encoded_image.tobytes()

        # Call Gemini with the prompt
        try:
            response = call_model(image_bytes, PROMPT)
            trajectory_data = json.loads(response.text)
            print("Trajectory points from Gemini:")
            for point in trajectory_data:
                print(point)
            
            # Optionally, deproject to 3D points using intrinsics
            trajectory_3d = []
            for point in trajectory_data:
                u = point['u']
                v = point['v']
                depth_mm = point['depth_mm']
                if depth_mm > 0:
                    height, width = rgb_image.shape[:2]
                    abs_y1 = int(u / 1000 * height)
                    abs_x1 = int(v / 1000 * (width*2))
                    point_3d = pc.deproject_pixel(abs_y1, abs_x1, depth_mm, intrinsics)
                    trajectory_3d.append(point_3d)
                    print(f"3D point: {point_3d}")

                    target_pos = point_3d  # il tuo punto nel frame waist

                    # Orientamento: identità = polso allineato agli assi del waist
                    # Oppure copia l'orientamento corrente da tele_data
                    target_pose = np.eye(4)
                    target_pose[:3, 3] = target_pos
                    # target_pose[:3, :3] = desired_rotation  # se hai un orientamento specifico


                    current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
                    current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

                    current_left_pose = arm_ik.get_current_left_wrist_pose(current_lr_arm_q)
                    # current_right_pose = arm_ik.get_current_right_wrist_pose(current_lr_arm_q)

                    # solve ik using motor data and wrist pose, then use ik results to control arms.
                    time_ik_start = time.time()
                    sol_q, sol_tauff = arm_ik.solve_ik(
                                    left_wrist=current_left_pose,   # posa attuale del braccio sinistro (4x4)
                                    right_wrist=target_pose,         # il tuo target (4x4)
                                    current_lr_arm_q=current_lr_arm_q,
                                    current_lr_arm_dq=current_lr_arm_dq)            
                    time_ik_end = time.time()
                    logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
                    arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
                    time.sleep(0.2)  # pausa breve per vedere il movimento
                    visualize_robot_on_meshcat(sol_q[:7], sol_q[7:14])  # visualizza su meshcat


            
        except Exception as e:
            print(f"Error calling Gemini or parsing response: {e}")

    finally:
        # Clean up shared memory (always executed)
        if tv_img_shm is not None:
            tv_img_shm.close()
            tv_img_shm.unlink()
        if tv_depth_shm is not None:
            tv_depth_shm.close()
            tv_depth_shm.unlink()

if __name__ == "__main__":
    main()