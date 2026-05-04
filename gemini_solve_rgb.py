import argparse
import logging_mp

# Configura il logging PRIMA di altri import
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

from drawing_utils import draw_trajectory_points
from wcwidth import width
from image_server.image_client_with_depth import ImageClient_depth
import cv2
import numpy as np
import json
from secrets_API import API_KEY
from google import genai
from google.genai import types
from robot_visualizer import visualize_robot_on_meshcat
import position_calculator as pc
from call_gemini import call_model
import threading
import time
from multiprocessing import shared_memory
from camera.read_D435 import RealSenseCamera



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
Analyze the provided RGB image to detect 
the cylinder and the basket, then generate a trajectory of 15 points avoiding obstacles while moving the cylinder
from its current position to the basket.
 Return a list of points following this JSON format:
 [{"u": <u_pixel>, "v": <v_pixel>}, ...].
Coordinates should be in pixel format (u, v) normalized to 0-1000 range.
"""

def main():
    parser = argparse.ArgumentParser(description="Script to initialize CameraIntrinsics, stream RGB and depth, and use Gemini to detect trajectory points.")
    parser.add_argument('--camera_id', type=str, required=True, help='Camera device ID/serial number')
    # ---------------------------  if we need to add more arguments, we can do it here ---------------------------
    args = parser.parse_args()

    tv_img_shm = None
    tv_depth_shm = None

    logger_mp.info("Initializing robot arm controller and IK solver...")
    arm_ik = G1_29_ArmIK(Visualization=False)
    logger_mp.info("Arm IK solver initialized")
    arm_ctrl = G1_29_ArmController(False, False)
    logger_mp.info("G1_29 Initialized")
            

    try: #blocco di try per assicurarsi che la memoria condivisa venga pulita anche in caso di errori
        # Initialize CameraIntrinsics
        try:
            intrinsics = pc.CameraIntrinsics(args.camera_id)
            logger_mp.info(f"CameraIntrinsics initialized with ID: {args.camera_id}")
            logger_mp.info(f"fx={intrinsics.fx}, fy={intrinsics.fy}, cx={intrinsics.cx}, cy={intrinsics.cy}")
        except Exception as e:
            logger_mp.error(f"Error initializing CameraIntrinsics: {e}")
            return

        # Set up image client for RGB and depth streaming 
        # img_config = {
        #     'fps': 15,
        #     'head_camera_type': 'realsense',
        #     'head_camera_image_shape': [720, 1280],  # Head camera resolution
        #     'depth_image_shape': [720, 1280],
        #     'head_camera_id_numbers': [args.camera_id],  # Use the provided camera ID
        # }

        # tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)
        # tv_depth_shape = tuple(img_config['depth_image_shape'])

        # # Create shared memory for RGB and depth
        # tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
        # tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)

        # tv_depth_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_depth_shape) * np.uint16().itemsize)
        # tv_depth_array = np.ndarray(tv_depth_shape, dtype=np.uint16, buffer=tv_depth_shm.buf)

        # # Initialize ImageClient_depth
        # img_client = ImageClient_depth(
        #     tv_img_shape=tv_img_shape,
        #     tv_img_shm_name=tv_img_shm.name,
        #     tv_img_shape_resize=None,  # Not resizing for this script
        #     tv_img_resized_shm_name=None,
        #     server_address="192.168.123.164",  # Assuming same as teleop.py
        #     tv_depth_shape=tv_depth_shape,
        #     tv_depth_shm_name=tv_depth_shm.name,
        # )

        # # Start image receiving thread
        # image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
        # image_receive_thread.start()

        # # Wait a bit for images to start streaming
        # time.sleep(2)

        # # Capture current RGB and depth images
        # rgb_image = tv_img_array.copy()
        # depth_image = tv_depth_array.copy()
        realsenseCamera=RealSenseCamera(args.camera_id,width=1280,height=720,fps=30)
        realsenseCamera.init_realsense()
        rgb_image, depth_image = realsenseCamera.get_frame()
        # Salva l'immagine RGB
        cv2.imwrite("images/rgb_image.png", rgb_image)
        # Codifica solo l'immagine RGB per Gemini
        success, encoded_image = cv2.imencode('.jpg', rgb_image)
        if not success:
            logger_mp.error("Failed to encode rgb image")
            return
        image_bytes = encoded_image.tobytes()

        # Call Gemini with the prompt
        try:
            response = call_model(image_bytes, PROMPT)
            # Estrai il testo (adatta in base alla tua libreria Gemini)
            if hasattr(response, "text"):
                response_text = response.text
            elif hasattr(response, "candidates"):
                response_text = response.candidates[0].content.parts[0].text
            else:
                response_text = str(response)

            # Ora salva il testo come JSON
            trajectory_data = json.loads(response_text)
            with open("response.json", "w") as f:
                json.dump(trajectory_data, f, indent=2)
            with open("response.json", "r") as f:
                trajectory_data = json.load(f)
            logger_mp.info("Trajectory points from Gemini:")

            # Calcola la profondità per ogni punto usando la depth image locale
            for point in trajectory_data:
                u = point['u']
                v = point['v']
                height, width = rgb_image.shape[:2]
                y = int(u / 1000 * height)
                x = int(v / 1000 * width)
                # Prendi la profondità dal frame depth
                depth_mm = int(depth_image[y, x])
                point['depth_mm'] = int(depth_mm)

            # Salva la nuova lista con la profondità aggiornata
            with open("response_with_depth.json", "w") as f:
                json.dump(trajectory_data, f, indent=2)

            # Visualizza i punti sull'immagine RGB
            draw_trajectory_points("images/rgb_image.png", "response.json", "images/trajectory_points_overlay.png")

            # ...continua con la logica di IK e visualizzazione se necessario...

            print("\n--- Traiettoria completata. Immagine con overlay salvata. ---")
            while True:
                time.sleep(1)

        except Exception as e:
            import traceback
            print(f"Error calling Gemini or parsing response: {e}")
            traceback.print_exc()

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