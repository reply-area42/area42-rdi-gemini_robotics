"""
This is the teleop_G1 updated from the xr_teleoperate repo with the tested and recording function.
"""

import numpy as np
import time
import argparse
import pickle
import cv2
from multiprocessing import shared_memory, Value, Array, Lock
import threading
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from teleop.robot_control.robot_hand_inspire_dfx import Inspire_Controller
from teleop.image_server.image_client_with_depth import ImageClient_depth
from teleop.image_server.realsense import RealSenseCamera
from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from sshkeyboard import listen_keyboard, stop_listening
import config_mocked_avp

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int,publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion  
STOP           = False  # Enable to begin system exit procedure
RECORD_TOGGLE  = False  # [Ready] ⇄ [Recording] ⟶ [AutoSave] ⟶ [Ready]         (⇄ manual) (⟶ auto)
RECORD_RUNNING = False  # True if [Recording]
RECORD_READY   = True   # True if [Ready], False if [Recording] / [AutoSave]
# task info
TASK_NAME = None
TASK_DESC = None
ITEM_ID = None
def on_press(key):
    global STOP, START, RECORD_TOGGLE
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def on_info(info):
    """Only handle CMD_TOGGLE_RECORD's task info"""
    global TASK_NAME, TASK_DESC, ITEM_ID
    TASK_NAME   = info.get("task_name")
    TASK_DESC   = info.get("task_desc")
    ITEM_ID     = info.get("item_id")
    logger_mp.debug(f"[on_info] Updated globals: {TASK_NAME}, {TASK_DESC}, {ITEM_ID}")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, RECORD_READY
    return {
        "START": START,
        "STOP": STOP,
        "RECORD_RUNNING": RECORD_RUNNING,
        "RECORD_READY": RECORD_READY,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', type = float, default = 15.0, help = 'save data\'s frequency')

    # basic control parameters
    parser.add_argument('--xr-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device tracking source')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dftp'], default='dftp', help='Select end effector controller')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', default = False, help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording')
    parser.add_argument('--use-mocked-avp', action = 'store_true', help = 'Enable Pre Recorded AVP Data')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick_cylinder', help = 'task name for recording')
    parser.add_argument('--task-desc', type = str, default = 'pick the grey cylinder from the table and put it in the box.', help = 'task goal for recording')
    parser.add_argument('--depth', type = bool, default = False, help = 'add depth camera')


    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    # ── Depth shared memory (allocated only when --depth is requested) ───────────
    tv_depth_shm       = None
    tv_depth_array     = None
    tv_depth_shape     = None


    # ---- EXT CAMERA VIEWER (for debugging) ----

    config = {        
        # Only the first serial number is used by ImageServer        
        "head_camera_id_numbers": ["233522077337"],        
        "head_camera_width":       1280,
        "head_camera_height":      720,        
        "head_camera_fps":        15,        
        # Set to True to print FPS / latency metrics every 5 seconds        
        "unit_test":              False,    
        }
    serial_number = config["head_camera_id_numbers"][0]
    width         = config["head_camera_width"]
    height        = config["head_camera_height"]
    fps           = config["head_camera_fps"]
    camera = RealSenseCamera(            
            serial_number=serial_number,            
            width=width,
            height=height,            
            fps=fps,
        )
    
    if not camera.init_realsense():            
        logger_mp.error("[ImageServer] Camera initialization failed. Aborting show_process().")
        exit(1)
    logger_mp.info("[ImageServer] Starting viewer loop. Press 'q' to quit.")
    # Performance metric accumulators (used when unit_test=True)

    try:
        # ipc communication. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press, on_info=on_info, get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, kwargs={"on_press": on_press, "until": None, "sequential": False,}, daemon=True)
            listen_keyboard_thread.start()

        # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
        img_config = {
            'fps': 15,
            'head_camera_type': 'realsense',
            'head_camera_image_shape': [720, 1280],  # Head camera resolution
            'depth_image_shape': [720, 1280],
            'head_camera_id_numbers': ['243122076269'],
            #'head_camera_image_shape' : [480,640],
            #'head_camera_id_numbers' : [0]
            #'wrist_camera_type': 'opencv',
            #'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
            #'wrist_camera_id_numbers': [2, 4],
        }

        ASPECT_RATIO_THRESHOLD = 2.0 # If the aspect ratio exceeds this value, it is considered binocular
        if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
            BINOCULAR = True
        else:
            BINOCULAR = False
        if 'wrist_camera_type' in img_config:
            WRIST = True
        else:
            WRIST = False
        
        if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
            tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
        else:
            tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)
       
        # ── RGB shared memory ────────────────────────────────────────────────────
        
        tv_img_shm   = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
        tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)

        #la depth camera della realsense ha un solo canale anche se binoculare, e la risoluzione è la stessa del frame RGB (non divisa a metà)
        tv_depth_shape = tuple(img_config.get('depth_image_shape', img_config['head_camera_image_shape']))
        tv_depth_shm   = shared_memory.SharedMemory(create=True, size=int(np.prod(tv_depth_shape)) * np.uint16().itemsize)
        tv_depth_array = np.ndarray(tv_depth_shape, dtype=np.uint16, buffer=tv_depth_shm.buf)
        # shared memory for Televouer
        tv_img_shape_resize = (480, 640, 3)
        tv_img_resized_shm = shared_memory.SharedMemory(create=True,size=np.prod(tv_img_shape_resize) * np.uint8().itemsize)
        tv_img_resized_array = np.ndarray(tv_img_shape_resize,dtype=np.uint8,buffer=tv_img_resized_shm.buf)
        
        tele_data_list = []
        printed_warning_no_more_data = False
        if args.use_mocked_avp:
            args.headless = True
            with open(config_mocked_avp.file_path, 'rb') as f:
                tele_data_list = pickle.load(f)
        else:
            # ── Pass depth shm name to ImageClient so the server can fill it ────
            img_client = ImageClient_depth(
                tv_img_shape=tv_img_shape,
                tv_img_shm_name=tv_img_shm.name,
                tv_img_shape_resize=tv_img_shape_resize,
                tv_img_resized_shm_name=tv_img_resized_shm.name,
                server_address="192.168.123.164",
                # New optional kwargs – ImageClient must accept them (see note below)
                tv_depth_shape=tv_depth_shape if args.depth else None,
                tv_depth_shm_name=tv_depth_shm.name if args.depth else None,
            )
            image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
            image_receive_thread.start()
            # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
            tv_wrapper = TeleVuerWrapper(binocular=BINOCULAR, use_hand_tracking=args.xr_mode == "hand", img_shape=tv_img_shape_resize, img_shm_name=tv_img_resized_shm.name, 
                                        return_state_data=True, return_hand_rot_data = False)

        # arm
        arm_ik = G1_29_ArmIK()
        arm_ctrl = G1_29_ArmController(simulation_mode=args.sim, motion_mode=True)

         #end-effector
        hand_ctrl = None
        if args.ee == "dftp":
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            dual_hand_touch_right_array = Array('d', 1062, lock = False)  # [output] current left, right hand touch(1062) data.
            dual_hand_touch_left_array = Array('d', 1062, lock = False)  # [output] current left, right hand touch(1062) data.

            hand_ctrl = Inspire_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, dual_hand_touch_right_array, dual_hand_touch_left_array, simulation_mode=args.sim)
        else:
            logger_mp.warning("No end effector selected")
            pass
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20) # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # controller + motion mode
        if args.xr_mode == "controller" and args.motion:
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
            sport_client = LocoClient()
            sport_client.SetTimeout(0.0001)
            sport_client.Init()
        
        # record + headless mode
        if args.record and args.headless:
            recorder = EpisodeWriter(task_dir = args.task_dir + args.task_name, task_goal = args.task_desc, frequency = args.frequency, rerun_log = False)
        elif args.record and not args.headless:
            recorder = EpisodeWriter(task_dir = args.task_dir + args.task_name, task_goal = args.task_desc, frequency = args.frequency, rerun_log = True)

        logger_mp.info("Please enter the start signal (enter 'r' to start the subsequent program)")
        while not START and not STOP:
            time.sleep(0.01)
        logger_mp.info("start program.")
        arm_ctrl.speed_gradual_max()



        while not STOP:
            start_time = time.time()

            if not args.headless:

                color_image_ext, depth_image_ext = camera.get_frame() 
                

                if color_image_ext is None or depth_image_ext is None:                
                    logger_mp.warning("[ImageServer] Skipping frame due to capture error.")
                    continue

                # Determine target dimensions from the color frame                
                target_h, target_w = color_image_ext.shape[:2]
                # Convert depth uint16 → colorized 8-bit BGR                
                depth_colormap = camera._depth_to_colormap(depth_image_ext, target_h, target_w)
                # Concatenate color and depth frames horizontally                
                combined = np.hstack((color_image_ext, depth_colormap))
                cv2.imshow("RGB | Depth", combined)
                tv_resized_image = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
                cv2.imshow("record image", tv_resized_image)


                # opencv GUI communication
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    START = False
                    STOP = True
                elif key == ord('s'):
                    RECORD_TOGGLE = True

            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()

            # get input data
            if args.use_mocked_avp:
                tele_data = config_mocked_avp.get_item(tele_data_list)
                if tele_data is None:
                    if printed_warning_no_more_data is not True:
                        printed_warning_no_more_data = True
                        print("No more items in tele_data_list")
                    continue
            else:
                tele_data = tv_wrapper.get_motion_state_data()

            if (args.ee == "dftp" or args.ee == "inspire1") and args.xr_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            else:
                pass        

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # get current hand touch data.


            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_arm_pose, tele_data.right_arm_pose, current_lr_arm_q, current_lr_arm_dq)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            #record data
            if args.record:
                RECORD_READY = recorder.is_ready()
                # dex hand or gripper
                if (args.ee == "dftp") and args.xr_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        left_hand_touch = dual_hand_touch_left_array[:]
                        right_hand_touch = dual_hand_touch_right_array[:]
                        current_body_state = []
                        current_body_action = []
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    left_hand_touch = []
                    right_hand_touch = []
                    current_body_state = []
                    current_body_action = []
                # head image
                current_tv_image = tv_img_array.copy()
                #ext image
                current_ext_img= color_image_ext.copy() if color_image_ext is not None else np.zeros((480,640,3), dtype=np.uint8)
                current_ext_depth_img= depth_image_ext.copy() if depth_image_ext is not None else np.zeros((480,640), dtype=np.uint16)
                # ── Snapshot depth (uint16, mm) ──────────────────────────────────
                current_depth_image = tv_depth_array.copy() 
                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    ext_colors = {}
                    ext_depths = {}
                    if BINOCULAR:
                        colors[f"color_{0}"] = current_tv_image[:, :tv_img_shape[1]//2]
                        colors[f"color_{1}"] = current_tv_image[:, tv_img_shape[1]//2:]
                    else:
                        colors[f"color_{0}"] = current_tv_image
                    #colors["color_1"] = current_ext_img
                    depths["depth_0"] = current_depth_image
                    #depths["depth_1"] = current_ext_depth_img
                    #adding ext camera
                    ext_colors["ext_color_0"] = current_ext_img
                    ext_depths["ext_depth_0"] = current_ext_depth_img
                    # ── Tactile split ────────────────────────────────────────────
                    cut_points = np.cumsum([9, 96, 80, 9, 96, 80, 9, 96, 80, 9, 96, 80, 9, 96, 9, 96])
                    # Splits the array into a list of sub-arrays
                    right_parts = np.split(dual_hand_touch_right_array, cut_points)
                    #left_parts = np.split(dual_hand_touch_left_array, cut_points)  SINISTRA
                    
                   
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }

                    tactiles = {
                        "right_ee": {
                            "fingerone_tip_touch": right_parts[0].tolist(),
                            "fingerone_top_touch": right_parts[1].tolist(),
                            "fingerone_palm_touch": right_parts[2].tolist(),
                            "fingertwo_tip_touch": right_parts[3].tolist(),
                            "fingertwo_top_touch": right_parts[4].tolist(),
                            "fingertwo_palm_touch": right_parts[5].tolist(),
                            "fingerthree_tip_touch": right_parts[6].tolist(),
                            "fingerthree_top_touch": right_parts[7].tolist(),
                            "fingerthree_palm_touch": right_parts[8].tolist(),
                            "fingerfour_tip_touch": right_parts[9].tolist(),
                            "fingerfour_top_touch": right_parts[10].tolist(),
                            "fingerfour_palm_touch": right_parts[11].tolist(),
                            "fingerfive_tip_touch": right_parts[12].tolist(),
                            "fingerfive_top_touch": right_parts[13].tolist(),
                            "fingerfive_middle_touch": right_parts[14].tolist(),
                            "fingerfive_palm_touch": right_parts[15].tolist(),
                            "palm_touch": right_parts[16].tolist(),
                        },
                        "left_ee": { # SINISTRA
                                "fingerone_tip_touch": [], # left_parts[0].tolist(),
                                "fingerone_top_touch": [], # left_parts[1].tolist(),
                                "fingerone_palm_touch": [], # left_parts[2].tolist(),
                                "fingertwo_tip_touch": [], # left_parts[3].tolist(),
                                "fingertwo_top_touch": [], # left_parts[4].tolist(),
                                "fingertwo_palm_touch": [], # left_parts[5].tolist(),
                                "fingerthree_tip_touch": [], # left_parts[6].tolist(),
                                "fingerthree_top_touch": [], # left_parts[7].tolist(),
                                "fingerthree_palm_touch": [], # left_parts[8].tolist(),
                                "fingerfour_tip_touch": [], # left_parts[9].tolist(),
                                "fingerfour_top_touch": [], # left_parts[10].tolist(),
                                "fingerfour_palm_touch": [], # left_parts[11].tolist(),
                                "fingerfive_tip_touch": [], # left_parts[12].tolist(),
                                "fingerfive_top_touch": [], # left_parts[13].tolist(),
                                "fingerfive_middle_touch": [], # left_parts[14].tolist(),
                                "fingerfive_palm_touch": [], # left_parts[15].tolist(),
                                "palm_touch": [], # left_parts[16].tolist(),
                        }
                    }
                    
                    recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, tactiles=tactiles, ext_colors=ext_colors, ext_depths=ext_depths)
                    #recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, tactiles=tactiles)                  
            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program...")
    except Exception as e4:
        logger_mp.info("Exception")
        logger_mp.info(e4)

    finally:
        arm_ctrl.ctrl_dual_arm_go_home()
        # if hand_ctrl is not None:
        #     hand_ctrl.close()

        if args.ipc:
            ipc_server.stop()
        else:
            stop_listening()
            listen_keyboard_thread.join()

        tv_img_shm.close()
        tv_img_shm.unlink()
        tv_depth_shm.close()
        tv_depth_shm.unlink()
        tv_img_resized_shm.close()
        tv_img_resized_shm.unlink()

        if args.record:
            recorder.close()
        logger_mp.info("Finally, exiting program.")
        exit(0)