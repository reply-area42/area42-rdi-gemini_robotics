from robot_control.find_interface import find_interface_with_subnet
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from inspire_sdkpy import inspire_hand_defaut, inspire_dds
 
from robot_control.hand_retargeting import HandRetargeting, HandType
import numpy as np
from enum import Enum, IntEnum
import threading
import time
from multiprocessing import Process, Array
 
import logging_mp
logger_mp = logging_mp.getLogger(__name__)

Inspire_Num_Motors = 6
Inspire_Num_Hand_States = 6
kTopicInspireCommandRight = "rt/inspire_hand/ctrl/r"
kTopicInspireStateRight = "rt/inspire_hand/state/r"

kTopicInspireCommandLeft = "rt/inspire_hand/ctrl/l"
kTopicInspireStateLeft = "rt/inspire_hand/state/l"

Inspire_Num_Touch_Sensors = 1062
kTopicInspireTouchRight = "rt/inspire_hand/touch/r"
kTopicInspireTouchLeft = "rt/inspire_hand/touch/l"
 
class Inspire_Controller:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock = None, dual_hand_state_array = None,
                       dual_hand_action_array = None, dual_hand_right_touch_array = None, dual_hand_left_touch_array = None, fps = 100.0, Unit_Test = False, simulation_mode = False):
        logger_mp.info("Initialize Inspire_Controller...")
        self.fps = fps
        self.running = True
        self._stop_event = threading.Event()
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_Unit_Test)
 
        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            interface = find_interface_with_subnet()
            if interface is None:
                logger_mp.warning("No interface found with IP 192.168.123.*, using default interface for dds communication.")
                ChannelFactoryInitialize(0)
            else:
                ChannelFactoryInitialize(0, interface)
 
        # initialize handcmd publisher and handstate subscriber
        
        self.HandCmb_right_publisher = ChannelPublisher(kTopicInspireCommandRight, inspire_dds.inspire_hand_ctrl)
        self.HandCmb_right_publisher.Init()
    
        self.HandCmb_left_publisher = ChannelPublisher(kTopicInspireCommandLeft, inspire_dds.inspire_hand_ctrl)
        self.HandCmb_left_publisher.Init()

        self.HandState_right_subscriber = ChannelSubscriber(kTopicInspireStateRight, inspire_dds.inspire_hand_state)
        self.HandState_right_subscriber.Init()

        # SINISTRA
        # self.HandState_left_subscriber = ChannelSubscriber(kTopicInspireStateLeft, inspire_dds.inspire_hand_state)
        # self.HandState_left_subscriber.Init()

        # Create a subscriber to subscribe the data defined in UserData class
        self.HandTouch_right_subscriber = ChannelSubscriber(kTopicInspireTouchRight, inspire_dds.inspire_hand_touch)
        self.HandTouch_right_subscriber.Init()

        # SINISTRA
        # self.HandTouch_left_subscriber = ChannelSubscriber(kTopicInspireTouchLeft, inspire_dds.inspire_hand_touch)
        # self.HandTouch_left_subscriber.Init()
 
        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', Inspire_Num_Hand_States, lock=True)  
        self.right_hand_state_array = Array('d', Inspire_Num_Hand_States, lock=True)

        self.left_hand_touch_array  = Array('d', Inspire_Num_Touch_Sensors, lock=True)  
        self.right_hand_touch_array = Array('d', Inspire_Num_Touch_Sensors, lock=True)
 
        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_touch_thread = threading.Thread(target=self._subscribe_hand_touch)

        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()
        self.subscribe_touch_thread.daemon = True
        self.subscribe_touch_thread.start()
        # print("State thread alive:", self.subscribe_state_thread.is_alive())
        # print("Touch thread alive:", self.subscribe_touch_thread.is_alive())


        # while True:
        #     if any(self.right_hand_state_array): # any(self.left_hand_state_array) and
        #         break
        #     time.sleep(1)
        #     logger_mp.warning("[Inspire_Controller] Waiting to subscribe dds...")
        # logger_mp.info("[Inspire_Controller] Subscribe dds ok.")
 
        # hand_control_process = Process(target=self.control_process, args=(left_hand_array, right_hand_array,  self.left_hand_state_array, self.right_hand_state_array,
        #                                                                   dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, dual_hand_right_touch_array, dual_hand_left_touch_array))
        # hand_control_process.daemon = True
        # hand_control_process.start()

        self.hand_control_process = threading.Thread(target=self.control_process, args=(left_hand_array, right_hand_array,  self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, dual_hand_right_touch_array, dual_hand_left_touch_array))

        self.hand_control_process.daemon = True
        self.hand_control_process.start()
 
        logger_mp.info("Initialize Inspire_Controller OK!\n")

        self.FINGER_TOUCH_FIELDS = {
        "one":   ["tip", "top", "palm"],
        "two":   ["tip", "top", "palm"],
        "three": ["tip", "top", "palm"],
        "four":  ["tip", "top", "palm"],
        "five":  ["tip", "top", "middle", "palm"],  # dito 5 ha anche "middle"
        }
    def _extract_touch_data(self, touch_msg) -> list:
        """Estrae i dati tattili da un messaggio in una lista piatta."""
        data = []
        for finger, parts in self.FINGER_TOUCH_FIELDS.items():
            for part in parts:
                data.extend(getattr(touch_msg, f"finger{finger}_{part}_touch"))
        data.extend(touch_msg.palm_touch)
        return data
    
    def _write_touch_to_array(self, data: list, shared_array):
        """Scrive la lista nei dati condivisi in modo thread-safe."""
        with shared_array.get_lock():
            shared_array[:len(data)] = data

    def _subscribe_hand_state(self):
        
        while self.running and not self._stop_event.is_set():
            r_hand_msg  = self.HandState_right_subscriber.Read()
            if r_hand_msg is not None:
                # print(r_hand_msg)
                # print(r_hand_msg.pos_act)
                for i in range(6):
                    self.right_hand_state_array[i] = r_hand_msg.pos_act[i]


            # print("right hand state:", self.right_hand_state_array[:]) # debug

            # SINISTRA
            # l_hand_msg  = self.HandState_right_subscriber.Read()
            # if l_hand_msg is not None:
            #     # print(l_hand_msg)
            #     # print(l_hand_msg.pos_act)
            #     for i in range(6):
            #         self.left_hand_state_array[i] = l_hand_msg.pos_act[i]

       
            # print("right hand state:", self.right_hand_state_array[:]) # debug

            time.sleep(0.002)

    def _subscribe_hand_touch(self):

        while self.running and not self._stop_event.is_set():
            # print("PRIMA DELLA READ")
            touch_msg_r = self.HandTouch_right_subscriber.Read()
            # touch_msg_l = self.HandTouch_left_subscriber.Read()  # SINISTRA

            if touch_msg_r is not None:
                data_r = self._extract_touch_data(touch_msg_r)
                self._write_touch_to_array(data_r, self.right_hand_touch_array)

            # SINISTRA 
            # if touch_msg_l is not None:
            #     data_l = self._extract_touch_data(touch_msg_l)
            #     self._write_touch_to_array(data_l, self.left_hand_touch_array)
            # print("right hand touch:", self.right_hand_touch_array[:10]) # debug
            time.sleep(0.002)
            
                
 
    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        Set current left, right hand motor state target q
        """
        reply_cmd_l = inspire_hand_defaut.get_inspire_hand_ctrl()
        reply_cmd_l.mode=0b0001
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):             
            reply_cmd_l.angle_set[id] = int(left_q_target[idx]*1000)

        reply_cmd_r = inspire_hand_defaut.get_inspire_hand_ctrl()
        reply_cmd_r.mode=0b0001        
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):             
            reply_cmd_r.angle_set[id] = int(right_q_target[idx]*1000)

        self.HandCmb_right_publisher.Write(reply_cmd_r)
        self.HandCmb_left_publisher.Write(reply_cmd_l)

        # logger_mp.debug("hand ctrl publish ok.")
        #print(right_q_target*1000)


    def control_process(self, left_hand_array, right_hand_array, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array = None, dual_hand_action_array = None, 
                              dual_hand_right_touch_array = None, dual_hand_left_touch_array = None):
        left_q_target  = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)

        # initialize inspire hand's cmd msg
        self.hand_msg  = MotorCmds_()
        self.hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(Inspire_Right_Hand_JointIndex) + len(Inspire_Left_Hand_JointIndex))]
 
        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0
 
        try:
            while self.running and not self._stop_event.is_set():
                start_time = time.time()
                # get dual hand state
                with left_hand_array.get_lock():
                    left_hand_data  = np.array(left_hand_array[:]).reshape(25, 3).copy()
                with right_hand_array.get_lock():
                    right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()
 
                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))
                # print("current hand state:", state_data) # debug
                if not np.all(right_hand_data == 0.0) and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15])): # if hand data has been initialized.
                    ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]
 
                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]
 
                    # In website https://support.unitree.com/home/en/G1_developer/inspire_dfx_dexterous_hand, you can find
                    #     In the official document, the angles are in the range [0, 1] ==> 0.0: fully closed  1.0: fully open
                    # The q_target now is in radians, ranges:
                    #     - idx 0~3: 0~1.7 (1.7 = closed)
                    #     - idx 4:   0~0.5
                    #     - idx 5:  -0.1~1.3
                    # We normalize them using (max - value) / range
                    def normalize(val, min_val, max_val):
                        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)
 
                    for idx in range(Inspire_Num_Motors):
                        if idx <= 3:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 1.7)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.7)
                        elif idx == 4:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 0.5)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 0.5)
                        elif idx == 5:
                            left_q_target[idx]  = normalize(left_q_target[idx], -0.1, 1.3)
                            right_q_target[idx] = normalize(right_q_target[idx], -0.1, 1.3)
 
                # get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))   
                # print("current hand action:", action_data) # debug 
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                # get dual hand touch
                    
                if dual_hand_right_touch_array and dual_hand_left_touch_array:
                    with dual_hand_data_lock:
                        dual_hand_right_touch_array[:] = self.right_hand_touch_array
                        #dual_hand_left_touch_array[:] = self.left_hand_touch_array
                    
                # print("current hand touch:", dual_hand_right_touch_array[:10]) # debug       
 
                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                #sleep_time = 2.0
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Inspire_Controller has been closed.")

    def close(self, join_timeout=1.0):
        """Stop Inspire controller threads and close DDS channels."""
        self.running = False
        self._stop_event.set()

        # Closing DDS entities helps unblock Read() in subscriber threads.
        for obj_name in (
            "HandState_subscriber",
            "HandTouch_right_subscriber",
            "HandCmb_right_publisher",
            "HandCmb_left_publisher",
        ):
            obj = getattr(self, obj_name, None)
            if obj is not None:
                try:
                    obj.Close()
                except Exception as exc:
                    logger_mp.debug(f"Failed closing {obj_name}: {exc}")

        for thread_name in ("hand_control_process", "subscribe_state_thread", "subscribe_touch_thread"):
            thread = getattr(self, thread_name, None)
            if thread is not None and thread.is_alive():
                thread.join(timeout=join_timeout)
                if thread.is_alive():
                    logger_mp.warning(f"{thread_name} did not exit within {join_timeout}s.")
 
# Update hand state, according to the official documentation, https://support.unitree.com/home/en/G1_developer/inspire_dfx_dexterous_hand
# the state sequence is as shown in the table below
# ┌──────┬───────┬──────┬────────┬────────┬────────────┬────────────────┬───────┬──────┬────────┬────────┬────────────┬────────────────┐
# │ Id   │   0   │  1   │   2    │   3    │     4      │       5        │   6   │  7   │   8    │   9    │    10      │       11       │
# ├──────┼───────┼──────┼────────┼────────┼────────────┼────────────────┼───────┼──────┼────────┼────────┼────────────┼────────────────┤
# │      │                    Right Hand                                │                   Left Hand                                  │
# │Joint │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │
# └──────┴───────┴──────┴────────┴────────┴────────────┴────────────────┴───────┴──────┴────────┴────────┴────────────┴────────────────┘
class Inspire_Right_Hand_JointIndex(IntEnum):
    kRightHandPinky = 0
    kRightHandRing = 1
    kRightHandMiddle = 2
    kRightHandIndex = 3
    kRightHandThumbBend = 4
    kRightHandThumbRotation = 5
 
class Inspire_Left_Hand_JointIndex(IntEnum):
    kLeftHandPinky = 6
    kLeftHandRing = 7
    kLeftHandMiddle = 8
    kLeftHandIndex = 9
    kLeftHandThumbBend = 10
    kLeftHandThumbRotation = 11
