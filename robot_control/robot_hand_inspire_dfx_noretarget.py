from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from inspire_sdkpy import inspire_hand_defaut,inspire_dds
from teleop.robot_control.find_interface import find_interface_with_subnet
 
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
import numpy as np
from enum import IntEnum
import threading
import time
from multiprocessing import Process, Array
 
import logging_mp
logger_mp = logging_mp.getLogger(__name__)
 
Inspire_Num_Motors = 6
kTopicInspireCommandRight = "rt/inspire_hand/ctrl/r"
kTopicInspireStateRight = "rt/inspire_hand/state/r"

kTopicInspireCommandLeft = "rt/inspire_hand/ctrl/l"
kTopicInspireStateLeft = "rt/inspire_hand/state/l"
 
class Inspire_Controller:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock = None, dual_hand_state_array = None,
                       dual_hand_action_array = None, fps = 100.0, Unit_Test = False, simulation_mode = False):
        logger_mp.info("Initialize Inspire_Controller...")
        self.fps = fps
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

        self.HandState_subscriber = ChannelSubscriber(kTopicInspireStateRight, MotorStates_)
        self.HandState_subscriber.Init()
 
        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', Inspire_Num_Motors, lock=True)  
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)
 
        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()
 
        # while True:
        #     if any(self.right_hand_state_array): # any(self.left_hand_state_array) and
        #         break
        #     time.sleep(1)
        #     logger_mp.warning("[Inspire_Controller] Waiting to subscribe dds...")
        # logger_mp.info("[Inspire_Controller] Subscribe dds ok.")
 
        hand_control_process = Process(target=self.control_process, args=(left_hand_array, right_hand_array,  self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array))
        hand_control_process.daemon = True
        hand_control_process.start()
 
        logger_mp.info("Initialize Inspire_Controller OK!\n")
 
    def _subscribe_hand_state(self):
        while True:
            hand_msg  = self.HandState_subscriber.Read()
            if hand_msg is not None:
                for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
                    self.left_hand_state_array[idx] = hand_msg.states[id].q
                for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
                    self.right_hand_state_array[idx] = hand_msg.states[id].q
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
                              dual_hand_data_lock = None, dual_hand_state_array = None, dual_hand_action_array = None):
        self.running = True
 
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
            while self.running:
                start_time = time.time()
                # get dual hand state
                with left_hand_array.get_lock():
                    left_q_target  = left_hand_array
                with right_hand_array.get_lock():
                    right_q_target = right_hand_array
 
                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                #sleep_time = 2.0
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Inspire_Controller has been closed.")
 
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