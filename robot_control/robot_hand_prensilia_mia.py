# =========================================================
# File: robot_hand_prensilia_mia.py
# Author: Giacomo Maccagni
# Date: 2026-03-26 10:20
# Description: This module is used to adopt Prensilia Mia HAnd with Unitree G1.
#              Here the 6 fingers captured from the XR device are then
#              reduced to 3 fingers for prensilia, that are thumb, index
#              and mri.
#              Finally all the values are published on cyclone dds topic.
#              You need to start the prensilia driver in c++
#              on board of Unitree G1 PC2, since the Prensilia Mia Hand are
#              connected with a RS485 to USBC Serial port.
# =========================================================

from teleop.robot_control.find_interface import find_interface_with_subnet
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from inspire_sdkpy import inspire_hand_defaut, inspire_dds

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.qos import Qos, Policy

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
import numpy as np
from enum import Enum, IntEnum
import threading
import time
from multiprocessing import Process, Array

import logging_mp
from teleop.robot_control.prensilia_mia_hand import HandMsg

logger_mp = logging_mp.getLogger(__name__)

Inspire_Num_Motors = 6
Inspire_Num_Hand_States = 6

# rt/prensilia_hand/ctrl/r
K_TOPIC_PRENSILIA_COMMAND_RIGHT = "prensilia_ttyUSB0"
K_TOPIC_PRENSILIA_COMMAND_LEFT = "prensilia_ttyUSB1"


class Prensilia_Controller:
    def __init__(self,
                 left_hand_array, right_hand_array,
                 fps=100.0,
                 Unit_Test=False,
                 simulation_mode=False):
        logger_mp.info("Initialize Prensilia_Controller...")
        self.fps = fps
        self.running = True
        self._stop_event = threading.Event()
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_Unit_Test)

        # DDS setup

        qos = Qos(
            Policy.Reliability.Reliable(max_blocking_time=1),
            Policy.Durability.TransientLocal  # no parentheses
        )

        participant = DomainParticipant(10)

        # RIGHT 
        topic = Topic(participant, K_TOPIC_PRENSILIA_COMMAND_RIGHT, HandMsg)
        self.publisher_right = Publisher(participant)
        self.writer_r = DataWriter(self.publisher_right, topic, qos=qos)

        # LEFT
        topic = Topic(participant, K_TOPIC_PRENSILIA_COMMAND_LEFT, HandMsg)
        self.publisher_left = Publisher(participant)
        self.writer_l = DataWriter(self.publisher_left, topic, qos=qos)

        # initialize handcmd publisher and handstate subscriber
        # self.hand_cmd_right_publisher = ChannelPublisher(K_TOPIC_PRENSILIA_COMMAND_RIGHT, HandMsg)
        # self.hand_cmd_right_publisher.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array = Array('d', Inspire_Num_Hand_States, lock=True) 
        self.right_hand_state_array = Array('d', Inspire_Num_Hand_States, lock=True)

        self.hand_control_process = threading.Thread(target=self.control_process,
                                                     args=(
                                                         left_hand_array,
                                                         right_hand_array,
                                                         self.left_hand_state_array,
                                                         self.right_hand_state_array))

        self.hand_control_process.daemon = True
        self.hand_control_process.start()

        logger_mp.info("Initialize Prensilia_Controller OK!\n")

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        Set current left, right hand motor state target q
        """
        # TODO Here only right hand is used

        # kRightHandThumbBend
        thumb = np.uint8((1-right_q_target[4]) * 255)  # Will wrap around if >255
        # kRightHandIndex
        # Compute value, keep negative numbers
        index = np.uint8((1-right_q_target[3]) * 255)
        # kRightHandMiddle
        middle = np.uint8((1-right_q_target[2]) * 255)
        reply_cmd_r = HandMsg(thumb, index, middle)

        # kLeftHandThumbBend
        thumb = np.uint8((1-left_q_target[4]) * 255)  # Will wrap around if >255
        # kLeftHandIndex
        # Compute value, keep negative numbers
        index = np.uint8((1-left_q_target[3]) * 255)
        # kLeftHandMiddle
        middle = np.uint8((1-left_q_target[2]) * 255)
        reply_cmd_l = HandMsg(thumb, index, middle)
        # reply_cmd_r = HandMsg(thumb, 0, 0)
        # print(f"index:{index}")
        # self.hand_cmd_right_publisher.Write(reply_cmd_r)
        self.writer_r.write(reply_cmd_r)
        self.writer_l.write(reply_cmd_l)
        # time.sleep(0.5)

        # logger_mp.info(f"Prensilia HandMsg:>{reply_cmd_r}")
        # print(right_q_target*1000)

    def control_process(self, left_hand_array, right_hand_array, left_hand_state_array, right_hand_state_array):
        left_q_target = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)

        # initialize inspire hand's cmd msg
        self.hand_msg = MotorCmds_()
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
                    left_hand_data = np.array(left_hand_array[:]).reshape(25, 3).copy()
                with right_hand_array.get_lock():
                    right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()

                if not np.all(right_hand_data == 0.0) and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15])): # if hand data has been initialized.
                    ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]

                    left_q_target = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
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
                            left_q_target[idx] = normalize(left_q_target[idx], 0.0, 1.7)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.7)
                        elif idx == 4:
                            left_q_target[idx] = normalize(left_q_target[idx], 0.0, 0.5)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 0.5)
                        elif idx == 5:
                            left_q_target[idx] = normalize(left_q_target[idx], -0.1, 1.3)
                            right_q_target[idx] = normalize(right_q_target[idx], -0.1, 1.3)

                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                # sleep_time = 2.0
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
            "HandTouch_left_subscriber",
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
