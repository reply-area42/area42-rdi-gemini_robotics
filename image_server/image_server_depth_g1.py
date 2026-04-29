import cv2
import zmq
import time
import struct
from collections import deque
import numpy as np
import pyrealsense2 as rs
import logging_mp
logger_mp = logging_mp.get_logger(__name__, level=logging_mp.DEBUG)


class RealSenseCamera:
    def __init__(self, img_shape, img_shape_depth, fps, serial_number=None, enable_depth=True):
        self.img_shape = img_shape
        self.fps = fps  
        self.serial_number = serial_number
        self.enable_depth = enable_depth
        self.img_shape_depth = img_shape_depth
        self.align = rs.align(rs.stream.color) #allinea i frame depth a quelli color
        self.init_realsense()

    def init_realsense(self):
        self.pipeline = rs.pipeline() #inizializza da pipeline, che è un wrapper per la gestione del flusso di dati dalla camera
        config = rs.config()
        if self.serial_number:
            config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.color, self.img_shape[1], self.img_shape[0], rs.format.bgr8, self.fps)
        if self.enable_depth:
            config.enable_stream(rs.stream.depth,  self.img_shape_depth[1], self.img_shape_depth[0], rs.format.z16, self.fps)
        profile = self.pipeline.start(config)
        self._device = profile.get_device()
        if self._device is None:
            logger_mp.error("[Image Server] RealSense device not found")
        elif self.enable_depth:
            depth_sensor = self._device.first_depth_sensor()
            if depth_sensor is None:
                logger_mp.error("[Image Server] RealSense depth sensor not found")
            else:
                self.g_depth_scale = depth_sensor.get_depth_scale()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame() if self.enable_depth else None
        if not color_frame:
            return None, None
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
        return color_image, depth_image

    def release(self):
        self.pipeline.stop()


class OpenCVCamera:
    def __init__(self, device_id, img_shape, fps):
        self.id = device_id
        self.fps = fps
        self.img_shape = img_shape
        self.cap = cv2.VideoCapture(self.id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_shape[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._can_read_frame():
            logger_mp.error(f"[Image Server] Camera {self.id} cannot read frames")
            self.release()

    def _can_read_frame(self):
        success, _ = self.cap.read()
        return success

    def get_frame(self):
        ret, img = self.cap.read()
        if not ret:
            return None
        return img

    def release(self):
        self.cap.release()


class ImageServer:
    def __init__(self, config, port=5555, Unit_Test=False):
        logger_mp.info(config)
        self.fps = config.get('fps', 30)
        self.head_camera_type = config.get('head_camera_type', 'opencv')
        self.head_image_shape = config.get('head_camera_image_shape', [480, 640])
        self.depth_image_shape = config.get('depth_image_shape', [480, 640])  # [H, W]
        self.enable_depth = config.get('enable_depth', True)
        self.head_camera_id_numbers = config.get('head_camera_id_numbers', [0])
        self.wrist_camera_type = config.get('wrist_camera_type', None)
        self.wrist_image_shape = config.get('wrist_camera_image_shape', [480, 640])
        self.wrist_camera_id_numbers = config.get('wrist_camera_id_numbers', None)
        self.port = port
        self.Unit_Test = Unit_Test

        # Initialize cameras
        self.head_cameras = self._init_cameras(self.head_camera_type, self.head_camera_id_numbers, self.head_image_shape, self.depth_image_shape, self.enable_depth)
        self.wrist_cameras = self._init_cameras(self.wrist_camera_type, self.wrist_camera_id_numbers, self.wrist_image_shape)
        #self.wrist_cameras = None


        # ZMQ
        self.context = zmq.Context()
        self.socket_color = self.context.socket(zmq.PUB)
        self.socket_color.bind(f"tcp://*:{self.port}")
        self.socket_depth = self.context.socket(zmq.PUB)
        self.socket_depth.bind(f"tcp://*:{self.port+1}")

        if self.Unit_Test:
            self._init_performance_metrics()
            
        for cam in self.head_cameras:
            if isinstance(cam, OpenCVCamera):
                logger_mp.info(f"[Image Server] Head camera {cam.id} resolution: {cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} x {cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            elif isinstance(cam, RealSenseCamera):
                logger_mp.info(f"[RealSense Image Server] Head camera {cam.serial_number} resolution: {cam.img_shape[0]} x {cam.img_shape[1]}")
                if cam.enable_depth:
                    logger_mp.info(f"[RealSense Image Server] Depth camera {cam.serial_number} resolution: {cam.img_shape_depth[0]} x {cam.img_shape_depth[1]}")
            else:
                logger_mp.warning("[Image Server] Unknown camera type in head_cameras.")

        logger_mp.info("[Image Server] Started, waiting for clients...")

    def _init_cameras(self, cam_type, cam_ids, img_shape, depth_shape, enable_depth):
        cameras = []
        if not cam_type or not cam_ids:
            return cameras
        for cam_id in cam_ids:
            if cam_type == 'opencv':
                cameras.append(OpenCVCamera(cam_id, img_shape, self.fps))
            elif cam_type == 'realsense':
                cameras.append(RealSenseCamera(img_shape, depth_shape, self.fps, serial_number=cam_id, enable_depth=enable_depth))
            else:
                logger_mp.warning(f"[Image Server] Unsupported camera type: {cam_type}")
        return cameras

    def _init_performance_metrics(self):
        self.frame_count = 0
        self.time_window = 1.0
        self.frame_times = deque()
        self.start_time = time.time()

    def _update_performance_metrics(self, t):
        self.frame_times.append(t)
        while self.frame_times and self.frame_times[0] < t - self.time_window:
            self.frame_times.popleft()
        self.frame_count += 1

    def _print_performance_metrics(self, t):
        if self.frame_count % 30 == 0:
            fps = len(self.frame_times)/self.time_window
            logger_mp.info(f"[Image Server] FPS: {fps:.2f}, Total frames: {self.frame_count}")

    def _close(self):
        for cam in self.head_cameras + self.wrist_cameras:
            cam.release()
        self.socket_color.close()
        self.socket_depth.close()
        self.context.term()
        logger_mp.info("[Image Server] Closed.")

    def send_process(self):
        try:
            while True:
                # Head cameras
                head_frames = []
                depth_frames = []
                for cam in self.head_cameras:
                    if self.head_camera_type == 'opencv':
                        color_image = cam.get_frame()
                        if color_image is None:
                            logger_mp.error("[Image Server] Head camera read error")
                            break
                        head_frames.append(color_image)
                    else:  # Realsense
                        color_image, depth_image = cam.get_frame()
                        if color_image is None or (depth_image is None and cam.enable_depth):
                            logger_mp.error("[Image Server] Head camera read error")
                            break

                        depth_gray = depth_image.astype(np.uint16)  # mantieni dati raw

                        head_frames.append(color_image)
                        depth_frames.append(depth_gray)

                if not head_frames:
                    continue
                if len(head_frames) != len(self.head_cameras): #se tutte le camere non hanno letto correttamente, non inviare nulla e loggare l'errore
                    logger_mp.error(f"[Image Server] Head frames count mismatch: expected {len(self.head_cameras)}, got {len(head_frames)}")
                    break
                head_color = cv2.hconcat(head_frames) #concatena le immagini delle camere head in orizzontale
                # Wrist cameras
                if self.wrist_cameras:
                    wrist_frames = []
                    for cam in self.wrist_cameras:
                        if self.wrist_camera_type == 'opencv':
                            color_image = cam.get_frame()
                            if color_image is None:
                                logger_mp.error("[Image Server] Wrist camera read error")
                                break
                            wrist_frames.append(color_image)
                        else:  # Realsense
                            color_image, depth_image = cam.get_frame()
                            if color_image is None or (depth_image is None and cam.enable_depth):
                                logger_mp.error("[Image Server] Wrist camera read error")
                                break
                            wrist_frames.append(color_image)

                    # Concatenate head and wrist
                    full_color = cv2.hconcat(head_frames + wrist_frames) if wrist_frames else cv2.hconcat(head_frames)
                else:
                    full_color = head_color
                # Encode color JPEG
                ret_color, buffer_color = cv2.imencode('.jpg', full_color) #comprime l'immagine concatenata in formato JPEG, restituendo un array di byte
                if not ret_color:
                    logger_mp.error("[Image Server] Failed to encode color frame")
                    continue
                self.socket_color.send(buffer_color.tobytes())
                # print(f"socket_color.send {buffer_color.tobytes()[:10]}")

                # Send depth raw (first head camera)
                if depth_frames:
                    depth_to_send = depth_frames[0]
                    target_h, target_w = self.depth_image_shape
                    if depth_to_send.shape[:2] != (target_h, target_w):
                        depth_to_send = cv2.resize(
                            depth_to_send,
                            (target_w, target_h),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(np.uint16)
                    self.socket_depth.send(depth_to_send.tobytes())
                    # print(f"socket_depth.send {depth_to_send.tobytes()[:10]}")

                # Performance metrics
                if self.Unit_Test:
                    t = time.time()
                    self._update_performance_metrics(t)
                    self._print_performance_metrics(t)

        except KeyboardInterrupt:
            logger_mp.warning("[Image Server] Interrupted by user")
        finally:
            self._close()


if __name__ == "__main__":
    config = {
        'fps': 30, # 6, 15, 30, 60, 90
        'head_camera_type': 'realsense',
        'head_camera_image_shape': [480, 640],
        'depth_image_shape': [270, 480], #1280 x 720, 848 x 480, 640 x 480, 640 x 360, 480 x 270, 424 x 240
        'head_camera_id_numbers': ["243122076269"],
    }

    server = ImageServer(config, Unit_Test=False)
    server.send_process()
