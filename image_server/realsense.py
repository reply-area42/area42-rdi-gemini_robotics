# RealSense Camera Viewer - Displays RGB and Depth frames side by side in a single OpenCV window

import cv2
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp
import logging
import time
import sys

# Configure multiprocessing-safe logger
logging_mp = mp.log_to_stderr()
logging_mp.setLevel(logging.INFO)


class RealSenseCamera:
    """Manages a single Intel RealSense camera stream."""

    def __init__(self, serial_number: str, width: int = 640, height: int = 480, fps: int = 30):
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.config = None

    def init_realsense(self) -> bool:
        """Initialize the RealSense pipeline and start streaming."""
        try:
            ctx = rs.context()
            for dev in ctx.query_devices():
                if dev.get_info(rs.camera_info.serial_number) == self.serial_number:
                    dev.hardware_reset()
                    time.sleep(3)  # attendi il riavvio
                    break

            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # Enable device by serial number
            self.config.enable_device(self.serial_number)

            # Enable color and depth streams
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

            self.pipeline.start(self.config)
            logging_mp.info(f"[RealSenseCamera] Started camera with serial: {self.serial_number}")
            return True

        except Exception as e:
            logging_mp.error(f"[RealSenseCamera] Failed to initialize camera {self.serial_number}: {e}")
            return False
    def get_frame(self):        
        """        Capture a single frameset from the camera.

        Returns:            tuple: (color_image np.ndarray uint8, depth_image np.ndarray uint16)                   or (None, None) on failure.
        """        
        try:            
            frameset = self.pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frameset.get_color_frame()            
            depth_frame = frameset.get_depth_frame()
            if not color_frame or not depth_frame:                
                logging_mp.warning("[RealSenseCamera] Incomplete frameset received.")
                return None, None
            color_image = np.asanyarray(color_frame.get_data())   # (H, W, 3) uint8 BGR            
            depth_image = np.asanyarray(depth_frame.get_data())   # (H, W)    uint16
            return color_image, depth_image

        except Exception as e:            
            logging_mp.error(f"[RealSenseCamera] get_frame error: {e}")            
            return None, None
        
    def stop(self):        
        """Stop the RealSense pipeline gracefully."""        
        if self.pipeline is not None:            
            try:                
                self.pipeline.stop()                
                logging_mp.info(f"[RealSenseCamera] Stopped camera {self.serial_number}")            
            except Exception as e:                
                logging_mp.error(f"[RealSenseCamera] Error stopping camera {self.serial_number}: {e}")
            finally:                
                self.pipeline = None


    @staticmethod
    def _depth_to_colormap(depth_image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        # Normalize to 0-255 (let OpenCV allocate the output buffer)
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply jet colormap → (H, W, 3) BGR uint8
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # Resize only when dimensions differ
        h, w = depth_colormap.shape[:2]
        if h != target_h or w != target_w:
            depth_colormap = cv2.resize(depth_colormap, (target_w, target_h))

        return depth_colormap


