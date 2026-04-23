# RealSense Camera Viewer - Displays RGB and Depth frames side by side in a single OpenCV window

import cv2
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp
import logging
import time
import sys

import get_rs_info as get_rs

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

            align_to = rs.stream.color      
            self.align = rs.align(align_to)

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
            aligned_frames = self.align.process(frameset)
            color_frame = aligned_frames.get_color_frame()            
            depth_frame = aligned_frames.get_depth_frame()
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


class ImageServer:    
    """    Manages a single RealSense head camera and displays    
    the RGB and depth streams side by side in an OpenCV window.    
    """
    def __init__(self, config: dict):        
        """        
        Args:            config (dict): Configuration dictionary containing camera settings.                
        Expected keys:                    
        - head_camera_id_numbers (list[str]): Serial numbers; first entry is used.                    
        - head_camera_width  (int)                    
        - head_camera_height (int)                    
        - head_camera_fps    (int)                    
        - unit_test          (bool, optional): Enable performance metrics.
        """        
        self.config = config
        # Extract the first (and only) head camera serial number        
        serial_number = config["head_camera_id_numbers"][0]        
        width  = config.get("head_camera_width",  640)        
        height = config.get("head_camera_height", 480)        
        fps    = config.get("head_camera_fps",    30)
        self.camera = RealSenseCamera(            
            serial_number=serial_number,            
            width=width,
            height=height,            
            fps=fps,
        )
        # Optional performance metrics flag        
        self.unit_test: bool = config.get("unit_test", False)
        logging_mp.info(f"[ImageServer] Initialized with camera serial: {serial_number}")

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
    # ------------------------------------------------------------------    # Public interface
    # ------------------------------------------------------------------
    def show_process(self):        
        """        Main loop: captures frames from the head camera and displays        
        the RGB and depth images side by side in one OpenCV window.
        Press 'q' to quit.
        """        
        # Initialize the camera before entering the loop        
        if not self.camera.init_realsense():            
            logging_mp.error("[ImageServer] Camera initialization failed. Aborting show_process().")
            return
        logging_mp.info("[ImageServer] Starting viewer loop. Press 'q' to quit.")
        # Performance metric accumulators (used when unit_test=True)
        frame_count = 0        
        total_latency = 0.0        
        loop_start = time.time()
        elapsed = 0.0
        
        try:
            while True:                
                frame_ts = time.time()
                color_image, depth_image = self.camera.get_frame()

                if color_image is None or depth_image is None:                    
                    logging_mp.warning("[ImageServer] Skipping frame due to capture error.")
                    continue
                # Determine target dimensions from the color frame                
                target_h, target_w = color_image.shape[:2]
                # Convert depth uint16 → colorized 8-bit BGR                
                depth_colormap = self._depth_to_colormap(depth_image, target_h, target_w)
                # Concatenate color and depth frames horizontally                
                combined = np.hstack((color_image, depth_colormap))
                cv2.imshow("RGB | Depth", combined)
                # --- Optional performance metrics ---                
                if self.unit_test:                    
                    frame_count   += 1                    
                    frame_latency  = (time.time() - frame_ts) * 1000  # ms
                    total_latency += frame_latency
                    elapsed = time.time() - loop_start                    
                if elapsed >= 5.0:                        
                    avg_fps     = frame_count / elapsed                        
                    avg_latency = total_latency / frame_count                        
                    logging_mp.info(                            f"[ImageServer][UnitTest] "                            f"FPS: {avg_fps:.2f} | "                            f"Avg latency: {avg_latency:.2f} ms | "                            f"Frames: {frame_count}"                        )                        # Reset accumulators for the next window                        frame_count   = 0                        total_latency = 0.0                        loop_start    = time.time()
                # Exit on 'q' key press (waitKey in ms)                
                key = cv2.waitKey(1) & 0xFF
                #print(key)
                if key == ord('q'):
                    logging_mp.info("[ImageServer] 'q' pressed – exiting viewer loop.")
                    break

                if key == ord('s'):
                    cv2.imwrite("RGB_img.png", color_image)
                    cv2.imwrite("Depth_img.png", depth_image)
                    logging_mp.info("[ImageServer] Image saved.")

                time.sleep(0.01)

        except KeyboardInterrupt:            
            logging_mp.info("[ImageServer] KeyboardInterrupt received – exiting viewer loop.")

        finally:            
            self._close()

    def _close(self):        
        """Stop the camera and destroy all OpenCV windows."""        
        logging_mp.info("[ImageServer] Closing resources.")
        self.camera.stop()        
        cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":  

    ID = get_rs.get_rs_info()[0]

    config = {        
        # Only the first serial number is used by ImageServer        
        "head_camera_id_numbers": [f"{ID}"],        
        "head_camera_width":      640,
        "head_camera_height":     480,        
        "head_camera_fps":        30,        
        # Set to True to print FPS / latency metrics every 5 seconds        
        "unit_test":              False,    
        }
    server = ImageServer(config)    
    server.show_process()    
    sys.exit(0)