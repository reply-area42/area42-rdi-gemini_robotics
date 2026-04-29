import cv2
import zmq
import numpy as np
import time
import struct
import threading
from collections import deque
from multiprocessing import shared_memory
import logging_mp

import position_calculator as pc

logger_mp = logging_mp.getLogger(__name__)

# CAMERA INTRINSICS
fx = 605.3919677734375
fy = 605.1267700195312
cx = 316.6646728515625
cy = 252.62767028808594
width = 640
height = 480


class ImageClient_depth:
    def __init__(
        self,
        tv_img_shape=None,
        tv_img_shm_name=None,
        tv_img_shape_resize=None,
        tv_img_resized_shm_name=None,   
        wrist_img_shape=None,
        wrist_img_shm_name=None,
        image_show=False,
        server_address="192.168.123.164",
        port=5555,
        Unit_Test=False,
        tv_depth_shape=None,
        tv_depth_shm_name=None,
    ):
        """
        Color stream is read from tcp://<server_address>:<port>.
        Depth stream is read from tcp://<server_address>:<port+1> as raw uint16 bytes.
        """
        self.running = True
        self._image_show = image_show
        self._server_address = server_address
        self._color_port = port
        self._depth_port = port + 1
        self.tv_img_shape_resize = tv_img_shape_resize
        self.tv_img_shape = tv_img_shape
        self.wrist_img_shape = wrist_img_shape
        self.tv_depth_shape = tv_depth_shape

        self.tv_enable_shm = False
        if self.tv_img_shape_resize is not None and tv_img_resized_shm_name is not None:
            self.tv_image_resized_shm = shared_memory.SharedMemory(name=tv_img_resized_shm_name)
            self.tv_img_resized_array = np.ndarray(tv_img_shape_resize, dtype=np.uint8, buffer=self.tv_image_resized_shm.buf)
            self.tv_enable_resize_shm = True
        if self.tv_img_shape is not None and tv_img_shm_name is not None:
            self.tv_image_shm = shared_memory.SharedMemory(name=tv_img_shm_name)
            self.tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=self.tv_image_shm.buf)
            self.tv_enable_shm = True

        self.wrist_enable_shm = False
        if self.wrist_img_shape is not None and wrist_img_shm_name is not None:
            self.wrist_image_shm = shared_memory.SharedMemory(name=wrist_img_shm_name)
            self.wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=self.wrist_image_shm.buf)
            self.wrist_enable_shm = True

        self.tv_depth_enable_shm = False
        if self.tv_depth_shape is not None and tv_depth_shm_name is not None:
            self.tv_depth_shm = shared_memory.SharedMemory(name=tv_depth_shm_name)
            self.tv_depth_array = np.ndarray(self.tv_depth_shape, dtype=np.uint16, buffer=self.tv_depth_shm.buf)
            self.tv_depth_enable_shm = True

        self._enable_performance_eval = Unit_Test
        if self._enable_performance_eval:
            self._init_performance_metrics()

    def _init_performance_metrics(self):
        self._frame_count = 0
        self._last_frame_id = -1
        self._time_window = 1.0
        self._frame_times = deque()
        self._latencies = deque()
        self._lost_frames = 0
        self._total_frames = 0

    def _update_performance_metrics(self, timestamp, frame_id, receive_time):
        latency = receive_time - timestamp
        self._latencies.append(latency)

        while self._latencies and self._frame_times and self._latencies[0] < receive_time - self._time_window:
            self._latencies.popleft()

        self._frame_times.append(receive_time)
        while self._frame_times and self._frame_times[0] < receive_time - self._time_window:
            self._frame_times.popleft()

        expected_frame_id = self._last_frame_id + 1 if self._last_frame_id != -1 else frame_id
        if frame_id != expected_frame_id:
            lost = frame_id - expected_frame_id
            if lost < 0:
                logger_mp.info(f"[Image Client] Received out-of-order frame ID: {frame_id}")
            else:
                self._lost_frames += lost
                logger_mp.warning(
                    f"[Image Client] Detected lost frames: {lost}, "
                    f"Expected frame ID: {expected_frame_id}, Received frame ID: {frame_id}"
                )
        self._last_frame_id = frame_id
        self._total_frames = frame_id + 1
        self._frame_count += 1

    def _print_performance_metrics(self):
        if self._frame_count % 30 != 0:
            return

        real_time_fps = len(self._frame_times) / self._time_window if self._time_window > 0 else 0

        if self._latencies:
            avg_latency = sum(self._latencies) / len(self._latencies)
            max_latency = max(self._latencies)
            min_latency = min(self._latencies)
            jitter = max_latency - min_latency
        else:
            avg_latency = max_latency = min_latency = jitter = 0

        lost_frame_rate = (self._lost_frames / self._total_frames) * 100 if self._total_frames > 0 else 0

        logger_mp.info(
            f"[Image Client] Real-time FPS: {real_time_fps:.2f}, "
            f"Avg Latency: {avg_latency*1000:.2f} ms, "
            f"Max Latency: {max_latency*1000:.2f} ms, "
            f"Min Latency: {min_latency*1000:.2f} ms, "
            f"Jitter: {jitter*1000:.2f} ms, "
            f"Lost Frame Rate: {lost_frame_rate:.2f}%"
        )

    def _decode_depth(self, depth_bytes, color_shape=None):
        if depth_bytes is None:
            return None

        np_depth = np.frombuffer(depth_bytes, dtype=np.uint16)

        if self.tv_depth_shape is not None:
            expected_size = int(np.prod(self.tv_depth_shape)) 
            if np_depth.size != expected_size:
                logger_mp.warning(
                    f"[Image Client] Depth size mismatch: got {np_depth.size}, expected {expected_size}."
                )
                # If depth SHM is enabled, shape must match exactly.
                if self.tv_depth_enable_shm:
                    return None
                # Otherwise keep going and try to infer the real depth shape.
            else:
                return np_depth.reshape(self.tv_depth_shape)

        if color_shape is not None:
            h, w = color_shape[:2]
            if np_depth.size == h * w:
                return np_depth.reshape((h, w))

        for h, w in ((720, 1280), (1080, 1920), (480, 640)):
            if np_depth.size == h * w:
                return np_depth.reshape((h, w))

        logger_mp.warning(
            "[Image Client] Unable to infer depth shape. "
            "Pass tv_depth_shape=(H, W) to enable depth decode/shm copy."
        )
        return None

    def _close(self):
        if hasattr(self, "_color_socket"):
            self._color_socket.close()
        if hasattr(self, "_depth_socket"):
            self._depth_socket.close()
        if hasattr(self, "_context"):
            self._context.term()

        if self._image_show:
            cv2.destroyAllWindows()

        if self.tv_enable_shm:
            self.tv_image_shm.close()
        if self.wrist_enable_shm:
            self.wrist_image_shm.close()
        if self.tv_depth_enable_shm:
            self.tv_depth_shm.close()

        logger_mp.info("Image client has been closed.")

    def receive_process(self):
        if self._image_show and threading.current_thread() is not threading.main_thread():
            logger_mp.warning(
                "[Image Client] image_show=True but not in main thread; disabling cv2.imshow to avoid Qt thread errors."
            )
            self._image_show = False

        self._context = zmq.Context()

        self._color_socket = self._context.socket(zmq.SUB)
        self._color_socket.connect(f"tcp://{self._server_address}:{self._color_port}")
        self._color_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self._depth_socket = self._context.socket(zmq.SUB)
        self._depth_socket.connect(f"tcp://{self._server_address}:{self._depth_port}")
        self._depth_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        logger_mp.info(
            f"Image client started. Color: tcp://{self._server_address}:{self._color_port}, "
            f"Depth: tcp://{self._server_address}:{self._depth_port}"
        )         

        try:
            while self.running:

                color_message = self._color_socket.recv()
                receive_time = time.time()

                if self._enable_performance_eval:
                    header_size = struct.calcsize("dI")
                    try:
                        header = color_message[:header_size]
                        jpg_bytes = color_message[header_size:]
                        timestamp, frame_id = struct.unpack("dI", header)
                    except struct.error as e:
                        logger_mp.warning(f"[Image Client] Error unpacking header: {e}, discarding message.")
                        continue
                else:
                    jpg_bytes = color_message

                np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
                current_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                if current_image is None:
                    logger_mp.warning("[Image Client] Failed to decode color image.")
                    continue
                # else:
                #     print("RGB shape: ", current_image.shape[:2])
                current_image_resized = cv2.resize(current_image, (self.tv_img_shape_resize[1], self.tv_img_shape_resize[0]))
                depth_message = self._depth_socket.recv()
                

                depth_image = self._decode_depth(depth_message, color_shape=current_image.shape)
                # if depth_image is not None:
                #     print("Depth shape: ", depth_image.shape)
                if self.tv_enable_resize_shm:
                    np.copyto(self.tv_img_resized_array, np.array(current_image_resized[:, :self.tv_img_shape_resize[1]]))
                if self.tv_enable_shm:
                    np.copyto(self.tv_img_array, np.array(current_image[:, :self.tv_img_shape[1]]))

                if self.wrist_enable_shm:
                    np.copyto(self.wrist_img_array, np.array(current_image[:, -self.wrist_img_shape[1]:]))

                if self.tv_depth_enable_shm and depth_image is not None:
                    np.copyto(self.tv_depth_array, depth_image)

                if self._image_show:
                    height, width = current_image.shape[:2]
                    resized_image = cv2.resize(current_image, (width // 2, height // 2))
                    cv2.imshow("Image Client Stream", resized_image)

                    if depth_image is not None:
                        depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
                        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                        cv2.imshow("Depth Stream", depth_vis)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.running = False


                # if self._enable_performance_eval:
                #     self._update_performance_metrics(timestamp, frame_id, receive_time)
                #     self._print_performance_metrics()

        except KeyboardInterrupt:
            logger_mp.info("Image client interrupted by user.")
        except Exception as e:
            logger_mp.warning(f"[Image Client] An error occurred while receiving data: {e}")
        finally:
            self._close()


if __name__ == "__main__":
    # Example
    # client = ImageClient_depth(
    #     image_show=True,
    #     server_address="127.0.0.1",
    #     port=5555,
    #     tv_depth_shape=(720, 1280),
    # )

    client = ImageClient_depth(
        image_show=True,
        server_address="192.168.123.164",
        port=5555,
        Unit_Test=False,
    )
    client.receive_process()
