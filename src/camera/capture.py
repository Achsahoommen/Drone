import time
import cv2
import numpy as np
import threading
import logging
from pathlib import Path
from typing import Optional, Callable, Tuple

from utils.config import CONFIG, CAMERA_CFG, setup_logging

logger = setup_logging("camera")


class CameraCapture:
    def __init__(self, source=None, width=None, height=None, fps=None):
        self.source = source if source is not None else CAMERA_CFG["source"]
        self.width = width or CAMERA_CFG["width"]
        self.height = height or CAMERA_CFG["height"]
        self.fps = fps or CAMERA_CFG["fps"]

        self.capture = None
        self.is_opened = False
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def open(self) -> bool:
        try:
            self.capture = cv2.VideoCapture(self.source)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)

            if not self.capture.isOpened():
                logger.error("Failed to open camera: {}".format(self.source))
                return False

            self.is_opened = True
            logger.info("Camera opened: {}x{} @ {}fps".format(
                self.width, self.height, self.fps
            ))
            return True
        except Exception as e:
            logger.error("Camera error: {}".format(e))
            return False

    def read(self) -> Optional[np.ndarray]:
        if self.capture is None or not self.capture.isOpened():
            return None
        ret, frame = self.capture.read()
        if ret:
            return frame
        return None

    def start_continuous_capture(self, callback: Callable, interval_sec: float = None):
        if interval_sec is None:
            interval_sec = CAMERA_CFG.get("capture_interval_sec", 1.0)

        self._running = True

        def _capture_loop():
            logger.info("Continuous capture started (interval={}s)".format(interval_sec))
            while self._running:
                frame = self.read()
                if frame is not None:
                    with self._lock:
                        self._frame = frame
                    callback(frame)
                time.sleep(interval_sec)
            logger.info("Continuous capture stopped")

        self._thread = threading.Thread(target=_capture_loop, daemon=True)
        self._thread.start()

    def stop_continuous_capture(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def save_frame(self, frame: np.ndarray, path: str):
        cv2.imwrite(path, frame)

    def close(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self.capture:
            self.capture.release()
            self.is_opened = False
            logger.info("Camera closed")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


class FramePreprocessor:
    @staticmethod
    def resize(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def center_crop(frame: np.ndarray, crop_size: int) -> np.ndarray:
        h, w = frame.shape[:2]
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        return frame[start_y:start_y + crop_size, start_x:start_x + crop_size]

    @staticmethod
    def enhance_brightness(frame: np.ndarray, factor: float = 1.2) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def denoise(frame: np.ndarray, strength: int = 10) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, 7, 21)


if __name__ == "__main__":
    cam = CameraCapture(source=0)
    if cam.open():
        logger.info("Press 'q' to quit")
        while True:
            frame = cam.read()
            if frame is None:
                break
            cv2.imshow("Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cam.close()
    else:
        logger.error("Could not open camera")
