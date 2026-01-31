"""
PyQt6 thread for camera acquisition.

Provides non-blocking camera frame acquisition with:
- Continuous acquisition mode
- Single frame capture
- Frame rate control
- Thread-safe image delivery via Qt signals
"""

from typing import Optional, Tuple
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

from raman_control.config.hardware_config import CameraConfig
from raman_control.config.logging_config import get_logger

logger = get_logger('camera_thread')


class CameraThread(QThread):
    """
    Qt thread for camera acquisition.

    Signals:
        frame_ready: Emitted when a new frame is available (numpy array, timestamp)
        connected: Emitted when camera connects
        disconnected: Emitted when camera disconnects
        error: Emitted on errors (error_message)
        fps_updated: Emitted with current frame rate (fps)

    Usage:
        camera = CameraThread(camera_id='widefield', config=config)
        camera.frame_ready.connect(self.on_frame)
        camera.start()

        camera.start_acquisition()
        # ... receive frames via signal
        camera.stop_acquisition()

        camera.stop()
    """

    frame_ready = pyqtSignal(np.ndarray, float)  # frame, timestamp
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)
    fps_updated = pyqtSignal(float)

    def __init__(
        self,
        camera_id: str = 'default',
        config: Optional[CameraConfig] = None,
        parent=None
    ):
        super().__init__(parent)

        self._camera_id = camera_id
        self._config = config or CameraConfig()
        self._mutex = QMutex()

        # State
        self._running = False
        self._acquiring = False
        self._connected = False

        # Frame timing
        self._last_frame_time = 0.0
        self._frame_count = 0
        self._fps = 0.0

        # Latest frame storage for on-demand access
        self._latest_frame: Optional[np.ndarray] = None

        # Simulated camera (replace with actual SDK)
        self._simulated = True  # Set False when real SDK available

    @property
    def is_connected(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._connected

    @property
    def is_acquiring(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._acquiring

    @property
    def current_fps(self) -> float:
        with QMutexLocker(self._mutex):
            return self._fps

    def run(self):
        """Thread main loop."""
        logger.info(f"Camera thread [{self._camera_id}] starting")
        self._running = True

        # Connect to camera
        if not self._connect_camera():
            self.error.emit(f"Failed to connect to camera [{self._camera_id}]")
            self._running = False
            return

        self.connected.emit()
        logger.info(f"Camera [{self._camera_id}] connected")

        # Acquisition loop
        import time
        frame_interval = 1.0 / self._config.frame_rate
        fps_update_interval = 1.0
        last_fps_time = time.time()

        while self._running:
            if self._acquiring:
                frame_start = time.time()

                # Acquire frame
                frame = self._acquire_frame()

                if frame is not None:
                    with QMutexLocker(self._mutex):
                        self._latest_frame = frame.copy()
                    self.frame_ready.emit(frame, frame_start)
                    self._frame_count += 1

                    # Update FPS
                    elapsed = frame_start - last_fps_time
                    if elapsed >= fps_update_interval:
                        with QMutexLocker(self._mutex):
                            self._fps = self._frame_count / elapsed
                        self.fps_updated.emit(self._fps)
                        self._frame_count = 0
                        last_fps_time = frame_start

                # Maintain frame rate
                elapsed = time.time() - frame_start
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                self.msleep(50)  # Idle

        # Cleanup
        self._disconnect_camera()
        self.disconnected.emit()
        logger.info(f"Camera thread [{self._camera_id}] stopped")

    def stop(self):
        """Stop the camera thread."""
        self._running = False
        self._acquiring = False
        self.wait(5000)

    def start_acquisition(self):
        """Start continuous frame acquisition."""
        with QMutexLocker(self._mutex):
            self._acquiring = True
            self._frame_count = 0
        logger.info(f"Camera [{self._camera_id}] acquisition started")

    def stop_acquisition(self):
        """Stop frame acquisition."""
        with QMutexLocker(self._mutex):
            self._acquiring = False
        logger.info(f"Camera [{self._camera_id}] acquisition stopped")

    def capture_single(self) -> Optional[np.ndarray]:
        """Capture a single frame (blocking)."""
        if not self._connected:
            return None
        return self._acquire_frame()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recently acquired frame (thread-safe)."""
        with QMutexLocker(self._mutex):
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def set_exposure(self, exposure_time: float):
        """Set camera exposure time in seconds."""
        self._config.exposure_time = exposure_time
        # TODO: Apply to actual camera SDK
        logger.info(f"Camera [{self._camera_id}] exposure set to {exposure_time}s")

    def set_gain(self, gain: float):
        """Set camera gain."""
        self._config.gain = gain
        # TODO: Apply to actual camera SDK
        logger.info(f"Camera [{self._camera_id}] gain set to {gain}")

    def _connect_camera(self) -> bool:
        """Connect to the camera hardware."""
        try:
            if self._simulated:
                # Simulated connection
                self._connected = True
                return True

            # TODO: Replace with actual Thorlabs SDK initialization
            # Example:
            # from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
            # self._sdk = TLCameraSDK()
            # cameras = self._sdk.discover_available_cameras()
            # self._camera = self._sdk.open_camera(cameras[0])
            # self._camera.exposure_time_us = int(self._config.exposure_time * 1e6)

            self._connected = True
            return True

        except Exception as e:
            logger.error(f"Camera connection error: {e}")
            return False

    def _disconnect_camera(self):
        """Disconnect from the camera."""
        try:
            if not self._simulated:
                # TODO: Cleanup actual camera SDK
                # self._camera.dispose()
                # self._sdk.dispose()
                pass

            self._connected = False

        except Exception as e:
            logger.error(f"Camera disconnect error: {e}")

    def _acquire_frame(self) -> Optional[np.ndarray]:
        """Acquire a single frame from the camera."""
        try:
            if self._simulated:
                # Generate simulated frame (grayscale gradient with noise)
                import time
                height, width = 480, 640

                # Create gradient
                x = np.linspace(0, 1, width)
                y = np.linspace(0, 1, height)
                xx, yy = np.meshgrid(x, y)

                # Add time-varying pattern
                t = time.time()
                pattern = np.sin(xx * 10 + t) * np.cos(yy * 10 + t * 0.5)

                # Add noise
                noise = np.random.normal(0, 0.1, (height, width))

                # Combine and scale to uint8
                frame = ((pattern + noise + 1) * 127.5).astype(np.uint8)

                return frame

            # TODO: Replace with actual camera SDK frame acquisition
            # frame = self._camera.get_pending_frame_or_null()
            # if frame:
            #     return frame.image_buffer.copy()

            return None

        except Exception as e:
            logger.error(f"Frame acquisition error: {e}")
            return None
