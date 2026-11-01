"""
PyQt6 thread for camera acquisition using Thorlabs TSI SDK.

Both camera threads share a single TLCameraSDK instance via ThorlabsCameraManager.
Each thread owns its TLCamera handle and runs its own acquisition loop.
Falls back to simulated mode if SDK is unavailable or cameras not found.
"""

import os
import sys
import time
import threading
from typing import Optional, List

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

from raman_control.config.hardware_config import CameraConfig, _default_dll_dir
from raman_control.config.logging_config import get_logger

logger = get_logger('camera_thread')

# ---------------------------------------------------------------------------
# DLL path setup – must happen before any thorlabs_tsi_sdk import
# ---------------------------------------------------------------------------
_DLL_PATH_CONFIGURED = False


def configure_thorlabs_dll_path(dll_dir: Optional[str] = None):
    """Add the Thorlabs DLL directory to the process search path."""
    global _DLL_PATH_CONFIGURED
    if _DLL_PATH_CONFIGURED:
        return

    dll_dir = dll_dir or _default_dll_dir()
    if not os.path.isdir(dll_dir):
        logger.warning(f"Thorlabs DLL directory not found: {dll_dir}")
        return

    os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')
    try:
        os.add_dll_directory(dll_dir)
    except (AttributeError, OSError):
        pass

    _DLL_PATH_CONFIGURED = True
    logger.info(f"Thorlabs DLL path configured: {dll_dir}")


# ---------------------------------------------------------------------------
# Shared camera SDK manager (singleton)
# ---------------------------------------------------------------------------

class ThorlabsCameraManager:
    """
    Manages a single TLCameraSDK instance shared by all camera threads.

    The Thorlabs SDK requires:
    1. Only one TLCameraSDK instance per process
    2. discover_available_cameras() before open_camera()
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._sdk = None
        self._available: List[str] = []
        self._ref_count = 0
        self._sdk_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'ThorlabsCameraManager':
        with cls._lock:
            if cls._instance is None:
                cls._instance = ThorlabsCameraManager()
            return cls._instance

    def initialize(self, dll_dir: Optional[str] = None) -> bool:
        """Initialize SDK if not already running. Thread-safe, ref-counted.

        The TLCameraSDK is created exactly once per process and kept alive
        for the process lifetime (see ``release``/``dispose`` for why). On
        a fresh connect cycle — i.e. when no cameras are currently using
        the SDK (ref_count == 0) — cameras are (re)discovered so that a
        disconnect/reconnect re-detects the hardware.
        """
        with self._sdk_lock:
            try:
                # Create the SDK only once. Disposing and re-creating a
                # TLCameraSDK in the same process leaves the native library
                # in an invalid state (image_poll -> error 1004), so we
                # never tear it down on a routine disconnect.
                if self._sdk is None:
                    configure_thorlabs_dll_path(dll_dir)
                    from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
                    self._sdk = TLCameraSDK()
                    logger.info("ThorlabsCameraManager: SDK created")

                # (Re)discover cameras at the start of each connect cycle.
                # Safe here because no camera handle is open while
                # ref_count is 0 (Thorlabs forbids discover after open).
                if self._ref_count == 0:
                    self._available = self._sdk.discover_available_cameras()
                    logger.info(f"ThorlabsCameraManager: discovered "
                                f"{self._available}")

                self._ref_count += 1
                return True
            except Exception as e:
                logger.warning(f"ThorlabsCameraManager: SDK init failed: {e}")
                # Only drop the SDK reference if we just failed to create it.
                if self._ref_count == 0:
                    self._sdk = None
                return False

    def open_camera(self, serial_number: str):
        """Open a camera by serial number. Returns TLCamera or None."""
        with self._sdk_lock:
            if self._sdk is None:
                return None
            if serial_number not in self._available:
                logger.warning(
                    f"Serial '{serial_number}' not in {self._available}"
                )
                return None
            try:
                camera = self._sdk.open_camera(serial_number)
                logger.info(f"Opened camera: {serial_number}")
                return camera
            except Exception as e:
                logger.error(f"Failed to open camera {serial_number}: {e}")
                return None

    @property
    def available_cameras(self) -> List[str]:
        return list(self._available)

    def release(self):
        """Decrement ref count when a camera thread disconnects.

        The SDK is intentionally NOT disposed here, even when the last
        user releases. The Thorlabs SDK cannot be re-initialized after
        dispose within the same process, so disposing on disconnect is
        what breaks subsequent reconnects (image_poll error 1004 and
        crashes). The SDK lives until the process exits via dispose().
        """
        with self._sdk_lock:
            if self._ref_count > 0:
                self._ref_count -= 1

    def dispose(self):
        """Dispose the SDK. Call ONCE at application exit only.

        After this, no camera operations are valid for the rest of the
        process. Used by the main window's closeEvent.
        """
        with self._sdk_lock:
            if self._sdk is not None:
                try:
                    self._sdk.dispose()
                except Exception:
                    pass
                self._sdk = None
                self._available = []
                self._ref_count = 0
                logger.info("ThorlabsCameraManager: SDK disposed")


# ---------------------------------------------------------------------------
# Camera acquisition thread
# ---------------------------------------------------------------------------

class CameraThread(QThread):
    """
    Qt thread for camera acquisition.

    Signals:
        frame_ready: Emitted when a new frame is available (numpy array, timestamp)
        connected: Emitted when camera connects
        disconnected: Emitted when camera disconnects
        error: Emitted on errors (error_message)
        fps_updated: Emitted with current frame rate (fps)
    """

    frame_ready = pyqtSignal(np.ndarray, float)  # frame, timestamp
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)
    fps_updated = pyqtSignal(float)

    # Serialize camera connections across all instances to prevent SDK race conditions
    _connect_lock = threading.Lock()

    def __init__(
        self,
        camera_id: str = 'default',
        config: Optional[CameraConfig] = None,
        dll_dir: Optional[str] = None,
        parent=None
    ):
        super().__init__(parent)

        self._camera_id = camera_id
        self._config = config or CameraConfig()
        self._dll_dir = dll_dir
        self._mutex = QMutex()

        # State
        self._running = False
        self._acquiring = False
        self._connected = False

        # Frame timing
        self._frame_count = 0
        self._fps = 0.0

        # Latest frame storage for on-demand access
        self._latest_frame: Optional[np.ndarray] = None

        # Camera handle (from shared manager)
        self._camera = None
        self._mono_to_color_sdk = None
        self._mono_to_color_processor = None
        self._is_color = False

        # Camera properties cached after connect
        self._bit_depth = 0
        self._image_width = 0
        self._image_height = 0

        # Simulated camera flag
        self._simulated = True
        self._sdk_ref_held = False

        # Optional raw-frame listeners (calibration tab). Called on THIS
        # thread with (raw_frame, timestamp, sdk_frame_count) for every new
        # frame, BEFORE bit-depth scaling and flip/rotate/crop. Empty in
        # normal operation, so the acquisition loop is unchanged.
        self._raw_listeners: list = []
        self._raw_lock = threading.Lock()
        self._sim_frame_count = 0
        # Sensor properties read once at connect (on this thread), so other
        # threads never have to touch the SDK handle to report them.
        self._sensor_info: dict = {}

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

    @property
    def is_simulated(self) -> bool:
        return self._simulated

    def run(self):
        """Thread main loop."""
        logger.info(f"Camera thread [{self._camera_id}] starting")
        self._running = True

        if not self._connect_camera():
            # Release any SDK ref/handle acquired during the failed attempt
            # so the ref count returns to 0 and a later reconnect can
            # re-discover cameras.
            self._disconnect_camera()
            self.error.emit(f"Failed to connect to camera [{self._camera_id}]")
            self._running = False
            return

        self.connected.emit()
        logger.info(f"Camera [{self._camera_id}] connected "
                     f"({'simulated' if self._simulated else 'hardware'})")

        # Auto-start acquisition so both cameras stream immediately
        with QMutexLocker(self._mutex):
            self._acquiring = True
            self._frame_count = 0

        # Acquisition loop
        frame_interval = 1.0 / max(1.0, self._config.frame_rate)
        fps_update_interval = 1.0
        last_fps_time = time.time()

        while self._running:
            if self._acquiring:
                frame_start = time.time()

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

    def add_raw_frame_listener(self, callback) -> None:
        """Register ``callback(raw, timestamp, frame_count)``, called on the
        camera thread for every new frame with an untouched copy of the
        sensor buffer (native bit depth, full sensor, no flip/rotate/crop).
        The callback must return quickly (e.g. hand off to a queue)."""
        with self._raw_lock:
            if callback not in self._raw_listeners:
                self._raw_listeners.append(callback)

    def remove_raw_frame_listener(self, callback) -> None:
        with self._raw_lock:
            if callback in self._raw_listeners:
                self._raw_listeners.remove(callback)

    @property
    def sensor_info(self) -> dict:
        """Sensor properties captured at connect (empty if not connected)."""
        info = dict(self._sensor_info)
        info['exposure_ms'] = self._config.exposure_time * 1000.0
        return info

    def _notify_raw_listeners(self, raw: np.ndarray, frame_count: int) -> None:
        with self._raw_lock:
            listeners = list(self._raw_listeners)
        if not listeners:
            return
        t = time.time()
        for cb in listeners:
            try:
                cb(raw, t, frame_count)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Raw frame listener error: {e}")

    def set_exposure(self, exposure_time: float):
        """Set camera exposure time in seconds."""
        self._config.exposure_time = exposure_time
        try:
            if not self._simulated and self._camera is not None:
                self._camera.exposure_time_us = int(exposure_time * 1e6)
            logger.info(f"Camera [{self._camera_id}] exposure set to {exposure_time}s")
        except Exception as e:
            logger.error(f"Failed to set exposure: {e}")

    def set_gain(self, gain: float):
        """Set camera gain."""
        self._config.gain = gain
        try:
            if not self._simulated and self._camera is not None:
                if hasattr(self._camera, 'gain'):
                    self._camera.gain = float(gain)
            logger.info(f"Camera [{self._camera_id}] gain set to {gain}")
        except Exception as e:
            logger.error(f"Failed to set gain: {e}")

    # ------------------------------------------------------------------
    # Internal: connect / disconnect / acquire
    # ------------------------------------------------------------------

    def _connect_camera(self) -> bool:
        """Connect to the Thorlabs camera via the shared manager.

        Uses a class-level lock to serialize connections across all camera
        thread instances, preventing SDK race conditions when both cameras
        are started simultaneously.

        Returns False on failure (no simulation fallback).
        """
        with CameraThread._connect_lock:
            mgr = ThorlabsCameraManager.get_instance()
            if not mgr.initialize(self._dll_dir):
                logger.error(f"[{self._camera_id}] Thorlabs SDK not available")
                return False

            self._sdk_ref_held = True  # We incremented the SDK ref count

            serial = self._config.serial_number
            if not serial:
                logger.error(f"[{self._camera_id}] No serial number configured")
                return False

            self._camera = mgr.open_camera(serial)
            if self._camera is None:
                logger.error(f"[{self._camera_id}] Could not open camera {serial}")
                return False

            try:
                # Configure camera
                self._camera.exposure_time_us = int(self._config.exposure_time * 1e6)
                self._camera.image_poll_timeout_ms = self._config.image_poll_timeout_ms
                self._bit_depth = self._camera.bit_depth
                self._image_width = self._camera.image_width_pixels
                self._image_height = self._camera.image_height_pixels

                logger.info(f"[{self._camera_id}] {self._image_width}x{self._image_height}, "
                            f"bit_depth={self._bit_depth}, sensor={self._camera.camera_sensor_type}")

                info = {
                    'camera_id': self._camera_id,
                    'serial_number': serial,
                    'width': self._image_width,
                    'height': self._image_height,
                    'bit_depth': self._bit_depth,
                    'sensor_type': str(self._camera.camera_sensor_type),
                }
                for key, attr in (('model', 'model'), ('gain', 'gain'),
                                  ('exposure_us_at_connect', 'exposure_time_us')):
                    try:
                        info[key] = getattr(self._camera, attr)
                    except Exception:  # noqa: BLE001 - optional property
                        pass
                self._sensor_info = info

                # Setup color processing if camera has Bayer sensor
                try:
                    from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
                    from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK

                    if self._camera.camera_sensor_type != SENSOR_TYPE.BAYER:
                        self._is_color = False
                    else:
                        self._mono_to_color_sdk = MonoToColorProcessorSDK()
                        self._mono_to_color_processor = (
                            self._mono_to_color_sdk.create_mono_to_color_processor(
                                SENSOR_TYPE.BAYER,
                                self._camera.color_filter_array_phase,
                                self._camera.get_color_correction_matrix(),
                                self._camera.get_default_white_balance_matrix(),
                                self._camera.bit_depth,
                            )
                        )
                        self._is_color = True
                        logger.info(f"[{self._camera_id}] Color processing enabled")
                except Exception as e:
                    logger.warning(f"[{self._camera_id}] Color processing setup failed: {e}")
                    self._is_color = False

                # Arm camera for continuous acquisition (2-frame buffer)
                self._camera.frames_per_trigger_zero_for_unlimited = 0
                self._camera.arm(2)
                self._camera.issue_software_trigger()

                self._simulated = False
                self._connected = True
                return True

            except Exception as e:
                logger.error(f"[{self._camera_id}] Camera setup error: {e}")
                self._cleanup_camera()
                return False

    def _disconnect_camera(self):
        """Disconnect from the camera and release SDK ref."""
        self._cleanup_camera()
        # Release our ref on the shared SDK manager only if we hold one
        if self._sdk_ref_held:
            mgr = ThorlabsCameraManager.get_instance()
            mgr.release()
            self._sdk_ref_held = False
        self._connected = False

    def _cleanup_camera(self):
        """Dispose camera and color processing objects."""
        if self._camera is not None:
            try:
                self._camera.disarm()
            except Exception:
                pass
            try:
                self._camera.dispose()
            except Exception:
                pass
            self._camera = None

        if self._mono_to_color_processor is not None:
            try:
                self._mono_to_color_processor.dispose()
            except Exception:
                pass
            self._mono_to_color_processor = None

        if self._mono_to_color_sdk is not None:
            try:
                self._mono_to_color_sdk.dispose()
            except Exception:
                pass
            self._mono_to_color_sdk = None

    def _acquire_frame(self) -> Optional[np.ndarray]:
        """Acquire a single frame from the camera or simulated generator."""
        try:
            if self._simulated or self._camera is None:
                image = self._generate_simulated_frame()
                if self._raw_listeners:
                    self._sim_frame_count += 1
                    self._notify_raw_listeners(image.copy(), self._sim_frame_count)
                return image

            frame = self._camera.get_pending_frame_or_null()
            if frame is None:
                return None

            if self._raw_listeners:
                self._notify_raw_listeners(
                    np.array(frame.image_buffer, copy=True),
                    int(getattr(frame, 'frame_count', -1)))

            if self._is_color:
                image = self._process_color_frame(frame)
            else:
                image = self._process_mono_frame(frame)

            image = self._apply_transforms(image)
            # Always return a contiguous copy to prevent SDK buffer reuse issues
            return np.ascontiguousarray(image)

        except Exception as e:
            logger.error(f"Frame acquisition error: {e}")
            return None

    def _process_mono_frame(self, frame) -> np.ndarray:
        """Process a monochrome frame: scale bit depth to 8-bit."""
        scaled = frame.image_buffer >> (self._bit_depth - 8)
        image = np.asarray(scaled, dtype=np.uint8).squeeze()
        return image

    def _process_color_frame(self, frame) -> np.ndarray:
        """Process a color (Bayer) frame: demosaic and convert to grayscale."""
        width = frame.image_buffer.shape[1]
        height = frame.image_buffer.shape[0]

        if width != self._image_width or height != self._image_height:
            self._image_width = width
            self._image_height = height
            logger.info("Image dimension change detected")

        color_data = self._mono_to_color_processor.transform_to_24(
            frame.image_buffer, self._image_width, self._image_height
        )
        color_data = color_data.reshape(self._image_height, self._image_width, 3)

        # Convert to single-channel grayscale (mean of RGB)
        gray = color_data.mean(axis=2).astype(np.uint8)
        return gray

    def _apply_transforms(self, image: np.ndarray) -> np.ndarray:
        """Apply flip, rotate, and crop transformations."""
        cfg = self._config

        if cfg.flip_horizontal:
            image = np.fliplr(image).copy()

        if cfg.rotate_minus90:
            image = np.rot90(image, k=-1).copy()

        if cfg.crop_enabled:
            cx, cy = cfg.crop_center_x, cfg.crop_center_y
            cw, ch = cfg.crop_width, cfg.crop_height
            h, w = image.shape[:2]

            x1 = max(0, cx - cw // 2)
            y1 = max(0, cy - ch // 2)
            x2 = min(w, x1 + cw)
            y2 = min(h, y1 + ch)
            image = image[y1:y2, x1:x2].copy()

        return image

    def _generate_simulated_frame(self) -> np.ndarray:
        """Generate a simulated grayscale frame for testing without hardware."""
        height, width = 480, 640

        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)

        t = time.time()
        pattern = np.sin(xx * 10 + t) * np.cos(yy * 10 + t * 0.5)
        noise = np.random.normal(0, 0.1, (height, width))

        frame = ((pattern + noise + 1) * 127.5).astype(np.uint8)
        return frame
