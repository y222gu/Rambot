"""
MadCity Labs NanoDrive piezo controller.

Uses ctypes to interface with Madlib.dll for Z-axis nano-positioning.
Falls back to simulated mode if the DLL is not available.
"""

import os
import threading
import time
from ctypes import CDLL, c_double, c_uint
from typing import Optional, Callable, Tuple

from raman_control.hardware.base_device import MotionDevice
from raman_control.config.hardware_config import NanoDriveConfig
from raman_control.config.logging_config import get_logger

logger = get_logger('nano_piezo')


class NanoPiezo(MotionDevice):
    """
    MadCity Labs NanoDrive piezo controller.

    Loads Madlib.dll via ctypes and calls MCL_InitHandle / MCL_SingleWriteN /
    MCL_SingleReadN / MCL_ReleaseHandle for Z-axis positioning.

    Falls back to simulated mode when DLL is unavailable.
    """

    def __init__(self, config: Optional[NanoDriveConfig] = None):
        super().__init__("nano_piezo")

        self._config = config or NanoDriveConfig()
        self._lock = threading.Lock()

        # Current position (um)
        self._position = 0.0

        # Position limits
        self._min_position = self._config.min_position
        self._max_position = self._config.max_position

        # DLL / hardware handle
        self._dll = None
        self._handle = None
        self._axis = c_uint(self._config.axis)

        # Simulated mode flag
        self._simulated = True

        # Callbacks
        self._on_position_changed: Optional[Callable[[float], None]] = None

    @property
    def position(self) -> float:
        """Get current position in um."""
        with self._lock:
            return self._position

    @property
    def min_position(self) -> float:
        return self._min_position

    @property
    def max_position(self) -> float:
        return self._max_position

    @property
    def is_simulated(self) -> bool:
        return self._simulated

    def connect(self) -> bool:
        """Connect to the NanoDrive controller via Madlib.dll.

        Returns False on failure (no simulation fallback).
        """
        try:
            dll_path = self._config.dll_path
            if not dll_path or not os.path.isfile(dll_path):
                self.logger.error(f"Madlib.dll not found at '{dll_path}'")
                return False

            # Load DLL
            self._dll = CDLL(dll_path)

            # Configure function signatures
            self._dll.MCL_ReleaseHandle.restype = None
            self._dll.MCL_SingleReadN.restype = c_double

            # Initialize handle
            self._handle = self._dll.MCL_InitHandle()
            if self._handle == 0:
                raise RuntimeError("MCL_InitHandle returned 0 – device not found")

            self.logger.info(f"MCL Handle = {self._handle}")

            # Move to initial position (0 um like old code)
            pos = c_double(0.0)
            error = self._dll.MCL_SingleWriteN(pos, self._axis, self._handle)
            time.sleep(self._config.stabilization_time)
            if error != 0:
                raise RuntimeError(f"NanoDrive init write error: {error}")

            # Read back actual position
            self._position = round(
                self._dll.MCL_SingleReadN(self._axis, self._handle)
            )

            self._simulated = False
            self._connected = True
            self.logger.info(
                f"NanoDrive connected (hardware), position={self._position} um"
            )
            return True

        except Exception as e:
            self.logger.error(f"NanoDrive connection failed: {e}")
            self._dll = None
            self._handle = None
            return False

    def disconnect(self) -> None:
        """Disconnect from the NanoDrive controller."""
        try:
            if not self._simulated and self._dll is not None and self._handle:
                self._dll.MCL_ReleaseHandle(self._handle)
                self.logger.info("NanoDrive handle released")

            self._handle = None
            self._dll = None
            self._connected = False
            self.logger.info("NanoDrive disconnected")

        except Exception as e:
            self.logger.error(f"NanoDrive disconnect error: {e}")

    def is_ready(self) -> bool:
        return self._connected

    def get_position(self) -> Tuple[float]:
        """Read current Z position from hardware."""
        if not self._connected:
            return (self._position,)

        try:
            if self._simulated:
                return (self._position,)

            raw = self._dll.MCL_SingleReadN(self._axis, self._handle)
            self._position = round(raw)
            return (self._position,)

        except Exception as e:
            self.logger.error(f"Position read error: {e}")
            return (self._position,)

    def move_to(self, z: Optional[float] = None, **kwargs) -> bool:
        """Move to absolute Z position in um."""
        if z is None:
            return False

        if not self._connected:
            self.logger.error("NanoDrive not connected")
            return False

        # Clamp to limits
        target = max(self._min_position, min(self._max_position, float(z)))

        try:
            with self._lock:
                if self._simulated:
                    self._position = target
                else:
                    pos = c_double(target)
                    error = self._dll.MCL_SingleWriteN(
                        pos, self._axis, self._handle
                    )
                    time.sleep(self._config.stabilization_time)
                    if error != 0:
                        self.logger.error(
                            f"NanoDrive move_to error={error}, "
                            f"target={target}, axis={self._axis.value}"
                        )
                        return False
                    # Read back actual position
                    self._position = round(
                        self._dll.MCL_SingleReadN(self._axis, self._handle)
                    )

            self.logger.debug(f"NanoDrive moved to Z={self._position:.1f} um")

            if self._on_position_changed:
                self._on_position_changed(self._position)

            return True

        except Exception as e:
            self.logger.error(f"NanoDrive move error: {e}")
            return False

    def move_by(self, z: Optional[float] = None, **kwargs) -> bool:
        """Move by relative Z distance in um."""
        if z is None:
            return False

        if not self._connected:
            self.logger.error("NanoDrive not connected")
            return False

        try:
            with self._lock:
                if self._simulated:
                    current = self._position
                else:
                    current = self._dll.MCL_SingleReadN(
                        self._axis, self._handle
                    )

            target = current + float(z)
            # move_to handles clamping and write
            return self.move_to(z=target)

        except Exception as e:
            self.logger.error(f"NanoDrive move_by error: {e}")
            return False

    def stop(self) -> None:
        """NanoDrive responds instantly, so this is a no-op."""
        pass

    def home(self, axes: Optional[str] = None) -> bool:
        """Move to initial/home position."""
        return self.move_to(z=self._config.initial_position)

    def is_moving(self) -> bool:
        """NanoDrive settles within stabilization_time, so never 'moving'."""
        return False

    def get_limits(self) -> Tuple[float, float]:
        return (self._min_position, self._max_position)

    def set_callback(self, on_position_changed: Optional[Callable[[float], None]] = None):
        self._on_position_changed = on_position_changed

    def scan_range(
        self,
        start: float,
        end: float,
        steps: int,
        callback: Optional[Callable[[float, int], None]] = None,
        stabilization_time: Optional[float] = None,
    ) -> bool:
        """Scan through a range of Z positions."""
        if not self._connected:
            return False

        stab_time = stabilization_time or self._config.stabilization_time
        positions = [start + (end - start) * i / (steps - 1) for i in range(steps)]

        for i, pos in enumerate(positions):
            if not self.move_to(z=pos):
                return False
            time.sleep(stab_time)
            if callback:
                callback(pos, i)

        return True
