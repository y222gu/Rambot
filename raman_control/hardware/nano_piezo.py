"""
MadCity Labs NanoDrive piezo controller.

Provides high-level API for nano-positioning with:
- Thread-safe position control
- Position feedback
- Movement with stabilization
"""

from typing import Optional, Callable, Tuple
import threading
import time

from raman_control.hardware.base_device import MotionDevice
from raman_control.config.hardware_config import NanoDriveConfig
from raman_control.config.logging_config import get_logger

logger = get_logger('nano_piezo')


class NanoPiezo(MotionDevice):
    """
    MadCity Labs NanoDrive piezo controller.

    Provides fine Z-axis positioning for objective focus adjustment.

    Features:
    - Micron-level positioning
    - Fast response time
    - Position feedback
    - Thread-safe operation

    Example:
        piezo = NanoPiezo(config)
        piezo.connect()

        piezo.move_to(z=50.0)  # Move to 50 um
        piezo.move_by(z=1.0)   # Move up 1 um

        position = piezo.get_position()
        print(f"Current Z: {position} um")

        piezo.disconnect()
    """

    def __init__(self, config: Optional[NanoDriveConfig] = None):
        super().__init__("nano_piezo")

        self._config = config or NanoDriveConfig()
        self._lock = threading.Lock()

        # Current position (um)
        self._position = self._config.initial_position

        # Position limits
        self._min_position = self._config.min_position
        self._max_position = self._config.max_position

        # Simulated mode (replace with actual SDK)
        self._simulated = True

        # Hardware handle
        self._handle = None

        # Callbacks
        self._on_position_changed: Optional[Callable[[float], None]] = None

    @property
    def position(self) -> float:
        """Get current position in um."""
        with self._lock:
            return self._position

    @property
    def min_position(self) -> float:
        """Get minimum position limit."""
        return self._min_position

    @property
    def max_position(self) -> float:
        """Get maximum position limit."""
        return self._max_position

    def connect(self) -> bool:
        """Connect to the NanoDrive controller."""
        try:
            if self._simulated:
                self._connected = True
                self.logger.info("NanoDrive connected (simulated)")
                return True

            # TODO: Replace with actual MadCity Labs SDK
            # Example:
            # import madlib  # MadCity Labs library
            # self._handle = madlib.MCL_InitHandle()
            # if self._handle == 0:
            #     raise RuntimeError("Failed to initialize NanoDrive")
            # self._min_position = madlib.MCL_GetCalLowEnd(3, self._handle)
            # self._max_position = madlib.MCL_GetCalHighEnd(3, self._handle)

            self._connected = True
            self.logger.info("NanoDrive connected")
            return True

        except Exception as e:
            self.logger.error(f"NanoDrive connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from the NanoDrive controller."""
        try:
            if not self._simulated and self._handle:
                # TODO: Cleanup actual SDK
                # madlib.MCL_ReleaseHandle(self._handle)
                pass

            self._handle = None
            self._connected = False
            self.logger.info("NanoDrive disconnected")

        except Exception as e:
            self.logger.error(f"NanoDrive disconnect error: {e}")

    def is_ready(self) -> bool:
        """Check if NanoDrive is ready."""
        return self._connected

    def get_position(self) -> Tuple[float]:
        """
        Get current position.

        Returns:
            Tuple containing Z position in um
        """
        if not self._connected:
            return (self._position,)

        try:
            if self._simulated:
                return (self._position,)

            # TODO: Read actual position
            # position = madlib.MCL_SingleReadZ(self._handle)
            # self._position = position
            # return (position,)

            return (self._position,)

        except Exception as e:
            self.logger.error(f"Position read error: {e}")
            return (self._position,)

    def move_to(self, z: Optional[float] = None, **kwargs) -> bool:
        """
        Move to absolute position.

        Args:
            z: Target Z position in um

        Returns:
            True if move successful
        """
        if z is None:
            return False

        if not self._connected:
            self.logger.error("NanoDrive not connected")
            return False

        # Clamp to limits
        target = max(self._min_position, min(self._max_position, z))

        try:
            with self._lock:
                if self._simulated:
                    self._position = target
                else:
                    # TODO: Actual movement
                    # madlib.MCL_SingleWriteZ(target, self._handle)
                    # time.sleep(self._config.stabilization_time)
                    # self._position = madlib.MCL_SingleReadZ(self._handle)
                    pass

            self.logger.debug(f"NanoDrive moved to Z={target:.2f} um")

            if self._on_position_changed:
                self._on_position_changed(self._position)

            return True

        except Exception as e:
            self.logger.error(f"NanoDrive move error: {e}")
            return False

    def move_by(self, z: Optional[float] = None, **kwargs) -> bool:
        """
        Move by relative distance.

        Args:
            z: Relative Z distance in um

        Returns:
            True if move successful
        """
        if z is None:
            return False

        target = self._position + z
        return self.move_to(z=target)

    def stop(self) -> None:
        """Stop motion (NanoDrive responds instantly, so this is a no-op)."""
        pass

    def home(self, axes: Optional[str] = None) -> bool:
        """
        Move to initial/home position.

        Returns:
            True if successful
        """
        return self.move_to(z=self._config.initial_position)

    def is_moving(self) -> bool:
        """NanoDrive responds instantly, so never 'moving'."""
        return False

    def get_limits(self) -> Tuple[float, float]:
        """
        Get position limits.

        Returns:
            Tuple of (min_position, max_position) in um
        """
        return (self._min_position, self._max_position)

    def set_callback(self, on_position_changed: Optional[Callable[[float], None]] = None):
        """Set callback for position changes."""
        self._on_position_changed = on_position_changed

    def scan_range(
        self,
        start: float,
        end: float,
        steps: int,
        callback: Optional[Callable[[float, int], None]] = None,
        stabilization_time: Optional[float] = None
    ) -> bool:
        """
        Scan through a range of Z positions.

        Args:
            start: Start position in um
            end: End position in um
            steps: Number of steps
            callback: Function called at each position (position, step_index)
            stabilization_time: Time to wait at each position (uses config if None)

        Returns:
            True if scan completed successfully
        """
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
