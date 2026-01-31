"""
PRUSA 3D printer stage control with robust serial communication.

This module wraps the SerialProtocol to provide a high-level API
for stage motion control.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import threading
import time

from raman_control.hardware.base_device import MotionDevice
from raman_control.core.serial_protocol import SerialProtocol
from raman_control.config.hardware_config import StageConfig


@dataclass
class StagePosition:
    """Represents stage position in 3D space."""
    x: float
    y: float
    z: float

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __str__(self):
        return f"X:{self.x:.2f} Y:{self.y:.2f} Z:{self.z:.2f}"


class StepperStage(MotionDevice):
    """
    PRUSA 3D printer stage controller.

    Provides high-level motion control with:
    - Non-blocking movement commands
    - Automatic position updates
    - Homing and referencing
    - Safety height enforcement

    Example:
        stage = StepperStage(config)
        stage.connect()
        stage.start()

        # Reference all axes
        stage.home('XYZ', callback=on_homing_complete)

        # Move to position
        stage.move_to(x=100, y=100, z=30)

        stage.stop()
    """

    def __init__(self, config: Optional[StageConfig] = None):
        """
        Initialize the stepper stage controller.

        Args:
            config: Stage configuration (uses defaults if None)
        """
        super().__init__("stepper_stage")

        self.config = config or StageConfig()

        # Serial protocol for communication
        self._protocol: Optional[SerialProtocol] = None

        # Current position (updated by position queries)
        self._position = StagePosition(0.0, 0.0, 0.0)
        self._position_lock = threading.Lock()

        # Reference status for each axis
        self._referenced = {'X': False, 'Y': False, 'Z': False}

        # Movement speed (mm/min)
        self._speed = self.config.default_speed

        # Keep-alive timer
        self._keepalive_timer: Optional[threading.Timer] = None
        self._keepalive_running = False

        # Callbacks
        self._on_position_changed: Optional[Callable[[StagePosition], None]] = None
        self._on_referenced: Optional[Callable[[str], None]] = None
        self._on_movement_complete: Optional[Callable[[], None]] = None

    @property
    def position(self) -> StagePosition:
        """Get current position (thread-safe)."""
        with self._position_lock:
            return StagePosition(
                self._position.x,
                self._position.y,
                self._position.z
            )

    @property
    def speed(self) -> int:
        """Get current movement speed in mm/min."""
        return self._speed

    @speed.setter
    def speed(self, value: int) -> None:
        """Set movement speed in mm/min."""
        self._speed = max(10, min(3000, value))
        self.logger.info(f"Speed set to {self._speed} mm/min")

    @property
    def is_referenced(self) -> dict:
        """Get reference status for each axis."""
        return self._referenced.copy()

    @property
    def all_referenced(self) -> bool:
        """Check if all axes are referenced."""
        return all(self._referenced.values())

    def connect(self) -> bool:
        """Connect to the PRUSA printer."""
        if self._connected:
            return True

        self._protocol = SerialProtocol(
            port=None,  # Auto-detect
            baudrate=self.config.serial.baudrate,
            timeout=self.config.serial.timeout,
            write_timeout=self.config.serial.write_timeout,
            command_delay=self.config.serial.command_delay,
            max_retries=self.config.serial.max_retries
        )

        if self._protocol.connect():
            self._connected = True
            self.logger.info("Connected to PRUSA stage")
            return True

        self.logger.error("Failed to connect to PRUSA stage")
        return False

    def disconnect(self) -> None:
        """Disconnect from the PRUSA printer."""
        self._stop_keepalive()

        if self._protocol:
            self._protocol.stop()
            self._protocol = None

        self._connected = False
        self._referenced = {'X': False, 'Y': False, 'Z': False}
        self.logger.info("Disconnected from PRUSA stage")

    def start(self) -> bool:
        """Start the stage controller (worker threads)."""
        if not self._connected:
            if not self.connect():
                return False

        if self._protocol.start():
            self._start_keepalive()
            return True

        return False

    def stop(self) -> None:
        """Stop the stage controller."""
        self._stop_keepalive()
        if self._protocol:
            self._protocol.stop()

    def is_ready(self) -> bool:
        """Check if stage is ready for operation."""
        return self._connected and self.all_referenced

    def home(
        self,
        axes: str = 'XYZ',
        callback: Optional[Callable[[bool], None]] = None
    ) -> bool:
        """
        Home specified axes.

        Args:
            axes: Axes to home ('X', 'Y', 'Z', 'XY', 'XYZ', etc.)
            callback: Function called when homing completes

        Returns:
            True if homing command sent successfully
        """
        if not self._protocol or not self._protocol.is_running:
            self.logger.error("Protocol not running")
            return False

        axes = axes.upper()

        def on_homing_complete(success: bool, response: str):
            if success:
                for axis in axes:
                    if axis in self._referenced:
                        self._referenced[axis] = True
                        if self._on_referenced:
                            self._on_referenced(axis)

                self.logger.info(f"Axes {axes} referenced successfully")
            else:
                self.logger.error(f"Homing failed for {axes}: {response}")

            if callback:
                callback(success)

        # For 'ALL' or 'XYZ', first raise Z for safety
        if len(axes) > 1 and 'Z' in axes:
            # Disable software endstops, raise Z, re-enable
            self._protocol.send_command('M211 S0')  # Disable endstops
            time.sleep(0.1)
            self._protocol.send_command('G91')  # Relative positioning
            time.sleep(0.1)
            self._protocol.send_command(f'G1 Z20 F{self.config.homing_speed}')
            time.sleep(2.0)
            self._protocol.send_command('M211 S1')  # Re-enable endstops
            time.sleep(0.1)

        # Send homing command
        gcode = f"G28 {' '.join(axes)} F{self.config.homing_speed}"
        self._protocol.send_command(
            gcode,
            callback=on_homing_complete,
            timeout_ms=60000  # Homing can take a while
        )

        return True

    def get_position(self) -> Tuple[float, float, float]:
        """
        Query current position from printer.

        Returns:
            Tuple of (x, y, z) positions
        """
        if not self._protocol or not self._protocol.is_running:
            return tuple(self._position)

        def parse_position(response: str) -> Optional[StagePosition]:
            """Parse M114 response to extract position."""
            try:
                # Response format: "X:100.00 Y:100.00 Z:30.00 E:0.00 ..."
                parts = response.split()
                pos = {}
                for part in parts:
                    if ':' in part:
                        axis, value = part.split(':')
                        if axis in ('X', 'Y', 'Z'):
                            pos[axis] = float(value)

                if all(k in pos for k in ('X', 'Y', 'Z')):
                    return StagePosition(pos['X'], pos['Y'], pos['Z'])
            except (ValueError, IndexError) as e:
                self.logger.debug(f"Failed to parse position: {e}")

            return None

        def on_response(success: bool, response: str):
            if success:
                # Parse position from response lines
                for line in response.split('\n'):
                    pos = parse_position(line)
                    if pos:
                        with self._position_lock:
                            self._position = pos

                        if self._on_position_changed:
                            self._on_position_changed(pos)
                        return

        self._protocol.send_command('M114', callback=on_response, timeout_ms=3000)
        return tuple(self._position)

    def get_position_sync(self) -> Tuple[float, float, float]:
        """
        Query current position (blocking).

        Returns:
            Tuple of (x, y, z) positions
        """
        success, response = self._protocol.send_command_sync('M114', timeout_ms=3000)

        if success:
            for line in response.split('\n'):
                try:
                    parts = line.split()
                    pos = {}
                    for part in parts:
                        if ':' in part:
                            axis, value = part.split(':')
                            if axis in ('X', 'Y', 'Z'):
                                pos[axis] = float(value)

                    if all(k in pos for k in ('X', 'Y', 'Z')):
                        with self._position_lock:
                            self._position = StagePosition(pos['X'], pos['Y'], pos['Z'])
                        return (pos['X'], pos['Y'], pos['Z'])
                except (ValueError, IndexError):
                    continue

        return tuple(self._position)

    def move_to(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        speed: Optional[int] = None,
        callback: Optional[Callable[[bool], None]] = None
    ) -> bool:
        """
        Move to absolute position.

        Args:
            x, y, z: Target coordinates (None to keep current)
            speed: Movement speed (uses default if None)
            callback: Function called when movement completes

        Returns:
            True if command sent successfully
        """
        if not self._protocol or not self._protocol.is_running:
            self.logger.error("Protocol not running")
            return False

        # Build G-code command
        self._protocol.send_command('G90')  # Absolute positioning

        gcode = "G0"
        if x is not None and self._referenced['X']:
            gcode += f" X{x:.2f}"
        if y is not None and self._referenced['Y']:
            gcode += f" Y{y:.2f}"
        if z is not None and self._referenced['Z']:
            gcode += f" Z{z:.2f}"

        if gcode == "G0":
            self.logger.warning("No valid axes specified for move_to")
            return False

        gcode += f" F{speed or self._speed}"

        def on_complete(success: bool, response: str):
            if success:
                # Update position after move
                self.get_position()
                if self._on_movement_complete:
                    self._on_movement_complete()
            else:
                self.logger.error(f"Move failed: {response}")

            if callback:
                callback(success)

        self._protocol.send_command(gcode, callback=on_complete)
        return True

    def move_by(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        speed: Optional[int] = None,
        callback: Optional[Callable[[bool], None]] = None
    ) -> bool:
        """
        Move by relative distance.

        Args:
            x, y, z: Relative distances (None to skip)
            speed: Movement speed (uses default if None)
            callback: Function called when movement completes

        Returns:
            True if command sent successfully
        """
        if not self._protocol or not self._protocol.is_running:
            self.logger.error("Protocol not running")
            return False

        # Build G-code command
        self._protocol.send_command('G91')  # Relative positioning

        gcode = "G0"
        if x is not None and self._referenced['X']:
            gcode += f" X{x:.2f}"
        if y is not None and self._referenced['Y']:
            gcode += f" Y{y:.2f}"
        if z is not None and self._referenced['Z']:
            gcode += f" Z{z:.2f}"

        if gcode == "G0":
            self.logger.warning("No valid axes specified for move_by")
            return False

        gcode += f" F{speed or self._speed}"

        def on_complete(success: bool, response: str):
            if success:
                self.get_position()
                if self._on_movement_complete:
                    self._on_movement_complete()
            else:
                self.logger.error(f"Move failed: {response}")

            if callback:
                callback(success)

        self._protocol.send_command(gcode, callback=on_complete)
        return True

    def go_to_home(self, callback: Optional[Callable[[bool], None]] = None) -> bool:
        """
        Move to home position (with safety height).

        Args:
            callback: Function called when complete

        Returns:
            True if commands sent successfully
        """
        if not self.all_referenced:
            self.logger.warning("Cannot go to home - not all axes referenced")
            return False

        # Raise to safety height first
        self.move_to(z=self.config.safety_height)
        time.sleep(0.5)

        # Move to home X, Y
        self.move_to(
            x=self.config.home_x,
            y=self.config.home_y
        )
        time.sleep(0.5)

        # Lower to home Z
        self.move_to(z=self.config.home_z, callback=callback)

        return True

    def is_moving(self) -> bool:
        """Check if stage is currently in motion."""
        # Query movement status via M400 (wait for moves to finish)
        # For now, we assume not moving if queue is empty
        if self._protocol:
            return self._protocol.get_queue_size() > 0
        return False

    def stop_motion(self) -> None:
        """Emergency stop - immediately halt all motion."""
        if self._protocol:
            self._protocol.clear_queue()
            self._protocol.send_command('M410')  # Quick stop
            self.logger.warning("Emergency stop triggered")

    def set_callbacks(
        self,
        on_position_changed: Optional[Callable[[StagePosition], None]] = None,
        on_referenced: Optional[Callable[[str], None]] = None,
        on_movement_complete: Optional[Callable[[], None]] = None
    ) -> None:
        """Set callback functions for status updates."""
        self._on_position_changed = on_position_changed
        self._on_referenced = on_referenced
        self._on_movement_complete = on_movement_complete

    def _start_keepalive(self) -> None:
        """Start keep-alive timer to maintain connection."""
        self._keepalive_running = True
        self._send_keepalive()

    def _stop_keepalive(self) -> None:
        """Stop keep-alive timer."""
        self._keepalive_running = False
        if self._keepalive_timer:
            self._keepalive_timer.cancel()
            self._keepalive_timer = None

    def _send_keepalive(self) -> None:
        """Send keep-alive command and schedule next."""
        if not self._keepalive_running or not self._protocol:
            return

        # M105 queries temperature (harmless, keeps connection alive)
        self._protocol.send_command('M105', timeout_ms=2000)

        # Schedule next keep-alive
        self._keepalive_timer = threading.Timer(
            self.config.keepalive_interval,
            self._send_keepalive
        )
        self._keepalive_timer.daemon = True
        self._keepalive_timer.start()
