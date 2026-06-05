"""
PRUSA 3D printer stage control with robust serial communication.

This module wraps the SerialProtocol to provide a high-level API
for stage motion control.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import threading

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
        # Last commanded absolute target — the base for relative moves
        # (move_by) so successive jogs accumulate without depending on a
        # freshly-measured position. Set on homing and on every move.
        self._target_position = StagePosition(0.0, 0.0, 0.0)
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
        self._speed = max(10, min(self.config.default_speed, value))
        self.logger.info(f"Speed set to {self._speed} mm/min")

    def _parse_position_line(self, line: str) -> Optional[StagePosition]:
        """
        Parse a single line for position data (e.g. M114 response or G28 output).

        Handles formats like:
            "X:100.00 Y:200.00 Z:30.00 E:0.00 Count X:0 Y:0 Z:0"
            "ok X:0.00 Y:0.00 Z:0.00"

        Only the FIRST occurrence of each axis is kept (ignores Count values).

        Returns:
            StagePosition if X, Y, Z all found, None otherwise
        """
        try:
            parts = line.split()
            pos = {}
            for part in parts:
                if ':' in part:
                    # Use maxsplit=1 to handle unexpected extra colons
                    axis, value = part.split(':', 1)
                    if axis in ('X', 'Y', 'Z') and axis not in pos:
                        pos[axis] = float(value)

            if all(k in pos for k in ('X', 'Y', 'Z')):
                return StagePosition(pos['X'], pos['Y'], pos['Z'])
        except (ValueError, IndexError) as e:
            self.logger.debug(f"Failed to parse position from: {line!r} ({e})")

        return None

    def _update_position(self, pos: StagePosition) -> None:
        """Update cached position and notify listeners."""
        with self._position_lock:
            self._position = pos
        if self._on_position_changed:
            self._on_position_changed(pos)

    def _set_commanded_target(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
    ) -> None:
        """Record the latest commanded absolute target for the given axes.

        Used as the base for relative moves (move_by) so successive jogs
        accumulate correctly without waiting for each move to physically
        settle. Only updates the axes that were actually commanded.
        """
        with self._position_lock:
            if x is not None:
                self._target_position.x = x
            if y is not None:
                self._target_position.y = y
            if z is not None:
                self._target_position.z = z

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
        Home specified axes safely.

        For any home that includes Z, the probe is FIRST lifted clear of
        the sample/stage before homing, mirroring the proven noodlepy
        "Reference All" sequence:

          1. Drain stale serial responses (prevents ACK mismatches)
          2. Pre-homing safety lift (Z not yet referenced):
               M211 S0  -> disable software endstops (allow move on an
                           unhomed axis; Marlin rejects it otherwise)
               G91      -> relative positioning
               G1 Z+lift-> raise the probe away from the sample
               M400     -> wait for the lift to physically complete
               M211 S1  -> re-enable software endstops
               G90      -> back to absolute positioning
          3. Combined G28 (e.g. G28 X Y Z) — firmware homes X/Y, then
             (with Z_SAFE_HOMING) moves to the safe XY point before
             lowering Z to probe. Sending one combined G28 preserves
             this firmware behaviour; splitting it bypasses the safe
             XY repositioning and makes Z descend in place.
          4. After Z is referenced, M400 then raise to safety_height
             (reliable absolute move on a now-referenced axis)
          5. Notify listeners (GUI enables buttons)

        The pre-homing lift is what keeps homing X (which drives toward
        X=0) from dragging the probe across the sample. It is skipped
        for XY-only homes, which involve no Z risk.

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

        # Pause keepalive and drain any stale responses that could
        # be mistaken for ACKs during the homing sequence.
        self._stop_keepalive()
        self._protocol.drain_response_queue()

        has_xy = any(a in axes for a in 'XY')
        has_z = 'Z' in axes
        homed_axes = ''.join(a for a in 'XYZ' if a in axes)

        def _notify_referenced():
            """Notify listeners and resume keepalive."""
            self._start_keepalive()
            for axis in axes:
                if axis in self._referenced and self._on_referenced:
                    self._on_referenced(axis)

        def _on_fail(msg: str):
            """Handle homing failure: log, resume keepalive, notify."""
            self.logger.error(msg)
            self._start_keepalive()
            if callback:
                callback(False)

        def _on_z_safety_raise_done(raise_ok: bool):
            """After post-homing Z raise completes, notify listeners."""
            if not raise_ok:
                self.logger.warning(
                    "Failed to raise to safety height after homing"
                )
            # Re-anchor the relative-move base to the now-settled measured
            # position (the safety raise queried M114). This covers all
            # axes even when the G28 reply carried no position line.
            measured = self.position
            self._set_commanded_target(measured.x, measured.y, measured.z)
            _notify_referenced()
            if callback:
                callback(True)

        def _do_post_homing_safety_raise(ok: bool, _resp: str):
            """After M400 confirms homing physically done, raise Z."""
            if not ok:
                self.logger.warning("M400 after homing failed — raising Z anyway")
            self.logger.info(
                f"Raising Z to safety height "
                f"({self.config.safety_height} mm)"
            )
            self.move_to(
                z=self.config.safety_height,
                callback=_on_z_safety_raise_done
            )

        def _finish_homing():
            """Final step after G28: safety raise (if Z) or notify."""
            if has_z:
                # Wait for homing moves to physically finish, then raise
                self._protocol.send_command(
                    'M400', callback=_do_post_homing_safety_raise,
                    timeout_ms=60000,
                )
            else:
                _notify_referenced()
                if callback:
                    callback(True)

        def _do_home(_ok: bool = True, _resp: str = ""):
            """Send the combined G28 for all requested axes."""
            self.logger.info(f"Homing {homed_axes} (combined G28)")

            def on_homed(success: bool, response: str):
                if not success:
                    _on_fail(f"Homing failed for {homed_axes}: {response}")
                    return

                # Extract position from G28 response if available
                for line in response.split('\n'):
                    pos = self._parse_position_line(line)
                    if pos:
                        self._update_position(pos)
                        # Re-anchor the relative-move base to the freshly
                        # homed position.
                        self._set_commanded_target(pos.x, pos.y, pos.z)
                        break

                # Mark these axes as referenced
                for axis in homed_axes:
                    if axis in self._referenced:
                        self._referenced[axis] = True

                self.logger.info(f"Axes {homed_axes} referenced successfully")
                _finish_homing()

            g28_axes = ' '.join(homed_axes)
            self._protocol.send_command(
                f'G28 {g28_axes} F{self.config.homing_speed}',
                callback=on_homed,
                timeout_ms=60000,
            )

        def _pre_home_safety_lift():
            """Lift probe clear of the sample before homing (Z unhomed).

            Soft endstops are disabled so Marlin accepts the relative Z
            move on an axis it does not yet consider referenced. The
            chain is strictly serialized via the protocol's command
            queue; M400 guarantees the lift physically completes before
            G28 begins driving X/Y toward their endstops.
            """
            self.logger.info(
                f"Pre-homing safety lift: Z +{self.config.pre_home_z_lift} mm"
            )

            # Queued in order; the protocol processes them serially.
            self._protocol.send_command('M211 S0', timeout_ms=10000)  # disable soft endstops
            self._protocol.send_command('G91', timeout_ms=10000)      # relative
            # retries=0: this is a RELATIVE move. Re-sending it on a
            # spurious timeout would lift twice (Z +40 instead of +20),
            # so never auto-retry it. The M400 below confirms completion.
            self._protocol.send_command(
                f'G1 Z{self.config.pre_home_z_lift} F{self.config.homing_speed}',
                timeout_ms=30000,
                retries=0,
            )

            def on_lift_settled(ok: bool, _resp: str):
                if not ok:
                    _on_fail("M400 after pre-homing Z lift failed")
                    return
                self._protocol.send_command('M211 S1', timeout_ms=10000)  # re-enable
                self._protocol.send_command('G90', timeout_ms=10000)      # absolute
                _do_home()

            # M400 barrier: wait for the lift move to finish, then home.
            self._protocol.send_command(
                'M400', callback=on_lift_settled, timeout_ms=60000
            )

        # --- Start the chain ---
        if has_z:
            # Lift first, then home everything together so Z_SAFE_HOMING
            # repositions XY before probing Z.
            _pre_home_safety_lift()
        elif has_xy:
            # XY-only home: no Z risk, no lift needed.
            _do_home()
        else:
            self.logger.warning(f"No valid axes to home: {axes}")
            self._start_keepalive()
            return False

        return True

    def get_position(self) -> Tuple[float, float, float]:
        """
        Query current position from printer (non-blocking).

        Queues an M114 command. When the response arrives, the cached
        position is updated and ``_on_position_changed`` is called.

        Returns:
            Current cached position (may not yet reflect the queued query).
        """
        if not self._protocol or not self._protocol.is_running:
            return tuple(self._position)

        def on_response(success: bool, response: str):
            if success:
                for line in response.split('\n'):
                    pos = self._parse_position_line(line)
                    if pos:
                        self._update_position(pos)
                        return
                # ACK received but no position data — log for debugging
                self.logger.warning(
                    f"M114 ACK received but position not parsed. "
                    f"Response: {response!r}"
                )
            else:
                self.logger.debug(f"M114 failed: {response!r}")

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
                pos = self._parse_position_line(line)
                if pos:
                    self._update_position(pos)
                    return (pos.x, pos.y, pos.z)

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

        # Build G-code command. Track the per-axis targets actually
        # commanded (referenced axes only) so relative moves can use them
        # as their accumulation base.
        gcode = "G0"
        tx = ty = tz = None
        if x is not None and self._referenced['X']:
            gcode += f" X{x:.2f}"
            tx = x
        if y is not None and self._referenced['Y']:
            gcode += f" Y{y:.2f}"
            ty = y
        if z is not None and self._referenced['Z']:
            gcode += f" Z{z:.2f}"
            tz = z

        if gcode == "G0":
            self.logger.warning("No valid axes specified for move_to")
            return False

        # Record the commanded absolute target before dispatching so a
        # following move_by accumulates from here, not from a stale read.
        self._set_commanded_target(tx, ty, tz)

        gcode += f" F{speed or self._speed}"

        def on_complete(success: bool, response: str):
            if success:
                # Wait for motion to physically finish before querying position.
                # Marlin ACKs G0/G1 when the command is buffered, not when
                # the move is done.  M400 blocks until all moves finish.
                def on_settled(ok: bool, _resp: str):
                    self.get_position()
                    if self._on_movement_complete:
                        self._on_movement_complete()
                    if callback:
                        callback(success)

                self._protocol.send_command(
                    'M400', callback=on_settled, timeout_ms=30000
                )
                return
            else:
                self.logger.error(f"Move failed: {response}")

            if callback:
                callback(success)

        # Chain: G90 (wait for ACK) -> G0 move -> M400
        # G90 must be confirmed before sending the move to guarantee
        # the firmware is in absolute positioning mode.
        def on_absolute_set(success: bool, response: str):
            if success:
                self._protocol.send_command(gcode, callback=on_complete)
            else:
                self.logger.error(f"Failed to set absolute mode before move: {response}")
                if callback:
                    callback(False)

        self._protocol.send_command('G90', callback=on_absolute_set, timeout_ms=5000)
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
        Move by a relative distance, dispatched as an ABSOLUTE move.

        The delta is added to the last commanded absolute target
        (``self._target_position``) and sent via ``move_to``. Keeping every
        G-code move absolute means:
          * No global G90/G91 mode switching, so a move can never run in
            the wrong positioning mode if commands ever interleave.
          * Retries are safe — re-sending an absolute target is a no-op,
            whereas a relative move would compound the distance.
          * Successive jogs still accumulate correctly, because the base
            is the commanded target rather than a possibly-stale measured
            position.

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

        # Accumulate from the last commanded target (not the measured
        # position) so rapid successive jogs add up correctly.
        with self._position_lock:
            base_x = self._target_position.x
            base_y = self._target_position.y
            base_z = self._target_position.z

        target_x = base_x + x if x is not None else None
        target_y = base_y + y if y is not None else None
        target_z = base_z + z if z is not None else None

        if target_x is None and target_y is None and target_z is None:
            self.logger.warning("No valid axes specified for move_by")
            return False

        # Delegate to the absolute mover, which records the new commanded
        # target and applies the G90 + G0 + M400 chain.
        return self.move_to(
            x=target_x, y=target_y, z=target_z,
            speed=speed, callback=callback,
        )

    def go_to_home(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        callback: Optional[Callable[[bool], None]] = None,
    ) -> bool:
        """
        Move to home position (with safety height).

        Moves are chained via callbacks: raise Z -> move XY -> lower Z.
        Fully non-blocking.

        Args:
            x, y, z: Override target coordinates. When provided these are
                used instead of the configured home position. This allows
                the caller to supply view-adjusted (physical) coordinates.
            callback: Function called when complete

        Returns:
            True if commands sent successfully
        """
        if not self.all_referenced:
            self.logger.warning("Cannot go to home - not all axes referenced")
            return False

        target_x = x if x is not None else self.config.home_x
        target_y = y if y is not None else self.config.home_y
        target_z = z if z is not None else self.config.home_z

        def on_z_lowered(success: bool):
            if not success:
                self.logger.error("Go home: failed to lower Z")
            if callback:
                callback(success)

        def on_xy_moved(success: bool):
            if success:
                self.move_to(z=target_z, callback=on_z_lowered)
            else:
                self.logger.error("Go home: failed to move XY")
                if callback:
                    callback(False)

        def on_z_raised(success: bool):
            if success:
                self.move_to(
                    x=target_x,
                    y=target_y,
                    callback=on_xy_moved
                )
            else:
                self.logger.error("Go home: failed to raise Z")
                if callback:
                    callback(False)

        # Start the chain: raise to safety height
        self.move_to(z=self.config.safety_height, callback=on_z_raised)

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
