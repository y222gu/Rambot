"""
PyQt6 thread for motion control.

This module provides a QThread wrapper for the StepperStage,
enabling thread-safe GUI updates via Qt signals.
"""

import threading
from typing import Optional, Tuple
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

from raman_control.hardware.stepper_stage import StepperStage, StagePosition
from raman_control.config.hardware_config import StageConfig
from raman_control.config.logging_config import get_logger

logger = get_logger('motion_thread')


class MotionThread(QThread):
    """
    Qt thread for managing stage motion.

    Signals:
        position_changed: Emitted when position updates (x, y, z)
        axis_referenced: Emitted when an axis is homed (axis_name)
        movement_complete: Emitted when a movement finishes
        connected: Emitted when connection established
        disconnected: Emitted when connection lost
        error: Emitted on errors (error_message)

    Usage:
        motion = MotionThread(config)
        motion.position_changed.connect(self.on_position_update)
        motion.start()

        motion.request_move_to(100, 100, 30)
        motion.request_home('XYZ')
    """

    # Signals for thread-safe GUI updates
    position_changed = pyqtSignal(float, float, float)
    axis_referenced = pyqtSignal(str)
    movement_complete = pyqtSignal()
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, config: Optional[StageConfig] = None, parent=None):
        """
        Initialize the motion thread.

        Args:
            config: Stage configuration
            parent: Parent QObject
        """
        super().__init__(parent)

        self._config = config or StageConfig()
        self._stage: Optional[StepperStage] = None
        self._running = False
        self._mutex = QMutex()

    @property
    def is_connected(self) -> bool:
        """Check if stage is connected."""
        with QMutexLocker(self._mutex):
            return self._stage is not None and self._stage.is_connected

    @property
    def is_referenced(self) -> dict:
        """Get reference status for each axis."""
        with QMutexLocker(self._mutex):
            if self._stage:
                return self._stage.is_referenced
            return {'X': False, 'Y': False, 'Z': False}

    @property
    def all_referenced(self) -> bool:
        """Check if all axes are referenced."""
        with QMutexLocker(self._mutex):
            if self._stage:
                return self._stage.all_referenced
            return False

    @property
    def speed(self) -> int:
        """Get current movement speed."""
        with QMutexLocker(self._mutex):
            if self._stage:
                return self._stage.speed
            return self._config.default_speed

    @speed.setter
    def speed(self, value: int) -> None:
        """Set movement speed."""
        with QMutexLocker(self._mutex):
            if self._stage:
                self._stage.speed = value

    def run(self) -> None:
        """Thread main loop - connects and manages stage."""
        logger.info("Motion thread starting")
        self._running = True

        # Create and configure stage
        self._stage = StepperStage(self._config)

        # Set up callbacks to emit signals
        self._stage.set_callbacks(
            on_position_changed=self._on_position_changed,
            on_referenced=self._on_referenced,
            on_movement_complete=self._on_movement_complete
        )

        # Connect to stage
        if not self._stage.start():
            self.error.emit("Failed to connect to stage")
            self._running = False
            return

        self.connected.emit()
        logger.info("Motion thread connected to stage")

        # Main loop - just keep thread alive
        while self._running:
            self.msleep(100)

        # Cleanup
        if self._stage:
            self._stage.stop()

        self.disconnected.emit()
        logger.info("Motion thread stopped")

    def stop(self) -> None:
        """Stop the motion thread."""
        self._running = False
        self.wait(5000)  # Wait up to 5 seconds

    def _on_position_changed(self, pos: StagePosition) -> None:
        """Internal callback - emit signal for position change."""
        self.position_changed.emit(pos.x, pos.y, pos.z)

    def _on_referenced(self, axis: str) -> None:
        """Internal callback - emit signal for axis referenced."""
        self.axis_referenced.emit(axis)

    def _on_movement_complete(self) -> None:
        """Internal callback - emit signal for movement complete."""
        self.movement_complete.emit()

    def request_connect(self) -> None:
        """Request connection to stage (call from GUI thread)."""
        # Connection happens in run(), this is for reconnection
        with QMutexLocker(self._mutex):
            if self._stage and not self._stage.is_connected:
                if self._stage.connect():
                    self._stage.start()
                    self.connected.emit()

    def request_disconnect(self) -> None:
        """Request disconnection from stage."""
        with QMutexLocker(self._mutex):
            if self._stage:
                self._stage.stop()
                self.disconnected.emit()

    def request_home(self, axes: str = 'XYZ') -> None:
        """
        Request homing of specified axes.

        Dispatches to a background thread so the GUI thread is never
        blocked by the homing sequence (including safety Z-raise).

        Args:
            axes: Axes to home ('X', 'Y', 'Z', 'XY', 'XYZ')
        """
        with QMutexLocker(self._mutex):
            if not self._stage or not self._stage.is_connected:
                self.error.emit("Stage not connected")
                return
            stage = self._stage

        def _do_home():
            def on_complete(success: bool):
                if not success:
                    self.error.emit(f"Homing failed for {axes}")

            stage.home(axes, callback=on_complete)

        t = threading.Thread(target=_do_home, name=f"Home-{axes}", daemon=True)
        t.start()

    def request_move_to(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        speed: Optional[int] = None
    ) -> None:
        """
        Request movement to absolute position.

        Args:
            x, y, z: Target coordinates (None to keep current)
            speed: Movement speed in mm/min
        """
        with QMutexLocker(self._mutex):
            if self._stage and self._stage.is_connected:
                def on_complete(success: bool):
                    if not success:
                        self.error.emit("Movement failed")

                self._stage.move_to(x=x, y=y, z=z, speed=speed, callback=on_complete)
            else:
                self.error.emit("Stage not connected")

    def request_move_by(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        speed: Optional[int] = None
    ) -> None:
        """
        Request movement by relative distance.

        Args:
            x, y, z: Relative distances (None to skip)
            speed: Movement speed in mm/min
        """
        with QMutexLocker(self._mutex):
            if self._stage and self._stage.is_connected:
                def on_complete(success: bool):
                    if not success:
                        self.error.emit("Movement failed")

                self._stage.move_by(x=x, y=y, z=z, speed=speed, callback=on_complete)
            else:
                self.error.emit("Stage not connected")

    def request_go_home(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
    ) -> None:
        """Request movement to home position.

        Args:
            x, y, z: Override target coordinates (physical). When provided
                these are forwarded to ``StepperStage.go_to_home`` instead
                of the configured home position.
        """
        with QMutexLocker(self._mutex):
            if not self._stage or not self._stage.is_connected:
                self.error.emit("Stage not connected")
                return
            stage = self._stage

        def _do_go_home():
            def on_complete(success: bool):
                if not success:
                    self.error.emit("Go home failed")

            stage.go_to_home(x=x, y=y, z=z, callback=on_complete)

        t = threading.Thread(target=_do_go_home, name="GoHome", daemon=True)
        t.start()

    def request_stop(self) -> None:
        """Request emergency stop."""
        with QMutexLocker(self._mutex):
            if self._stage:
                self._stage.stop_motion()

    def request_position_update(self) -> None:
        """Request position query."""
        with QMutexLocker(self._mutex):
            if self._stage and self._stage.is_connected:
                self._stage.get_position()

    def get_position_sync(self) -> Tuple[float, float, float]:
        """
        Get current position synchronously (blocking).

        Note: Prefer using position_changed signal for non-blocking updates.

        Returns:
            Tuple of (x, y, z)
        """
        with QMutexLocker(self._mutex):
            if self._stage and self._stage.is_connected:
                return self._stage.get_position_sync()
            return (0.0, 0.0, 0.0)

    def get_position(self) -> Optional[Tuple[float, float, float]]:
        """Get current position (alias for get_position_sync)."""
        return self.get_position_sync()

    def move_to(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        speed: Optional[int] = None
    ) -> None:
        """Convenience alias for request_move_to."""
        self.request_move_to(x=x, y=y, z=z, speed=speed)

    def move_z(self, z: float, speed: Optional[int] = None) -> None:
        """Move only Z axis to specified position."""
        self.request_move_to(z=z, speed=speed)

    def move_z_blocking(self, z: float, speed: Optional[int] = None,
                        timeout: float = 30.0) -> bool:
        """Move Z and block until the move completes.

        For use from a worker thread (e.g. autofocus) — NEVER call this on the
        GUI thread, it would freeze the UI. The mutex is held only while the
        command is queued; the wait happens outside it so the GUI thread is
        not blocked.
        """
        done = threading.Event()
        result = {'ok': False}

        def _cb(success: bool):
            result['ok'] = bool(success)
            done.set()

        with QMutexLocker(self._mutex):
            if not (self._stage and self._stage.is_connected):
                return False
            self._stage.move_to(z=z, speed=speed, callback=_cb)

        if not done.wait(timeout):
            logger.warning(f"move_z_blocking timed out moving to Z={z}")
            return False
        return result['ok']

    def move_xyz_blocking(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        speed: Optional[int] = None,
        timeout: float = 60.0,
    ) -> bool:
        """Move to an absolute (x, y, z) and block until the move completes.

        Like :meth:`move_z_blocking` but for full XYZ moves — for use from a
        worker thread (e.g. the acquisition loop). NEVER call this on the GUI
        thread; it would freeze the UI. Any axis left as None keeps its current
        value. The mutex is held only while the command is queued; the wait
        happens outside it so the GUI thread is never blocked.
        """
        done = threading.Event()
        result = {'ok': False}

        def _cb(success: bool):
            result['ok'] = bool(success)
            done.set()

        with QMutexLocker(self._mutex):
            if not (self._stage and self._stage.is_connected):
                return False
            self._stage.move_to(x=x, y=y, z=z, speed=speed, callback=_cb)

        if not done.wait(timeout):
            logger.warning(
                f"move_xyz_blocking timed out moving to ({x}, {y}, {z})")
            return False
        return result['ok']
