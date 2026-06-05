"""
State machine for acquisition workflow coordination.

Uses the `transitions` library to manage workflow states:
- Idle → Referencing → Ready → Acquiring → Processing
- Error handling and recovery
- State change callbacks for UI updates
"""

from typing import Optional, Callable, List, Dict, Any
from enum import Enum, auto
from transitions import Machine
from PyQt6.QtCore import QObject, pyqtSignal

from raman_control.config.logging_config import get_logger

logger = get_logger('state_machine')


class AcquisitionState(Enum):
    """Possible states for the acquisition workflow."""
    IDLE = auto()           # Initial state, disconnected
    CONNECTING = auto()     # Connecting to hardware
    CONNECTED = auto()      # Connected, not referenced
    REFERENCING = auto()    # Homing axes
    READY = auto()          # Referenced, ready to acquire
    MOVING = auto()         # Moving to position
    FOCUSING = auto()       # Running autofocus
    ACQUIRING = auto()      # Acquiring spectrum
    PROCESSING = auto()     # Processing data
    PAUSED = auto()         # Acquisition paused
    ERROR = auto()          # Error state


class AcquisitionStateMachine(QObject):
    """
    State machine for managing acquisition workflow.

    Signals:
        state_changed: Emitted when state changes (old_state, new_state)
        progress_updated: Emitted with progress info (current, total, message)
        error_occurred: Emitted on error (error_message)
        acquisition_complete: Emitted when acquisition finishes

    States:
        IDLE → CONNECTING → CONNECTED → REFERENCING → READY
        READY ↔ MOVING ↔ FOCUSING ↔ ACQUIRING → PROCESSING → READY
        Any state → ERROR → IDLE

    Example:
        sm = AcquisitionStateMachine()
        sm.state_changed.connect(self.on_state_change)

        sm.connect()         # IDLE → CONNECTING
        sm.on_connected()    # CONNECTING → CONNECTED
        sm.reference()       # CONNECTED → REFERENCING
        sm.on_referenced()   # REFERENCING → READY

        sm.start_acquisition()  # Begin workflow
    """

    state_changed = pyqtSignal(str, str)  # old_state, new_state
    progress_updated = pyqtSignal(int, int, str)  # current, total, message
    error_occurred = pyqtSignal(str)
    acquisition_complete = pyqtSignal()

    # State names for transitions library
    states = [s.name.lower() for s in AcquisitionState]

    # Define valid transitions
    transitions = [
        # Connection
        {'trigger': 'connect', 'source': 'idle', 'dest': 'connecting'},
        {'trigger': 'on_connected', 'source': 'connecting', 'dest': 'connected'},
        {'trigger': 'disconnect', 'source': '*', 'dest': 'idle'},

        # Referencing
        {'trigger': 'reference', 'source': 'connected', 'dest': 'referencing'},
        {'trigger': 'on_referenced', 'source': 'referencing', 'dest': 'ready'},

        # Acquisition workflow
        {'trigger': 'start_acquisition', 'source': 'ready', 'dest': 'moving'},
        {'trigger': 'on_position_reached', 'source': 'moving', 'dest': 'focusing'},
        {'trigger': 'on_focused', 'source': 'focusing', 'dest': 'acquiring'},
        {'trigger': 'on_spectrum_acquired', 'source': 'acquiring', 'dest': 'processing'},
        {'trigger': 'on_processed', 'source': 'processing', 'dest': 'moving',
         'conditions': 'has_more_positions'},
        {'trigger': 'on_processed', 'source': 'processing', 'dest': 'ready',
         'unless': 'has_more_positions'},

        # Pause/Resume
        {'trigger': 'pause', 'source': ['moving', 'focusing', 'acquiring'], 'dest': 'paused'},
        {'trigger': 'resume', 'source': 'paused', 'dest': 'moving'},

        # Skip position
        {'trigger': 'skip', 'source': ['focusing', 'acquiring'], 'dest': 'moving',
         'conditions': 'has_more_positions'},

        # Error handling
        {'trigger': 'error', 'source': '*', 'dest': 'error'},
        {'trigger': 'recover', 'source': 'error', 'dest': 'ready'},
        {'trigger': 'reset', 'source': '*', 'dest': 'idle'},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize state machine
        self._machine = Machine(
            model=self,
            states=self.states,
            transitions=self.transitions,
            initial='idle',
            auto_transitions=False,
            send_event=True,
            after_state_change='_on_state_changed'
        )

        # Acquisition progress tracking
        self._positions: List[tuple] = []
        self._current_position_index = 0
        self._acquired_data: List[Dict[str, Any]] = []

        # Error message
        self._last_error = ""

    @property
    def current_state(self) -> str:
        """Get current state name."""
        return self.state

    @property
    def is_busy(self) -> bool:
        """Check if currently in an active workflow state."""
        busy_states = ['connecting', 'referencing', 'moving', 'focusing', 'acquiring', 'processing']
        return self.state in busy_states

    @property
    def ready_to_acquire(self) -> bool:
        """Check if ready to start acquisition."""
        return self.state == 'ready'

    @property
    def progress(self) -> tuple:
        """Get current progress (current_index, total_count)."""
        return (self._current_position_index, len(self._positions))

    @property
    def last_error(self) -> str:
        """Get the last error message."""
        return self._last_error

    def has_more_positions(self, event=None) -> bool:
        """Condition: Check if more positions to acquire."""
        return self._current_position_index < len(self._positions) - 1

    def _on_state_changed(self, event):
        """Called after every state transition."""
        old_state = event.transition.source
        new_state = event.transition.dest
        logger.info(f"State transition: {old_state} -> {new_state}")
        self.state_changed.emit(old_state, new_state)

    def set_positions(self, positions: List[tuple]):
        """Set the list of positions to acquire."""
        self._positions = positions
        self._current_position_index = 0
        self._acquired_data = []
        logger.info(f"Set {len(positions)} positions for acquisition")

    def get_current_position(self) -> Optional[tuple]:
        """Get the current target position."""
        if 0 <= self._current_position_index < len(self._positions):
            return self._positions[self._current_position_index]
        return None

    def advance_position(self) -> bool:
        """Advance to the next position."""
        if self._current_position_index < len(self._positions) - 1:
            self._current_position_index += 1
            self.progress_updated.emit(
                self._current_position_index + 1,
                len(self._positions),
                f"Position {self._current_position_index + 1}/{len(self._positions)}"
            )
            return True
        return False

    def store_spectrum(self, data: Dict[str, Any]):
        """Store acquired spectrum data."""
        data['position_index'] = self._current_position_index
        data['position'] = self.get_current_position()
        self._acquired_data.append(data)
        logger.debug(f"Stored spectrum for position {self._current_position_index}")

    def get_acquired_data(self) -> List[Dict[str, Any]]:
        """Get all acquired spectrum data."""
        return self._acquired_data.copy()

    def report_error(self, message: str):
        """Report an error and transition to error state."""
        self._last_error = message
        logger.error(f"Acquisition error: {message}")
        self.error_occurred.emit(message)
        try:
            self.error()  # Transition to error state
        except Exception:
            pass  # Already in error state or invalid transition

    def finish_acquisition(self):
        """Called when acquisition workflow completes."""
        logger.info(f"Acquisition complete. Acquired {len(self._acquired_data)} spectra.")
        self.acquisition_complete.emit()

    # State entry callbacks (can be overridden or connected to signals)
    def on_enter_idle(self, event=None):
        """Called when entering IDLE state."""
        self._positions = []
        self._current_position_index = 0
        self._acquired_data = []

    def on_enter_error(self, event=None):
        """Called when entering ERROR state."""
        logger.warning(f"Entered error state: {self._last_error}")

    def on_enter_ready(self, event=None):
        """Called when entering READY state."""
        if self._acquired_data and self._current_position_index >= len(self._positions) - 1:
            self.finish_acquisition()
