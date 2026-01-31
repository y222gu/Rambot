"""
Protocol panel widget for acquisition workflow control.

Features:
- Protocol configuration (positions, parameters)
- Workflow state display
- Start/Stop/Pause controls
- Progress tracking
- Sample grid integration
"""

from typing import Optional, List, Tuple
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTextEdit, QComboBox, QCheckBox,
    QFrame, QSplitter, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from raman_control.core.state_machine import AcquisitionStateMachine
from raman_control.gui.widgets.sample_grid import SampleGridWidget
from raman_control.gui.styles import COLORS
from raman_control.config.logging_config import get_logger

logger = get_logger('protocol_panel')


class StateIndicator(QLabel):
    """Visual indicator for workflow state."""

    STATE_COLORS = {
        'idle': '#606060',
        'connecting': '#ff9800',
        'connected': '#2196f3',
        'referencing': '#ff9800',
        'ready': '#4caf50',
        'moving': '#9c27b0',
        'focusing': '#e91e63',
        'acquiring': '#00bcd4',
        'processing': '#8bc34a',
        'paused': '#ffc107',
        'error': '#f44336',
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(36)
        self.set_state('idle')

    def set_state(self, state: str):
        color = self.STATE_COLORS.get(state.lower(), '#606060')
        self.setText(state.upper())
        self.setStyleSheet(f'''
            QLabel {{
                background-color: {color};
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 4px;
                padding: 8px 16px;
            }}
        ''')


class ProtocolPanelWidget(QWidget):
    """
    Protocol configuration and execution panel.

    Features:
    - Sample position selection via grid
    - Acquisition parameter configuration
    - Workflow control (start/pause/stop)
    - Progress tracking and logging
    """

    acquisition_started = pyqtSignal()
    acquisition_stopped = pyqtSignal()
    position_requested = pyqtSignal(int, int)  # row, col

    def __init__(
        self,
        state_machine: Optional[AcquisitionStateMachine] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        # Create state machine if not provided
        self._state_machine = state_machine or AcquisitionStateMachine()

        # Build UI
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)

        # Left panel - Sample grid and position selection
        left_panel = self._create_position_panel()
        main_layout.addWidget(left_panel, 1)

        # Right panel - Parameters and control
        right_panel = self._create_control_panel()
        main_layout.addWidget(right_panel, 1)

    def _create_position_panel(self) -> QWidget:
        """Create the position selection panel."""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Sample grid
        self._sample_grid = SampleGridWidget()
        self._sample_grid.set_multi_select(True)
        self._sample_grid.selection_changed.connect(self._on_selection_changed)
        layout.addWidget(self._sample_grid)

        # Selection info
        info_frame = QFrame()
        info_frame.setStyleSheet(f'''
            QFrame {{
                background-color: {COLORS['bg_medium']};
                border: 1px solid {COLORS['border_dark']};
                border-radius: 4px;
            }}
        ''')
        info_layout = QHBoxLayout(info_frame)

        self._selection_label = QLabel("No positions selected (drag to select multiple)")
        self._selection_label.setStyleSheet(f'color: {COLORS["text_secondary"]};')
        info_layout.addWidget(self._selection_label)

        info_layout.addStretch()

        layout.addWidget(info_frame)

        return panel

    def _create_control_panel(self) -> QWidget:
        """Create the control and parameters panel."""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # State indicator
        state_group = QGroupBox("Workflow State")
        state_layout = QVBoxLayout(state_group)
        self._state_indicator = StateIndicator()
        state_layout.addWidget(self._state_indicator)
        layout.addWidget(state_group)

        # Acquisition parameters
        params_group = QGroupBox("Acquisition Parameters")
        params_layout = QGridLayout = self._create_parameters_layout(params_group)
        layout.addWidget(params_group)

        # Control buttons
        control_group = QGroupBox("Control")
        control_layout = QHBoxLayout(control_group)

        self._start_btn = QPushButton("▶ Start")
        self._start_btn.setProperty('primary', True)
        self._start_btn.setMinimumHeight(40)
        self._start_btn.clicked.connect(self._on_start_clicked)
        control_layout.addWidget(self._start_btn)

        self._pause_btn = QPushButton("⏸ Pause")
        self._pause_btn.setMinimumHeight(40)
        self._pause_btn.setEnabled(False)
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        control_layout.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("⏹ Stop")
        self._stop_btn.setProperty('danger', True)
        self._stop_btn.setMinimumHeight(40)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        control_layout.addWidget(self._stop_btn)

        layout.addWidget(control_group)

        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        self._progress_bar.setStyleSheet(f'''
            QProgressBar {{
                border: 1px solid {COLORS['border_dark']};
                border-radius: 4px;
                background-color: {COLORS['bg_light']};
                height: 20px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['accent_primary']};
                border-radius: 3px;
            }}
        ''')
        progress_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("Ready")
        self._progress_label.setStyleSheet(f'color: {COLORS["text_secondary"]};')
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self._progress_label)

        layout.addWidget(progress_group)

        # Data save settings
        data_group = QGroupBox("Data Save Settings")
        data_layout = QVBoxLayout(data_group)

        # Save location
        from PyQt6.QtWidgets import QFileDialog, QLineEdit

        save_loc_layout = QHBoxLayout()
        save_loc_layout.addWidget(QLabel("Save to:"))
        self._save_path_edit = QLineEdit()
        self._save_path_edit.setPlaceholderText("Select data save directory...")
        self._save_path_edit.setReadOnly(True)
        save_loc_layout.addWidget(self._save_path_edit, 1)

        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._on_browse_save_path)
        save_loc_layout.addWidget(self._browse_btn)
        data_layout.addLayout(save_loc_layout)

        # Filename template
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Filename:"))
        self._filename_edit = QLineEdit()
        self._filename_edit.setText("sample_{date}_{time}_{pos}")
        self._filename_edit.setToolTip(
            "Available placeholders:\n"
            "{date} - Current date (YYYYMMDD)\n"
            "{time} - Current time (HHMMSS)\n"
            "{pos} - Position index\n"
            "{row} - Row number\n"
            "{col} - Column number"
        )
        name_layout.addWidget(self._filename_edit, 1)
        data_layout.addLayout(name_layout)

        layout.addWidget(data_group)

        # Log output
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(150)
        self._log_text.setStyleSheet(f'''
            QTextEdit {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['border_dark']};
                border-radius: 4px;
                font-family: "SF Mono", "Consolas", monospace;
                font-size: 11px;
                color: {COLORS['text_secondary']};
            }}
        ''')
        log_layout.addWidget(self._log_text)

        layout.addWidget(log_group)

        layout.addStretch()

        return panel

    def _create_parameters_layout(self, parent: QGroupBox):
        """Create the parameters form layout."""
        from PyQt6.QtWidgets import QGridLayout

        layout = QGridLayout(parent)
        layout.setSpacing(8)

        row = 0

        # Integration time
        layout.addWidget(QLabel("Integration Time:"), row, 0)
        self._integration_spin = QSpinBox()
        self._integration_spin.setRange(1, 10000)
        self._integration_spin.setValue(100)
        self._integration_spin.setSuffix(" ms")
        layout.addWidget(self._integration_spin, row, 1)
        row += 1

        # Laser power (in mW)
        layout.addWidget(QLabel("Laser Power:"), row, 0)
        self._power_spin = QSpinBox()
        self._power_spin.setRange(0, 500)
        self._power_spin.setValue(100)
        self._power_spin.setSuffix(" mW")
        layout.addWidget(self._power_spin, row, 1)
        row += 1

        # Averaging
        layout.addWidget(QLabel("Averaging:"), row, 0)
        self._averaging_spin = QSpinBox()
        self._averaging_spin.setRange(1, 100)
        self._averaging_spin.setValue(1)
        self._averaging_spin.setSuffix(" scans")
        layout.addWidget(self._averaging_spin, row, 1)
        row += 1

        # Autofocus
        layout.addWidget(QLabel("Autofocus:"), row, 0)
        self._autofocus_check = QCheckBox("Enable")
        self._autofocus_check.setChecked(True)
        layout.addWidget(self._autofocus_check, row, 1)
        row += 1

        # Autofocus Z limits
        layout.addWidget(QLabel("AF Z Upper:"), row, 0)
        self._af_z_upper = QDoubleSpinBox()
        self._af_z_upper.setRange(0, 100)
        self._af_z_upper.setValue(40.0)
        self._af_z_upper.setSuffix(" mm")
        self._af_z_upper.setDecimals(1)
        layout.addWidget(self._af_z_upper, row, 1)
        row += 1

        layout.addWidget(QLabel("AF Z Lower:"), row, 0)
        self._af_z_lower = QDoubleSpinBox()
        self._af_z_lower.setRange(0, 100)
        self._af_z_lower.setValue(30.0)
        self._af_z_lower.setSuffix(" mm")
        self._af_z_lower.setDecimals(1)
        layout.addWidget(self._af_z_lower, row, 1)
        row += 1

        # Autofocus method
        layout.addWidget(QLabel("Focus Method:"), row, 0)
        self._focus_method_combo = QComboBox()
        self._focus_method_combo.addItems(["Spectrometer (Fast)", "Camera (Precise)", "Combined"])
        layout.addWidget(self._focus_method_combo, row, 1)

        return layout

    def _connect_signals(self):
        """Connect state machine signals."""
        self._state_machine.state_changed.connect(self._on_state_changed)
        self._state_machine.progress_updated.connect(self._on_progress_updated)
        self._state_machine.error_occurred.connect(self._on_error)
        self._state_machine.acquisition_complete.connect(self._on_acquisition_complete)

    def _on_state_changed(self, old_state: str, new_state: str):
        """Handle state machine state change."""
        self._state_indicator.set_state(new_state)
        self._log(f"State: {old_state} → {new_state}")

        # Update button states
        is_idle_or_ready = new_state in ('idle', 'ready', 'connected')
        is_running = new_state in ('moving', 'focusing', 'acquiring', 'processing')
        is_paused = new_state == 'paused'

        self._start_btn.setEnabled(is_idle_or_ready or is_paused)
        self._pause_btn.setEnabled(is_running)
        self._stop_btn.setEnabled(is_running or is_paused)

        # Update start button text
        if is_paused:
            self._start_btn.setText("▶ Resume")
        else:
            self._start_btn.setText("▶ Start")

    def _on_progress_updated(self, current: int, total: int, message: str):
        """Handle progress update."""
        if total > 0:
            percent = int(100 * current / total)
            self._progress_bar.setValue(percent)
            self._progress_label.setText(f"{message} ({percent}%)")
        self._log(message)

    def _on_error(self, message: str):
        """Handle error."""
        self._log(f"ERROR: {message}", error=True)

    def _on_acquisition_complete(self):
        """Handle acquisition complete."""
        self._log("Acquisition complete!")
        self._progress_bar.setValue(100)
        self._progress_label.setText("Complete")
        self.acquisition_stopped.emit()

    def _on_selection_changed(self, positions: list):
        """Handle sample grid selection change."""
        count = len(positions)
        self._selection_label.setText(f"{count} position{'s' if count != 1 else ''} selected")
        self._state_machine.set_positions(positions)

    def _on_start_clicked(self):
        """Handle start button click."""
        positions = self._sample_grid.get_selected_positions()
        if not positions:
            self._log("No positions selected", error=True)
            return

        self._state_machine.set_positions(positions)
        self._progress_bar.setValue(0)

        try:
            if self._state_machine.current_state == 'paused':
                self._state_machine.resume()
            elif self._state_machine.current_state == 'ready':
                self._state_machine.start_acquisition()
        except Exception as e:
            self._log(f"Cannot start: {e}", error=True)

        self.acquisition_started.emit()

    def _on_pause_clicked(self):
        """Handle pause button click."""
        try:
            self._state_machine.pause()
            self._log("Acquisition paused")
        except Exception as e:
            self._log(f"Cannot pause: {e}", error=True)

    def _on_stop_clicked(self):
        """Handle stop button click."""
        try:
            self._state_machine.reset()
            self._progress_bar.setValue(0)
            self._progress_label.setText("Stopped")
            self._log("Acquisition stopped")
            self.acquisition_stopped.emit()
        except Exception as e:
            self._log(f"Cannot stop: {e}", error=True)

    def _on_browse_save_path(self):
        """Open dialog to select save directory."""
        from PyQt6.QtWidgets import QFileDialog
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Data Save Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self._save_path_edit.setText(directory)
            self._log(f"Save path set to: {directory}")

    def _log(self, message: str, error: bool = False):
        """Add message to activity log."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = COLORS['error'] if error else COLORS['text_secondary']
        self._log_text.append(f'<span style="color: {color}">[{timestamp}] {message}</span>')
        # Auto-scroll to bottom
        scrollbar = self._log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def get_acquisition_parameters(self) -> dict:
        """Get current acquisition parameters."""
        return {
            'integration_time_ms': self._integration_spin.value(),
            'laser_power_mw': self._power_spin.value(),
            'averaging': self._averaging_spin.value(),
            'autofocus_enabled': self._autofocus_check.isChecked(),
            'autofocus_z_upper': self._af_z_upper.value(),
            'autofocus_z_lower': self._af_z_lower.value(),
            'focus_method': self._focus_method_combo.currentText(),
            'save_path': self._save_path_edit.text(),
            'filename_template': self._filename_edit.text(),
        }

    def set_state_machine(self, state_machine: AcquisitionStateMachine):
        """Set or update the state machine."""
        self._state_machine = state_machine
        self._connect_signals()
