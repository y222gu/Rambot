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
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTextEdit, QComboBox, QCheckBox,
    QRadioButton, QButtonGroup, QFrame, QSplitter, QScrollArea
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
        stage_control=None,
        live_view=None,
        spectrum_viewer=None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        # Create state machine if not provided
        self._state_machine = state_machine or AcquisitionStateMachine()

        # Live-tab widgets whose parameters are mirrored here. Editing a mirror
        # and pressing Start pushes the values back to keep the Live tab in sync.
        self._stage_control = stage_control
        self._live_view = live_view
        self._spectrum_viewer = spectrum_viewer
        # key -> (local spin box, callable returning the live target spin box)
        self._bindings: dict = {}

        # Build UI
        self._setup_ui()
        self._connect_signals()

        # Initialise the mirrors from the current Live-tab values.
        self.pull_from_live()

    # --- parameter mirroring -------------------------------------------------
    @staticmethod
    def _live_spin(widget, attr):
        return getattr(widget, attr, None) if widget is not None else None

    def _mirror(self, grid, row, key, label, get_target):
        """Add a local spin box mirroring a Live-tab spin box; register a
        binding so values can be pulled/pushed. Skips silently if the target
        widget isn't available."""
        target = get_target()
        if target is None:
            return row
        if isinstance(target, QDoubleSpinBox):
            spin = QDoubleSpinBox()
            spin.setDecimals(target.decimals())
        else:
            spin = QSpinBox()
        spin.setRange(target.minimum(), target.maximum())
        spin.setSingleStep(target.singleStep())
        if target.suffix():
            spin.setSuffix(target.suffix())
        spin.setValue(target.value())
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(spin, row, 1)
        self._bindings[key] = (spin, get_target)
        # Connect AFTER the initial setValue so setup doesn't trigger refresh.
        spin.valueChanged.connect(self._refresh_preview)
        return row + 1

    def pull_from_live(self):
        """Copy the current Live-tab values into the mirror fields."""
        for spin, get_target in self._bindings.values():
            t = get_target()
            if t is not None:
                spin.blockSignals(True)
                spin.setValue(t.value())
                spin.blockSignals(False)

    def push_to_live(self):
        """Write the mirror field values back to the Live-tab widgets."""
        for spin, get_target in self._bindings.values():
            t = get_target()
            if t is not None:
                t.setValue(spin.value())

    def showEvent(self, event):
        # Whenever the Auto Acquisition tab is shown, refresh the mirrors so
        # they reflect whatever the user last set in the Live tab.
        super().showEvent(event)
        self.pull_from_live()
        self._refresh_preview()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)

        # Left column - all parameter panels + controls
        main_layout.addWidget(self._create_control_panel(), 3)

        # Middle column - the sample droplet map
        main_layout.addWidget(self._create_position_panel(), 3)

        # Right column - live preview of every step the parameters will generate
        main_layout.addWidget(self._create_preview_panel(), 2)

        # Reflect the initial acquisition mode (enables/disables the grid)
        self._on_acquisition_mode_changed()
        self._refresh_preview()

    def _create_position_panel(self) -> QWidget:
        """Create the position selection panel."""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Sample grid (multi-select for acquisition)
        self._sample_grid = SampleGridWidget(multi_select=True)
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

    # --- step preview --------------------------------------------------------
    def _create_preview_panel(self) -> QWidget:
        """Right column: a live, read-only list of every step the current
        parameters + selection will generate."""
        group = QGroupBox("Acquisition Steps Preview")
        layout = QVBoxLayout(group)
        self._preview_text = QTextEdit()
        self._preview_text.setReadOnly(True)
        self._preview_text.setStyleSheet(f'''
            QTextEdit {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['border_dark']};
                border-radius: 4px;
                font-family: "SF Mono", "Consolas", monospace;
                font-size: 11px;
                color: {COLORS['text_primary']};
            }}
        ''')
        layout.addWidget(self._preview_text)
        return group

    def _refresh_preview(self):
        """Rebuild the step list from the current parameters + selection."""
        if not hasattr(self, '_preview_text'):
            return
        from raman_control.gui.widgets.sample_grid import well_name

        p = self.get_acquisition_parameters()
        mode = p['acquisition_mode']
        if mode == 'Current FoV':
            pos_names = ["Current FoV"]
        else:
            pos_names = [well_name(r, c)
                         for (r, c) in self._sample_grid.get_selected_positions()]

        repeats = p['repeat_each_sample']
        reps_per_point = p['reps_per_point']
        averaging = p['averaging']
        expo = self._binding_value('acq_exposure')
        power = self._binding_value('acq_power')
        expo_s = f"{expo} ms" if expo is not None else "exposure"
        power_s = f"{power} mW" if power is not None else "power"

        af_steps = []
        if p['autofocus_widefield_prusa']:
            af_steps.append("Autofocus widefield camera with Prusa")
        if p['autofocus_wasatch_prusa']:
            af_steps.append("Autofocus Wasatch with Prusa")
        if p['autofocus_wasatch_nanodrive'] and not p['z_stack']:
            af_steps.append("Autofocus Wasatch with NanoDrive")

        if not pos_names:
            self._preview_text.setPlainText(
                "No positions selected.\n\nSelect wells in the Sample Droplet "
                "Map (middle), or switch to Current FoV mode.")
            return

        lines = [
            f"Mode: {mode}",
            (f"Positions: {len(pos_names)}" if mode != 'Current FoV'
             else "Positions: current field of view"),
            f"Sample repeats: {repeats}",
            "",
        ]
        n = 0
        for rep in range(repeats):
            if repeats > 1:
                lines.append(f"=== Sample repeat {rep + 1}/{repeats} ===")
            for name in pos_names:
                lines.append(f"[{name}]")
                n += 1
                lines.append(f"  {n}. Move to {name}")
                for af in af_steps:
                    n += 1
                    lines.append(f"  {n}. {af}")
                if p['wait_before_measure']:
                    n += 1
                    lines.append(f"  {n}. Wait {p['wait_time_s']} s with laser on")
                if p['z_stack']:
                    n += 1
                    lines.append(f"  {n}. Z-stack at point (NanoDrive range/step)")
                for k in range(reps_per_point):
                    n += 1
                    lines.append(f"  {n}. Acquire spectrum {k + 1}/{reps_per_point}"
                                 f"  ({expo_s}, {power_s}, avg {averaging})")
                n += 1
                lines.append(f"  {n}. Save data")
            lines.append("")

        header = f"Total steps: {n}\n" + "-" * 30 + "\n"
        self._preview_text.setPlainText(header + "\n".join(lines))

    def _create_control_panel(self) -> QWidget:
        """Create the control and parameters panel (scrollable)."""
        content = QFrame()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Acquisition mode (Current FoV vs Robotic Deposition)
        layout.addWidget(self._create_mode_group())

        # Acquisition parameters (protocol-specific: reps / repeat / averaging)
        params_group = QGroupBox("Acquisition Parameters")
        self._create_parameters_layout(params_group)
        layout.addWidget(params_group)

        # --- Mirrored Live-tab parameters (editable; pushed back on Start) ---
        layout.addWidget(self._create_spectrum_acq_group())
        layout.addWidget(self._create_af_range_group())
        layout.addWidget(self._create_camera_af_group())
        layout.addWidget(self._create_spec_af_group())
        layout.addWidget(self._create_pattern_group())

        # Acquisition options (autofocus toggles, Z-stack, wait)
        layout.addWidget(self._create_options_group())

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

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    # --- mirrored Live-tab parameter groups ---------------------------------
    def _create_spectrum_acq_group(self) -> QGroupBox:
        """Spectrum acquisition exposure / laser power (NOT the autofocus ones)."""
        group = QGroupBox("Spectrum Acquisition (Exposure & Power)")
        grid = QGridLayout(group)
        grid.setSpacing(8)
        r = 0
        sv = self._spectrum_viewer
        r = self._mirror(grid, r, 'acq_exposure', "Exposure Time:",
                         lambda: self._live_spin(sv, '_integration_spin'))
        r = self._mirror(grid, r, 'acq_power', "Laser Power:",
                         lambda: self._live_spin(sv, '_power_spin'))
        return group

    def _create_af_range_group(self) -> QGroupBox:
        """Autofocus Z ranges (widefield / objective / nanodrive)."""
        group = QGroupBox("Autofocus Range")
        grid = QGridLayout(group)
        grid.setSpacing(8)
        r = 0
        sc = self._stage_control
        for key, label, attr in [
            ('wf_z_upper', "Widefield Z Upper:", '_wf_z_upper'),
            ('wf_z_lower', "Widefield Z Lower:", '_wf_z_lower'),
            ('obj_z_upper', "Objective Z Upper:", '_obj_z_upper'),
            ('obj_z_lower', "Objective Z Lower:", '_obj_z_lower'),
            ('nano_upper', "NanoDrive Upper:", '_nano_upper'),
            ('nano_lower', "NanoDrive Lower:", '_nano_lower'),
        ]:
            r = self._mirror(grid, r, key, label,
                             lambda a=attr: self._live_spin(sc, a))
        return group

    def _create_camera_af_group(self) -> QGroupBox:
        """Camera (widefield) autofocus step count."""
        group = QGroupBox("Camera Autofocus")
        grid = QGridLayout(group)
        grid.setSpacing(8)
        lv = self._live_view
        self._mirror(grid, 0, 'cam_af_steps', "Camera AF Steps:",
                     lambda: self._live_spin(lv, '_af_steps_spin'))
        return group

    def _create_spec_af_group(self) -> QGroupBox:
        """Spectrometer autofocus parameters (own exposure/power, steps, reps)."""
        group = QGroupBox("Spectrometer Autofocus")
        grid = QGridLayout(group)
        grid.setSpacing(8)
        r = 0
        sv = self._spectrum_viewer
        for key, label, attr in [
            ('spec_af_coarse', "Prusa Coarse Steps:", '_af_prusa_steps'),
            ('spec_af_fine', "NanoDrive Fine Steps:", '_af_nano_steps'),
            ('spec_af_exposure', "AF Exposure:", '_af_expo'),
            ('spec_af_power', "AF Laser Power:", '_af_power'),
            ('spec_af_reps', "AF Reps:", '_af_reps'),
        ]:
            r = self._mirror(grid, r, key, label,
                             lambda a=attr: self._live_spin(sv, a))
        return group

    def _create_pattern_group(self) -> QGroupBox:
        """Sample-point pattern parameters (grid / ring / line / random)."""
        group = QGroupBox("Sample Pattern")
        grid = QGridLayout(group)
        grid.setSpacing(8)
        r = 0
        lv = self._live_view
        for key, label, attr in [
            ('grid_rows', "Grid Rows:", '_grid_rows'),
            ('grid_cols', "Grid Cols:", '_grid_cols'),
            ('grid_buffer', "Grid Buffer:", '_grid_buffer'),
            ('ring_rays', "Ring Rays:", '_ring_rays'),
            ('ring_rings', "Ring Rings:", '_ring_rings'),
            ('ring_interval', "Ring Interval:", '_ring_interval'),
            ('ring_buffer', "Ring Buffer:", '_ring_buffer'),
            ('line_points', "Line Points:", '_line_points'),
            ('line_buffer', "Line Buffer:", '_line_buffer'),
            ('rand_points', "Random Points:", '_rand_points'),
        ]:
            r = self._mirror(grid, r, key, label,
                             lambda a=attr: self._live_spin(lv, a))
        return group

    def _create_mode_group(self) -> QGroupBox:
        """Acquisition mode: Current FoV vs Robotic Deposition (sample drops)."""
        group = QGroupBox("Acquisition Mode")
        lay = QHBoxLayout(group)
        self._mode_group = QButtonGroup(self)
        self._mode_current_fov = QRadioButton("Current FoV")
        self._mode_robotic = QRadioButton("Robotic Deposition")
        self._mode_robotic.setChecked(True)
        self._mode_group.addButton(self._mode_current_fov)
        self._mode_group.addButton(self._mode_robotic)
        self._mode_current_fov.toggled.connect(self._on_acquisition_mode_changed)
        lay.addWidget(self._mode_current_fov)
        lay.addWidget(self._mode_robotic)
        lay.addStretch()
        return group

    def _create_options_group(self) -> QGroupBox:
        """Acquisition options: per-channel autofocus, Z-stack, wait-before-measure."""
        group = QGroupBox("Acquisition Options")
        lay = QVBoxLayout(group)
        lay.setSpacing(6)

        self._af_widefield_prusa = QCheckBox("Autofocus Widefield Camera with Prusa")
        self._af_widefield_prusa.setChecked(True)
        self._af_wasatch_prusa = QCheckBox("Autofocus Wasatch with Prusa")
        self._af_wasatch_prusa.setChecked(True)
        self._af_wasatch_nanodrive = QCheckBox("Autofocus Wasatch with NanoDrive")
        self._af_wasatch_nanodrive.setChecked(True)
        lay.addWidget(self._af_widefield_prusa)
        lay.addWidget(self._af_wasatch_prusa)
        lay.addWidget(self._af_wasatch_nanodrive)
        for cb in (self._af_widefield_prusa, self._af_wasatch_prusa,
                   self._af_wasatch_nanodrive):
            cb.toggled.connect(self._refresh_preview)

        # Z-stack: when on, the NanoDrive autofocus is redundant -> off + disabled
        self._z_stack_check = QCheckBox(
            "Z Stack at Each Point (uses Wasatch NanoDrive range/step)")
        self._z_stack_check.toggled.connect(self._on_z_stack_toggled)
        lay.addWidget(self._z_stack_check)

        # Optional wait (laser on, no spectra saved) before measuring at each point
        self._wait_check = QCheckBox("Wait with laser on before measuring")
        self._wait_check.toggled.connect(self._on_wait_toggled)
        lay.addWidget(self._wait_check)

        wait_row = QHBoxLayout()
        self._wait_time_label = QLabel("Wait time [s]:")
        wait_row.addWidget(self._wait_time_label)
        self._wait_time_spin = QSpinBox()
        self._wait_time_spin.setRange(0, 3600)
        self._wait_time_spin.setValue(5)
        self._wait_time_spin.valueChanged.connect(self._refresh_preview)
        wait_row.addWidget(self._wait_time_spin)
        wait_row.addStretch()
        lay.addLayout(wait_row)

        # Reflect initial states
        self._on_z_stack_toggled()
        self._on_wait_toggled()
        return group

    def _on_acquisition_mode_changed(self):
        """Sample-drop selection only applies to Robotic Deposition; for
        Current FoV the measurement happens at the live camera position."""
        robotic = self._mode_robotic.isChecked()
        self._sample_grid.set_grid_enabled(robotic)
        self._refresh_preview()

    def _on_z_stack_toggled(self):
        """When Z-stack is enabled the NanoDrive autofocus is redundant."""
        if self._z_stack_check.isChecked():
            self._af_wasatch_nanodrive.setChecked(False)
            self._af_wasatch_nanodrive.setEnabled(False)
        else:
            self._af_wasatch_nanodrive.setEnabled(True)
        self._refresh_preview()

    def _on_wait_toggled(self):
        """The wait-time entry only matters when the wait step is enabled."""
        on = self._wait_check.isChecked()
        self._wait_time_label.setEnabled(on)
        self._wait_time_spin.setEnabled(on)
        self._refresh_preview()

    def _create_parameters_layout(self, parent: QGroupBox):
        """Create the parameters form layout."""
        from PyQt6.QtWidgets import QGridLayout

        layout = QGridLayout(parent)
        layout.setSpacing(8)

        row = 0

        # Repetitions per point per sample drop
        layout.addWidget(QLabel("Repetitions per point:"), row, 0)
        self._reps_per_point_spin = QSpinBox()
        self._reps_per_point_spin.setRange(1, 1000)
        self._reps_per_point_spin.setValue(3)
        layout.addWidget(self._reps_per_point_spin, row, 1)
        row += 1

        # Number of times to repeat each whole sample
        layout.addWidget(QLabel("Repeat each sample:"), row, 0)
        self._repeat_sample_spin = QSpinBox()
        self._repeat_sample_spin.setRange(1, 1000)
        self._repeat_sample_spin.setValue(1)
        layout.addWidget(self._repeat_sample_spin, row, 1)
        row += 1

        # Averaging
        layout.addWidget(QLabel("Averaging:"), row, 0)
        self._averaging_spin = QSpinBox()
        self._averaging_spin.setRange(1, 100)
        self._averaging_spin.setValue(1)
        self._averaging_spin.setSuffix(" scans")
        layout.addWidget(self._averaging_spin, row, 1)
        row += 1

        for s in (self._reps_per_point_spin, self._repeat_sample_spin,
                  self._averaging_spin):
            s.valueChanged.connect(self._refresh_preview)

        return layout

    def _connect_signals(self):
        """Connect state machine signals."""
        self._state_machine.state_changed.connect(self._on_state_changed)
        self._state_machine.progress_updated.connect(self._on_progress_updated)
        self._state_machine.error_occurred.connect(self._on_error)
        self._state_machine.acquisition_complete.connect(self._on_acquisition_complete)

    def _on_state_changed(self, old_state: str, new_state: str):
        """Handle state machine state change."""
        self._log(f"State: {old_state} -> {new_state}")

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
        self._refresh_preview()

    def _on_start_clicked(self):
        """Handle start button click."""
        # Push the mirrored parameters back to the Live tab first, so both tabs
        # stay consistent for the run.
        self.push_to_live()
        self._log("Parameters synced to Live tab")

        positions = self._sample_grid.get_selected_positions()
        if self._mode_robotic.isChecked() and not positions:
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

    def _binding_value(self, key):
        """Current value of a mirrored parameter (or None if not available)."""
        b = self._bindings.get(key)
        return b[0].value() if b else None

    def get_acquisition_parameters(self) -> dict:
        """Get current acquisition parameters (protocol-specific + mirrored)."""
        params = {
            'acquisition_mode': ('Current FoV' if self._mode_current_fov.isChecked()
                                 else 'Robotic Deposition'),
            'reps_per_point': self._reps_per_point_spin.value(),
            'repeat_each_sample': self._repeat_sample_spin.value(),
            'averaging': self._averaging_spin.value(),
            'autofocus_widefield_prusa': self._af_widefield_prusa.isChecked(),
            'autofocus_wasatch_prusa': self._af_wasatch_prusa.isChecked(),
            'autofocus_wasatch_nanodrive': self._af_wasatch_nanodrive.isChecked(),
            'z_stack': self._z_stack_check.isChecked(),
            'wait_before_measure': self._wait_check.isChecked(),
            'wait_time_s': self._wait_time_spin.value(),
            'save_path': self._save_path_edit.text(),
            'filename_template': self._filename_edit.text(),
        }
        # Mirrored Live-tab parameters
        for key in self._bindings:
            params[key] = self._binding_value(key)
        return params

    def set_state_machine(self, state_machine: AcquisitionStateMachine):
        """Set or update the state machine."""
        self._state_machine = state_machine
        self._connect_signals()
