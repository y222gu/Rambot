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
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QAbstractSpinBox,
    QProgressBar, QTextEdit, QComboBox, QCheckBox,
    QRadioButton, QButtonGroup, QFrame, QSplitter, QScrollArea,
    QStackedWidget, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from raman_control.core.state_machine import AcquisitionStateMachine
from raman_control.gui.threads.acquisition_thread import AcquisitionWorker
from raman_control.gui.widgets.sample_grid import SampleGridWidget
from raman_control.gui.widgets.live_view import (
    ClickableLabel, CYAN_OUTLINE_BTN_STYLE)
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

    # Current-FoV acquisition is driven by a worker the host window owns (it has
    # the hardware). The panel just requests start / pause / resume / stop and
    # the host reports progress back via the on_acquisition_* slots.
    acquisition_requested = pyqtSignal(dict)   # Current-FoV acquisition params
    robotic_requested = pyqtSignal(dict)       # Robotic-Deposition acquisition params
    pause_requested = pyqtSignal()
    resume_requested = pyqtSignal()
    stop_requested = pyqtSignal()

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
        # Read-only autofocus Z-range mirrors: shown as "None" until the user
        # registers the ranges in the Live tab (see is_registered / pull).
        self._range_keys: set = set()

        # Current-FoV run state (the worker lives in the host window).
        self._cfov_running = False
        self._cfov_paused = False
        self._cfov_stopping = False
        # Guard against recursion when syncing the linked Z ranges.
        self._linking = False

        # Build UI
        self._setup_ui()
        self._connect_signals()

        # React to the Live tab generating / editing sampling points: this both
        # gates the Current-FoV mode (only available once points exist) and
        # refreshes the Current-FoV image + point count.
        if self._live_view is not None and hasattr(self._live_view, 'positions_generated'):
            self._live_view.positions_generated.connect(self._on_live_positions_changed)
        # Mirror the Live tab's point-edit Confirm/Cancel state into this tab's
        # FoV toolbar, whichever tab started (or ended) the edit.
        if self._live_view is not None and hasattr(self._live_view, 'point_edit_mode_changed'):
            self._live_view.point_edit_mode_changed.connect(
                self._on_fov_edit_mode_changed)

        # Re-pull + re-gate Start when the user registers the autofocus ranges.
        if (self._stage_control is not None
                and hasattr(self._stage_control, 'ranges_registered')):
            self._stage_control.ranges_registered.connect(
                self._on_ranges_registered)
        # Re-gate Start when the ranges are invalidated (e.g. on disconnect).
        if (self._stage_control is not None
                and hasattr(self._stage_control, 'registration_reset')):
            self._stage_control.registration_reset.connect(
                self._update_start_enabled)

        # Initialise the mirrors from the current Live-tab values.
        self.pull_from_live()
        self._update_current_fov_availability()
        self._refresh_fov_view()
        self._update_start_enabled()

    # XY distance (mm) past which switching to Current-FoV warns the user that
    # the stage has moved away from where the image was captured.
    _FOV_MOVE_WARN_MM = 0.5

    # Compact width for every numeric entry box (px).
    _ENTRY_WIDTH = 72

    # Fixed width of the left (parameters) column (px), sized to its content
    # plus the surrounding padding and scrollbar.
    _LEFT_COLUMN_WIDTH = 366

    # Unit shown after each entry box, keyed by its parameter key. Keys not
    # listed (counts: rows, reps, steps, …) have no unit. Rendering the unit as
    # a trailing label (instead of an in-box suffix) keeps the narrow boxes
    # readable.
    _UNITS = {
        'acq_exposure': 'ms', 'acq_power': 'mW',
        'wf_z_upper': 'mm', 'wf_z_lower': 'mm',
        'obj_z_upper': 'mm', 'obj_z_lower': 'mm',
        'nano_upper': 'µm', 'nano_lower': 'µm',
        'spec_af_exposure': 'ms', 'spec_af_power': 'mW',
        'grid_buffer': 'µm', 'ring_interval': 'µm', 'ring_buffer': 'µm',
        'line_buffer': 'µm',
    }

    # --- parameter mirroring -------------------------------------------------
    @staticmethod
    def _live_spin(widget, attr):
        return getattr(widget, attr, None) if widget is not None else None

    def _entry_with_unit(self, spin, unit: str):
        """Wrap a (narrowed) spin box and an optional trailing unit label in a
        left-aligned row, so the unit sits right after the box and spare width
        goes to the right."""
        spin.setMaximumWidth(self._ENTRY_WIDTH)
        box = QHBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        box.addWidget(spin)
        if unit:
            lbl = QLabel(unit)
            lbl.setStyleSheet(f'color: {COLORS["text_secondary"]};')
            box.addWidget(lbl)
        box.addStretch()
        return box

    def _mirror(self, grid, row, key, label, get_target, read_only=False):
        """Add a local spin box mirroring a Live-tab spin box; register a
        binding so values can be pulled/pushed. Skips silently if the target
        widget isn't available.

        ``read_only=True`` makes the mirror display-only (no typing, no step
        buttons, no push to the Live tab) with a tooltip pointing the user to
        the Live tab — used for the Prusa Z / NanoDrive ranges, which are
        registered in the Live tab's Autofocus panel for safety."""
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
        spin.setValue(target.value())
        # Unit is shown as a trailing label (see _UNITS); fall back to the
        # source's in-box suffix only if we have no mapping for this key.
        unit = self._UNITS.get(key, (target.suffix() or '').strip())
        grid.addWidget(QLabel(label), row, 0)
        grid.addLayout(self._entry_with_unit(spin, unit), row, 1)
        self._bindings[key] = (spin, get_target)
        if read_only:
            # Display-only: still mirrors the Live tab (pulled), but shown as
            # plain text — no entry-box chrome, buttons, focus or editing. Shows
            # "None" (special value at the minimum) until the ranges are
            # registered in the Live tab.
            spin.setReadOnly(True)
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            spin.setStyleSheet(
                'QAbstractSpinBox { background: transparent; border: none; '
                'padding: 0; }')
            spin.setSpecialValueText("None")
            self._range_keys.add(key)
            spin.setToolTip(
                "Set + Register the autofocus Z ranges in the Live tab → "
                "Autofocus panel. Shown as None here until registered.")
        else:
            # Connect AFTER the initial setValue so setup doesn't trigger a
            # push. Editing a mirror pushes the value straight to the Live tab so
            # the two tabs stay consistent live (not only on Start).
            spin.valueChanged.connect(
                lambda _=None, k=key: self._on_mirror_changed(k))
        return row + 1

    def _on_mirror_changed(self, key):
        """A mirror spin box changed: push it to its Live-tab target so the
        Live tab updates immediately, then refresh the preview."""
        b = self._bindings.get(key)
        if b is not None:
            spin, get_target = b
            target = get_target()
            if target is not None and target.value() != spin.value():
                target.setValue(spin.value())
        self._refresh_preview()

    def pull_from_live(self):
        """Copy the current Live-tab values into the mirror fields. The
        autofocus Z-range mirrors show 'None' (their minimum) until the ranges
        have been registered in the Live tab."""
        registered = (self._stage_control is not None
                      and self._stage_control.is_registered())
        for key, (spin, get_target) in self._bindings.items():
            t = get_target()
            if t is None:
                continue
            spin.blockSignals(True)
            if key in self._range_keys and not registered:
                spin.setValue(spin.minimum())  # -> "None" via specialValueText
            else:
                spin.setValue(t.value())
            spin.blockSignals(False)
        self._pull_pattern_type()

    def push_to_live(self):
        """Write the mirror field values back to the Live-tab widgets."""
        for spin, get_target in self._bindings.values():
            t = get_target()
            if t is not None:
                t.setValue(spin.value())
        self._push_pattern_type()

    def _af_offset_z(self) -> float:
        """Wide->objective Z offset (mm) used to keep the widefield and
        spectrometer Prusa Z ranges in sync, matching the Live tab's Register."""
        sc = self._stage_control
        cm = getattr(sc, '_coord_mgr', None) if sc is not None else None
        return float(getattr(cm, 'offset_z', 0.0) or 0.0) if cm is not None else 0.0

    def _link_af_ranges(self):
        """Keep the widefield (Widefield Camera Autofocus) and objective
        (Spectrometer Autofocus) Prusa Z ranges in sync via the wide->objective
        Z offset: editing one updates the other (objective = widefield + offset),
        and the paired mirror's own handler pushes it to the Live tab too."""
        b = self._bindings
        pairs = [('wf_z_upper', 'obj_z_upper'),
                 ('wf_z_lower', 'obj_z_lower')]
        for wf_key, ob_key in pairs:
            if wf_key not in b or ob_key not in b:
                continue
            wf_spin = b[wf_key][0]
            ob_spin = b[ob_key][0]

            def make(src, dst, sign):
                def _sync(_=None):
                    if self._linking:
                        return
                    self._linking = True
                    dst.setValue(src.value() + sign * self._af_offset_z())
                    self._linking = False
                return _sync

            wf_spin.valueChanged.connect(make(wf_spin, ob_spin, +1.0))
            ob_spin.valueChanged.connect(make(ob_spin, wf_spin, -1.0))

    def showEvent(self, event):
        # Whenever the Auto Acquisition tab is shown, refresh the mirrors so
        # they reflect whatever the user last set in the Live tab.
        super().showEvent(event)
        self.pull_from_live()
        self._update_current_fov_availability()
        self._refresh_fov_view()
        self._refresh_preview()
        self._update_start_enabled()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)

        # Left column - all parameter panels (fixed compact width).
        main_layout.addWidget(self._create_control_panel(), 0)

        # Middle column - the droplet map / captured image. Gets the largest
        # share of the freed-up space so the map/image can be bigger.
        main_layout.addWidget(self._create_position_panel(), 5)

        # Right column - steps preview + control / progress / data save
        main_layout.addWidget(self._create_preview_panel(), 3)

        # Keep the widefield and spectrometer (objective) Prusa Z ranges in
        # sync via the wide->objective offset.
        self._link_af_ranges()

        # Reflect the initial acquisition mode (enables/disables the grid)
        self._on_acquisition_mode_changed()
        self._refresh_preview()

    def _create_position_panel(self) -> QWidget:
        """Create the middle column.

        It holds a stacked widget with two pages:
          * Robotic Deposition -> the sample droplet map (well grid).
          * Current FoV        -> the captured + segmented image with the
                                  generated sample points overlaid.
        The active page follows the acquisition mode.
        """
        self._position_stack = QStackedWidget()

        # --- Page 0: sample droplet map (Robotic Deposition) ---
        grid_page = QFrame()
        grid_layout = QVBoxLayout(grid_page)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(8)

        # Sample grid (multi-select for acquisition)
        self._sample_grid = SampleGridWidget(multi_select=True)
        self._sample_grid.selection_changed.connect(self._on_selection_changed)
        # Loading/clearing a config changes the displayed sample-name prefix.
        self._sample_grid.config_loaded.connect(
            lambda _=None: self._update_filename_format())
        grid_layout.addWidget(self._sample_grid)

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
        grid_layout.addWidget(info_frame)

        self._position_stack.addWidget(grid_page)

        # --- Page 1: Current-FoV captured image with sample points ---
        fov_page = QFrame()
        fov_layout = QVBoxLayout(fov_page)
        fov_layout.setContentsMargins(0, 0, 0, 0)
        fov_layout.setSpacing(8)

        # Zoomable preview (scroll-wheel zoom + drag-pan, same widget the Live
        # tab uses) so points can be inspected and edited up close. It mirrors
        # the Live tab's composite pixmap, so clicks map to the same points.
        self._fov_preview = ClickableLabel()
        self._fov_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fov_preview.setWordWrap(True)
        self._fov_preview.setMinimumHeight(240)
        self._fov_preview.setStyleSheet(
            f'color: {COLORS["text_secondary"]}; '
            f'background-color: {COLORS["bg_light"]}; '
            f'border: 1px solid {COLORS["border_dark"]}; border-radius: 4px;')
        self._fov_preview.setText(
            "Capture an image, run segmentation, and generate sample points "
            "in the Live tab to use Current FoV.\n\n"
            "Scroll to zoom, drag to pan.")
        self._fov_preview.clicked.connect(self._on_fov_preview_clicked)
        fov_layout.addWidget(self._fov_preview, 1)

        # Point-editing toolbar: mirrors the Live tab's Add/Remove Points. It has
        # two states that swap in place (like the Live tab): the normal Add/Remove
        # row, and — once editing — a Confirm/Cancel row. The edit session itself
        # (snapshot + revert) lives in the Live tab, driven via begin/confirm/
        # cancel_point_edit and kept in sync by point_edit_mode_changed.
        self._fov_edit_normal = QWidget()
        normal_row = QHBoxLayout(self._fov_edit_normal)
        normal_row.setContentsMargins(0, 0, 0, 0)
        normal_row.setSpacing(6)
        self._fov_add_btn = QPushButton("Add Points")
        self._fov_add_btn.setStyleSheet(CYAN_OUTLINE_BTN_STYLE)
        self._fov_add_btn.setToolTip("Click the preview to add sample points")
        self._fov_add_btn.clicked.connect(lambda: self._begin_fov_edit('add'))
        self._fov_remove_btn = QPushButton("Remove Points")
        self._fov_remove_btn.setStyleSheet(CYAN_OUTLINE_BTN_STYLE)
        self._fov_remove_btn.setToolTip("Click a point in the preview to remove it")
        self._fov_remove_btn.clicked.connect(lambda: self._begin_fov_edit('delete'))
        normal_row.addWidget(self._fov_add_btn)
        normal_row.addWidget(self._fov_remove_btn)
        normal_row.addStretch()
        fov_layout.addWidget(self._fov_edit_normal)

        self._fov_edit_active = QWidget()
        active_col = QVBoxLayout(self._fov_edit_active)
        active_col.setContentsMargins(0, 0, 0, 0)
        active_col.setSpacing(6)
        self._fov_edit_hint = QLabel("")
        self._fov_edit_hint.setWordWrap(True)
        self._fov_edit_hint.setStyleSheet(
            f'color: {COLORS["text_secondary"]}; font-size: 11px;')
        active_col.addWidget(self._fov_edit_hint)
        confirm_row = QHBoxLayout()
        confirm_row.setSpacing(6)
        self._fov_confirm_btn = QPushButton("Confirm")
        self._fov_confirm_btn.clicked.connect(self._confirm_fov_edit)
        self._fov_cancel_btn = QPushButton("Cancel")
        self._fov_cancel_btn.clicked.connect(self._cancel_fov_edit)
        confirm_row.addWidget(self._fov_confirm_btn, 1)
        confirm_row.addWidget(self._fov_cancel_btn, 1)
        active_col.addLayout(confirm_row)
        self._fov_edit_active.setVisible(False)
        fov_layout.addWidget(self._fov_edit_active)

        fov_info_frame = QFrame()
        fov_info_frame.setStyleSheet(f'''
            QFrame {{
                background-color: {COLORS['bg_medium']};
                border: 1px solid {COLORS['border_dark']};
                border-radius: 4px;
            }}
        ''')
        fov_info_layout = QHBoxLayout(fov_info_frame)
        self._fov_info_label = QLabel("No sample points generated")
        self._fov_info_label.setStyleSheet(f'color: {COLORS["text_secondary"]};')
        fov_info_layout.addWidget(self._fov_info_label)
        fov_info_layout.addStretch()
        fov_layout.addWidget(fov_info_frame)

        self._position_stack.addWidget(fov_page)

        return self._position_stack

    # --- right column: steps preview + control / progress / data save -------
    def _create_preview_panel(self) -> QWidget:
        """Right column: the live acquisition-steps preview (expands) plus the
        Control, Progress and Data Save panels below it."""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Build the steps preview group (added below, between Data and Progress).
        preview_group = QGroupBox("Acquisition Steps Preview")
        preview_layout = QVBoxLayout(preview_group)
        self._preview_text = QTextEdit()
        self._preview_text.setReadOnly(True)
        self._preview_text.setStyleSheet(f'''
            QTextEdit {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['border_dark']};
                border-radius: 4px;
                font-family: "Segoe UI";
                font-size: 13px;
                color: {COLORS['text_primary']};
            }}
        ''')
        preview_layout.addWidget(self._preview_text)

        # Progress bar + status label sit at the bottom of the preview panel.
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
        preview_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("Ready")
        self._progress_label.setStyleSheet(f'color: {COLORS["text_secondary"]};')
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self._progress_label)

        # Order: Data Save, Acquisition Steps Preview (+ progress), Control.
        layout.addWidget(self._create_data_save_group())
        layout.addWidget(preview_group, 1)
        layout.addWidget(self._create_control_group())

        return panel

    def _refresh_preview(self):
        """Rebuild the step list so it matches what the acquisition worker will
        actually execute.

        Key points (these caused confusion before):
        - The widefield-camera autofocus runs ONCE, before the point loop, not
          per point.
        - Current FoV measures EVERY generated sample point (the ring/grid/...
          pattern you set), so the preview enumerates them.
        """
        if not hasattr(self, '_preview_text'):
            return
        # Keep the Data Save file-name format in sync with the current pattern.
        self._update_filename_format()
        from raman_control.gui.widgets.sample_grid import well_name

        p = self.get_acquisition_parameters()
        mode = p['acquisition_mode']
        repeats = p['repeat_each_sample']
        reps_per_point = p['reps_per_point']
        # Z-stack: number of NanoDrive Z positions (>=2). The worker acquires
        # reps_per_point spectra at EACH Z step, so the preview enumerates them.
        z_stack = bool(p['z_stack'])
        z_steps = max(2, int(p.get('spec_af_fine') or 10)) if z_stack else 0
        expo = self._binding_value('acq_exposure')
        power = self._binding_value('acq_power')
        expo_s = f"{expo} ms" if expo is not None else "exposure"
        power_s = f"{power} mW" if power is not None else "power"

        # Per-point autofocus (the widefield/Prusa one is handled once, below).
        per_point_af = []
        if p['autofocus_wasatch_prusa']:
            per_point_af.append("Autofocus Wasatch with Prusa")
        if p['autofocus_wasatch_nanodrive'] and not p['z_stack']:
            per_point_af.append("Autofocus Wasatch with NanoDrive")

        warmup_s = int(AcquisitionWorker._LASER_WARMUP_S)

        def emit_point(lines, n, label, first=False):
            """Append the per-point steps; return the updated step counter.

            ``first`` marks the first point of the whole run, where the laser is
            switched on (with a one-time warm-up wait) before any spectrometer
            work — it then stays on until the end of the run."""
            lines.append(f"[{label}]")
            n += 1
            lines.append(f"  {n}. Move to {label}")
            if first:
                n += 1
                lines.append(f"  {n}. Turn laser on (warm up {warmup_s} s, once)")
            for af in per_point_af:
                n += 1
                lines.append(f"  {n}. {af}")
            if z_stack:
                # Z-stack owns the measurement: at each Z step the worker
                # acquires + saves reps_per_point spectra (no separate
                # photobleach / acquire / save steps).
                for zi in range(z_steps):
                    n += 1
                    lines.append(
                        f"  {n}. Z-step {zi + 1}/{z_steps}: NanoDrive Z, acquire "
                        f"+ save {reps_per_point} spectra  ({expo_s}, {power_s})")
                return n
            if p['wait_before_measure']:
                n += 1
                lines.append(f"  {n}. Photobleach {p['wait_time_s']} s with laser on")
            for k in range(reps_per_point):
                n += 1
                lines.append(f"  {n}. Acquire spectrum {k + 1}/{reps_per_point}"
                             f"  ({expo_s}, {power_s})")
            n += 1
            lines.append(f"  {n}. Save data")
            return n

        if mode == 'Current FoV':
            npts = self._current_fov_point_count()
            if npts == 0:
                self._set_preview_text(
                    "Current FoV: no sample points generated yet.\n\n"
                    "Capture an image, run segmentation, and generate sample "
                    "points (your Sample Pattern) in the Live tab.")
                return
            spectra_per_point = reps_per_point * (z_steps if z_stack else 1)
            zinfo = f"  ({z_steps} Z × {reps_per_point}/Z)" if z_stack else ""
            lines = [
                "Mode: Current FoV",
                f"Sample points: {npts}",
                f"Protocol repeats: {repeats}",
                f"Total spectra to save: {repeats * npts * spectra_per_point}"
                + zinfo,
                "",
            ]
            n = 0
            n += 1
            lines.append(f"  {n}. Switch to widefield view")
            if p['autofocus_widefield_prusa']:
                n += 1
                lines.append(f"  {n}. Turn on illumination LED")
                n += 1
                lines.append(f"  {n}. Autofocus widefield camera with Prusa (once)")
                n += 1
                lines.append(f"  {n}. Turn off illumination LED")
            n += 1
            lines.append(f"  {n}. Switch to objective view (Raman is collected "
                         "through it)")
            for rep in range(repeats):
                if repeats > 1:
                    lines.append(f"=== Protocol repeat {rep + 1}/{repeats} "
                                 "(all points) ===")
                for pi in range(npts):
                    n = emit_point(lines, n, f"point {pi + 1}/{npts}",
                                   first=(rep == 0 and pi == 0))
                lines.append("")
            n += 1
            lines.append(f"  {n}. Turn laser off (end of run)")
            n += 1
            lines.append(f"  {n}. Run complete (return to capture center, "
                         "Live controls restored)")
            header = f"Total steps: {n}\n" + "-" * 30 + "\n"
            self._set_preview_text(header + "\n".join(lines))
            return

        # --- Robotic Deposition ---
        # Droplet -> points hierarchy shown with indentation. Each droplet is
        # imaged + segmented at run time, so points/droplet is exact for
        # line/random and an upper bound for grid/ring (only points inside the
        # droplet's shape are kept).
        wells = [well_name(r, c)
                 for (r, c) in self._sample_grid.get_selected_positions()]
        if not wells:
            self._set_preview_text(
                "No droplets selected.\n\nSelect wells in the Sample Droplet "
                "Map (middle), or switch to Current FoV mode.")
            return

        ppd, exact = self._points_per_droplet()
        ptype = self.selected_pattern_type().capitalize()
        upto = "" if exact else "up to "
        shape_note = "" if exact else " (max; actual depends on each droplet's shape)"
        spectra_per_point = reps_per_point * (z_steps if z_stack else 1)
        total = len(wells) * repeats * ppd * spectra_per_point

        def point_substeps(indent, include_laser):
            """The per-generated-point measurement block (mirrors the worker)."""
            sub = [f"{indent}- Move to point"]
            if include_laser:
                sub.append(f"{indent}- Turn laser on (warm up {warmup_s} s) "
                           "(first point of the whole run only)")
            if p['autofocus_wasatch_prusa']:
                sub.append(f"{indent}- Autofocus Wasatch (Prusa)")
            if z_stack:
                # At each Z step the worker acquires + saves reps_per_point
                # spectra (no separate nano-AF / photobleach / acquire / save).
                for zi in range(z_steps):
                    sub.append(f"{indent}- Z-step {zi + 1}/{z_steps}: NanoDrive "
                               f"Z, acquire + save {reps_per_point} spectra")
                return sub
            if p['autofocus_wasatch_nanodrive']:
                sub.append(f"{indent}- Autofocus Wasatch (NanoDrive)")
            if p['wait_before_measure']:
                sub.append(f"{indent}- Photobleach {p['wait_time_s']} s with laser on")
            sub.append(f"{indent}- Acquire spectrum ×{reps_per_point} "
                       f"({expo_s}, {power_s})")
            sub.append(f"{indent}- Save data")
            return sub

        lines = [
            "Mode: Robotic Deposition",
            f"Selected droplets: {len(wells)}      Protocol repeats: {repeats}",
            f"Pattern: {ptype}, {upto}{ppd} points/droplet{shape_note}",
            f"Total spectra: {upto}{total}  "
            f"({len(wells)} droplets × {repeats} reps × {ppd} pts × "
            + (f"{z_steps} Z × {reps_per_point}/Z)" if z_stack
               else f"{reps_per_point}/pt)"),
            "Each droplet is imaged + segmented at run time, then the pattern "
            "is generated on it.",
            "-" * 30,
        ]
        for rep in range(repeats):
            if repeats > 1:
                lines.append(f"=== Protocol repeat {rep + 1}/{repeats} ===")
            for di, name in enumerate(wells):
                lines.append(f"Droplet {name}:")
                dsteps = ["Switch to widefield view",
                          f"Move to droplet {name}",
                          "Turn on illumination LED"]
                if p['autofocus_widefield_prusa']:
                    dsteps.append("Autofocus widefield camera (Prusa)")
                # Order mirrors the worker: capture (LED on) -> LED off ->
                # segment -> generate -> switch to objective.
                dsteps += ["Capture image", "Turn off illumination LED",
                           "Segment droplet",
                           f"Generate {ptype} points ({upto}{ppd})",
                           "Switch to objective view"]
                for i, txt in enumerate(dsteps, 1):
                    lines.append(f"    {i}. {txt}")
                lines.append(f"        For each generated point ({upto}{ppd}):")
                lines.extend(point_substeps("            ",
                                            rep == 0 and di == 0))
                lines.append("")
        lines += ["End of run:",
                  "    Turn laser off",
                  "    Run complete (Live controls restored)"]
        self._set_preview_text("\n".join(lines))

    # --- live step highlighting ---------------------------------------------
    def _set_preview_text(self, text: str):
        """Store the plain step text and render it (clearing any highlight)."""
        self._preview_plain = text
        self._highlight_idx = -1
        self._highlight_block = None  # forget the sticky per-droplet/point scope
        self._render_preview()

    def _render_preview(self):
        """Render the step preview, drawing a cyan ▶ on the current step (if
        any). Plain text when nothing is highlighted."""
        import html as _html
        text = getattr(self, '_preview_plain', '')
        idx = getattr(self, '_highlight_idx', -1)
        lines = text.split("\n")
        if not (0 <= idx < len(lines)):
            self._preview_text.setPlainText(text)
            return
        cyan = COLORS['accent_primary']
        out = []
        for i, ln in enumerate(lines):
            esc = _html.escape(ln)
            if i == idx:
                out.append(f'<span style="color:{cyan}; font-weight:bold;">'
                           f'▶ {esc}</span>')
            else:
                out.append(esc)
        self._preview_text.setHtml(
            "<pre style='margin:0; white-space:pre-wrap; "
            "font-family:\"Segoe UI\"; font-size:13px;'>"
            + "\n".join(out) + "</pre>")

    # Map a worker status/progress message (substring, lowercase) to a preview
    # line (substring, lowercase). First match wins, so order by specificity.
    _STEP_RULES = [
        ('illumination led off', 'turn off illumination led'),
        ('led off', 'turn off illumination led'),
        ('illumination led on', 'turn on illumination led'),
        ('led on', 'turn on illumination led'),
        ('widefield autofocus', 'autofocus widefield camera'),
        ('autofocusing widefield', 'autofocus widefield camera'),
        ('capture image', 'capture image'),
        ('segment', 'segment droplet'),
        ('generate', 'generate '),
        ('wasatch autofocus (nanodrive)', 'autofocus wasatch (nanodrive)'),
        ('wasatch autofocus (prusa)', 'autofocus wasatch (prusa)'),
        ('warming up', 'turn laser on'),
        ('laser on', 'turn laser on'),
        ('laser off', 'turn laser off'),
        ('moving', 'move to'),
        ('acquiring', 'acquire spectrum'),
        # Before 'saved': the completion message ends with "spectra saved".
        ('acquisition complete', 'run complete'),
        ('saved', 'save data'),
        ('switching to objective', 'switch to objective'),
        ('switching to widefield', 'switch to widefield'),
    ]

    def _highlight_step(self, message: str):
        """Highlight the preview line for the worker's current status/progress
        message. Scopes to the point/droplet block the message names so the
        right per-point line lights up; leaves the highlight as-is if there's no
        confident match."""
        if not message or not getattr(self, '_cfov_running', False):
            return
        text = getattr(self, '_preview_plain', '')
        if not text:
            return
        import re
        msg = message.lower()
        # Z-stack messages carry "z-step N/M"; highlight that exact Z-step line
        # (the generic rules below can't tell the Z steps apart).
        zstep = re.search(r'z-step (\d+)/(\d+)', msg)
        if zstep:
            target = f"z-step {zstep.group(1)}/{zstep.group(2)}"
        else:
            target = next((sub for key, sub in self._STEP_RULES if key in msg), None)
            if target is None:
                return
        lines = text.split("\n")
        # Scope to the point/droplet block named in the message: the point
        # ordinal ("point 3" in "point 3/10", matching a "[point 3/10]" header)
        # and/or a droplet well name ("B5", matching "Droplet B5:"). The bare rep
        # counter ("k/reps") is deliberately NOT used for scoping, or it would
        # false-match a "[point k/reps]" header and pin the cursor to the wrong
        # (usually first) point.
        lo, hi = 0, len(lines)
        msg_pt = re.search(r'point (\d+)', msg)
        wells = [w.lower() for w in re.findall(r'\b[a-p]\d{1,2}\b', msg)]
        headers = [i for i, ln in enumerate(lines)
                   if ln.strip().startswith('[') or ln.strip().startswith('Droplet ')]

        def _scopes(header_line: str) -> bool:
            hl = header_line.lower()
            if msg_pt and re.search(r'point ' + msg_pt.group(1) + r'\b', hl):
                return True
            return any(w in hl for w in wells)

        # When the message names a point/droplet, remember its block. Steps that
        # DON'T name one (shared LED on/off, capture, segment, ...) then stay in
        # the block currently being processed instead of jumping to the first.
        matched = None
        for k, hidx in enumerate(headers):
            if _scopes(lines[hidx]):
                matched = (hidx, headers[k + 1] if k + 1 < len(headers) else len(lines))
                break
        if matched is not None:
            self._highlight_block = matched
        lo, hi = getattr(self, '_highlight_block', None) or (0, len(lines))
        idx = next((i for i in range(lo, hi) if target in lines[i].lower()), -1)
        if idx < 0:
            # Run-level steps (switch views at start, laser off / run complete at
            # end) live outside every block: fall back to a whole-text search.
            idx = next((i for i, ln in enumerate(lines)
                        if target in ln.lower()), -1)
        if idx >= 0 and idx != self._highlight_idx:
            self._highlight_idx = idx
            self._render_preview()

    def _create_control_panel(self) -> QWidget:
        """Create the control and parameters panel (scrollable)."""
        content = QFrame()
        layout = QVBoxLayout(content)
        # Padding so the group boxes don't butt up against the column edges
        # (and clear of the scrollbar on the right).
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        # Acquisition mode (Current FoV vs Robotic Deposition)
        layout.addWidget(self._create_mode_group())

        # Acquisition options (autofocus toggles, Z-stack, wait) — kept right
        # under the mode so the toggles that drive the panels below are visible
        # first.
        layout.addWidget(self._create_options_group())

        # --- Mirrored Live-tab parameters (editable; pushed back on Start) ---
        # The Z ranges live with the autofocus that uses them: widefield range
        # in Camera Autofocus; objective + NanoDrive ranges in Spectrometer
        # Autofocus.
        layout.addWidget(self._create_camera_af_group())
        layout.addWidget(self._create_spec_af_group())
        layout.addWidget(self._create_pattern_group())

        # Spectrum Acquisition (exposure / power / repetitions per point) sits
        # below the sample pattern.
        layout.addWidget(self._create_spectrum_acq_group())

        # Acquisition repeats (whole-run repeat) — at the end.
        layout.addWidget(self._create_repeats_group())

        # Control / Progress / Data Save live in the right column now.
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        # Size the left column to its content (compact) instead of letting it
        # take a third of the window — the entry boxes are narrow now, so the
        # extra width was wasted. Freed space goes to the middle/right columns.
        scroll.setFixedWidth(self._LEFT_COLUMN_WIDTH)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        return scroll

    # --- control / progress / data-save groups (right column) ---------------
    def _create_control_group(self) -> QGroupBox:
        """Start / Pause / Stop buttons."""
        from PyQt6.QtGui import QFont

        group = QGroupBox("Control")
        control_layout = QHBoxLayout(group)
        control_layout.setSpacing(10)

        btn_font = QFont()
        btn_font.setPointSize(13)
        btn_font.setBold(True)

        def _big(btn):
            btn.setMinimumHeight(56)
            btn.setFont(btn_font)
            return btn

        # One primary button cycles Start -> Pause -> Resume -> Pause; Stop ends.
        self._start_btn = _big(QPushButton("Start"))
        self._start_btn.setProperty('primary', True)
        self._start_btn.clicked.connect(self._on_primary_clicked)
        control_layout.addWidget(self._start_btn)

        self._stop_btn = _big(QPushButton("Stop"))
        self._stop_btn.setProperty('danger', True)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        control_layout.addWidget(self._stop_btn)
        return group

    def _create_data_save_group(self) -> QGroupBox:
        """Save directory + filename template."""
        from PyQt6.QtWidgets import QLineEdit

        group = QGroupBox("Data Save Settings")
        data_layout = QVBoxLayout(group)

        # Save location
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

        # File-name format. Spectra are named automatically (see
        # AcquisitionWorker._build_filename); this read-only field shows the
        # user how each file will be named.
        #
        # Optional custom prefix, prepended in front of the default name.
        prefix_layout = QHBoxLayout()
        prefix_layout.addWidget(QLabel("Custom prefix:"))
        self._prefix_edit = QLineEdit()
        self._prefix_edit.setPlaceholderText(
            "(optional) prepended in front of the file name")
        self._prefix_edit.textChanged.connect(self._update_filename_format)
        prefix_layout.addWidget(self._prefix_edit, 1)
        data_layout.addLayout(prefix_layout)

        name_layout = QVBoxLayout()
        name_layout.addWidget(QLabel("Will be saved as:"))
        # A plain (wrapping) text label, not an entry box — not clickable.
        self._filename_edit = QLabel()
        self._filename_edit.setWordWrap(True)
        self._filename_edit.setStyleSheet(f'color: {COLORS["text_secondary"]};')
        self._filename_edit.setToolTip(
            "Spectra are named automatically; this shows the format for the "
            "current mode + sampling pattern. Each field is name_value.\n"
            "• a Custom prefix above is prepended to the name\n"
            "• Current FoV starts at pattern_ (no sample/droplet/position)\n"
            "• Robotic Deposition adds the sample name + droplet_ (from the "
            "config) and position_ (well)\n"
            "• the point position adapts to the pattern: row_/col_ (grid), "
            "ring_/ray_ (ring), point_ (line/random)\n"
            "• a z-stack adds zstep_<n>")
        name_layout.addWidget(self._filename_edit)
        data_layout.addLayout(name_layout)
        self._update_filename_format()
        return group

    def _update_filename_format(self):
        """Refresh the displayed file-name format for the current pattern, mode
        and custom prefix."""
        if hasattr(self, '_filename_edit'):
            prefix = (self._prefix_edit.text().strip()
                      if hasattr(self, '_prefix_edit') else '')
            current_fov = (self._mode_current_fov.isChecked()
                           if hasattr(self, '_mode_current_fov') else False)
            text = AcquisitionWorker.filename_format(
                self.selected_pattern_type(), prefix, current_fov=current_fov,
                sample_prefix=self._config_sample_prefix())
            # The name has underscores but no spaces, so word-wrap alone won't
            # break it — add zero-width break points (U+200B) after each '_' so
            # the label can wrap onto multiple rows when it doesn't fit.
            self._filename_edit.setText(text.replace('_', '_​'))

    def _on_ranges_registered(self):
        """The user registered the autofocus ranges in the Live tab: reveal the
        now-valid range values and re-evaluate the Start button."""
        self.pull_from_live()
        self._update_start_enabled()

    def _ready_to_start(self) -> bool:
        """Start is allowed only when (1) the autofocus Z ranges are registered,
        (2) a data save directory is set, and (3) — in Robotic Deposition mode —
        at least one droplet is selected."""
        sc = self._stage_control
        if sc is None or not sc.is_registered():
            return False
        if not self._save_path_edit.text().strip():
            return False
        if self._mode_current_fov.isChecked():
            # Current FoV needs sample points generated on the captured image.
            if self._current_fov_point_count() == 0:
                return False
        elif not self._sample_grid.get_selected_positions():
            # Robotic Deposition needs at least one droplet selected.
            return False
        return True

    def _update_start_enabled(self):
        """Enable the Start button only when prerequisites are met (skipped
        while a run is active — the button is then Pause/Resume)."""
        if getattr(self, '_cfov_running', False):
            return
        if not (hasattr(self, '_start_btn') and hasattr(self, '_save_path_edit')
                and hasattr(self, '_mode_current_fov')
                and hasattr(self, '_sample_grid')):
            return  # still constructing
        self._start_btn.setEnabled(self._ready_to_start())

    def _config_sample_prefix(self) -> str:
        """The common '<prefix>' shared by the loaded config's sample names
        (e.g. 'patient' from patient_304 / patient_506), so the format can show
        '<prefix>_<name>' explicitly. '' if no config / no shared prefix."""
        import os
        grid = getattr(self, '_sample_grid', None)
        if grid is None:
            return ''
        names = [str(e.get('sample_name', '')).strip()
                 for e in grid.droplet_map().values()]
        names = [n for n in names if n]
        if not names:
            return ''
        common = os.path.commonprefix(names)
        # Keep only whole '<prefix>_' segment(s); drop the variable tail.
        return common.rsplit('_', 1)[0] if '_' in common else ''

    # --- mirrored Live-tab parameter groups ---------------------------------
    def _create_spectrum_acq_group(self) -> QGroupBox:
        """Spectrum acquisition exposure / laser power (NOT the autofocus ones)."""
        group = QGroupBox("Spectrum Acquisition")
        grid = QGridLayout(group)
        grid.setSpacing(8)
        r = 0
        sv = self._spectrum_viewer
        r = self._mirror(grid, r, 'acq_exposure', "Exposure Time:",
                         lambda: self._live_spin(sv, '_integration_spin'))
        r = self._mirror(grid, r, 'acq_power', "Laser Power:",
                         lambda: self._live_spin(sv, '_power_spin'))
        # Repetitions per point.
        r = self._add_parameters_rows(grid, r)
        return group

    def _create_camera_af_group(self) -> QGroupBox:
        """Camera (widefield) autofocus: the widefield Prusa Z range it scans,
        then the step count for that scan."""
        group = QGroupBox("Widefield Camera Autofocus")
        grid = QGridLayout(group)
        grid.setSpacing(8)
        r = 0
        lv = self._live_view
        sc = self._stage_control
        for key, label, attr in [
            ('wf_z_upper', "Z Upper:", '_wf_z_upper'),
            ('wf_z_lower', "Z Lower:", '_wf_z_lower'),
        ]:
            r = self._mirror(grid, r, key, label,
                             lambda a=attr: self._live_spin(sc, a),
                             read_only=True)
        r = self._mirror(grid, r, 'cam_af_steps', "Prusa Steps:",
                         lambda: self._live_spin(lv, '_af_steps_spin'))
        return group

    def _create_spec_af_group(self) -> QGroupBox:
        """Spectrometer autofocus: each scan's range followed by its step count
        (Prusa coarse over the objective Z range; NanoDrive fine over the
        NanoDrive range), then the shared exposure / power / reps."""
        group = QGroupBox("Spectrometer Autofocus")
        grid = QGridLayout(group)
        grid.setSpacing(8)
        r = 0
        sv = self._spectrum_viewer
        sc = self._stage_control
        # Prusa scan: Z range, then its step count.
        r = self._mirror(grid, r, 'obj_z_upper', "Prusa Z Upper:",
                         lambda: self._live_spin(sc, '_obj_z_upper'),
                         read_only=True)
        r = self._mirror(grid, r, 'obj_z_lower', "Prusa Z Lower:",
                         lambda: self._live_spin(sc, '_obj_z_lower'),
                         read_only=True)
        r = self._mirror(grid, r, 'spec_af_coarse', "Prusa Steps:",
                         lambda: self._live_spin(sv, '_af_prusa_steps'))
        # NanoDrive scan: Z range, then its step count.
        r = self._mirror(grid, r, 'nano_upper', "NanoDrive Z Upper:",
                         lambda: self._live_spin(sc, '_nano_upper'),
                         read_only=True)
        r = self._mirror(grid, r, 'nano_lower', "NanoDrive Z Lower:",
                         lambda: self._live_spin(sc, '_nano_lower'),
                         read_only=True)
        r = self._mirror(grid, r, 'spec_af_fine', "NanoDrive Steps:",
                         lambda: self._live_spin(sv, '_af_nano_steps'))
        # Shared spectrometer settings for the autofocus scans.
        for key, label, attr in [
            ('spec_af_exposure', "AF Exposure:", '_af_expo'),
            ('spec_af_power', "AF Laser Power:", '_af_power'),
            ('spec_af_reps', "AF Reps:", '_af_reps'),
        ]:
            r = self._mirror(grid, r, key, label,
                             lambda a=attr: self._live_spin(sv, a))
        return group

    # Pattern types in radio/stack order; kept as a single source of truth so
    # the radios, the stacked pages and get_acquisition_parameters() agree.
    _PATTERN_TYPES = ['grid', 'ring', 'line', 'random']
    _PATTERN_LABELS = ['Grid', 'Ring', 'Line', 'Random']
    _PATTERN_DEFAULT = 1  # Ring, matching the Live tab default

    # Per-type parameter rows: (binding key, label, Live-tab spin attribute).
    _PATTERN_PAGES = [
        [('grid_rows', "Rows:", '_grid_rows'),
         ('grid_cols', "Cols:", '_grid_cols'),
         ('grid_buffer', "Buffer:", '_grid_buffer')],
        [('ring_rays', "Rays:", '_ring_rays'),
         ('ring_rings', "Rings:", '_ring_rings'),
         ('ring_interval', "Interval:", '_ring_interval'),
         ('ring_buffer', "Buffer:", '_ring_buffer')],
        [('line_points', "Points:", '_line_points'),
         ('line_buffer', "Buffer:", '_line_buffer')],
        [('rand_points', "Points:", '_rand_points')],
    ]

    def _create_pattern_group(self) -> QGroupBox:
        """Sample-point pattern parameters (grid / ring / line / random).

        A radio button selects the pattern type; only that type's parameters
        are shown (in a stacked widget). The selection and values mirror the
        Live tab and are pushed back on Start.
        """
        # Reuse the Live tab's auto-sizing stack so the group shrinks/grows to
        # the active page instead of padding to the tallest one. Imported
        # locally to avoid any import-order coupling between the widgets.
        from raman_control.gui.widgets.live_view import _AutoSizeStack

        group = QGroupBox("Sample Pattern")
        outer = QVBoxLayout(group)
        outer.setSpacing(8)
        lv = self._live_view

        # All the pattern-design controls live in one container so they can be
        # hidden as a unit when the mode is Current FoV (where the points are
        # already fixed on the captured image and only their count matters).
        self._pattern_params_widget = QWidget()
        params_outer = QVBoxLayout(self._pattern_params_widget)
        params_outer.setContentsMargins(0, 0, 0, 0)
        params_outer.setSpacing(8)

        # Pattern type radios
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self._pattern_type_group = QButtonGroup(self)
        self._pattern_type_radios = []
        for idx, name in enumerate(self._PATTERN_LABELS):
            radio = QRadioButton(name)
            self._pattern_type_group.addButton(radio, idx)
            self._pattern_type_radios.append(radio)
            type_row.addWidget(radio)
        type_row.addStretch()
        params_outer.addLayout(type_row)

        # One parameter page per pattern type, each mirroring its Live spins.
        self._pattern_stack = _AutoSizeStack()
        for page_rows in self._PATTERN_PAGES:
            page = QWidget()
            pg = QGridLayout(page)
            pg.setContentsMargins(0, 0, 0, 0)
            pg.setSpacing(8)
            r = 0
            for key, label, attr in page_rows:
                r = self._mirror(pg, r, key, label,
                                 lambda a=attr: self._live_spin(lv, a))
            self._pattern_stack.addWidget(page)
        params_outer.addWidget(self._pattern_stack)

        # Default selection (kept consistent with the stacked page).
        self._pattern_type_radios[self._PATTERN_DEFAULT].setChecked(True)
        self._pattern_stack.setCurrentIndex(self._PATTERN_DEFAULT)
        self._pattern_type_group.idClicked.connect(self._on_pattern_type_changed)
        outer.addWidget(self._pattern_params_widget)

        # Current-FoV summary: shown in place of the design controls — just the
        # number of points generated on the captured image (manual points
        # included, since they share the Live tab's point list).
        self._pattern_count_label = QLabel("No sample points generated")
        self._pattern_count_label.setWordWrap(True)
        self._pattern_count_label.setStyleSheet(f'color: {COLORS["text_primary"]};')
        self._pattern_count_label.setVisible(False)
        outer.addWidget(self._pattern_count_label)
        return group

    def _on_pattern_type_changed(self, idx: int):
        """Show the selected type's parameters and sync the Live-tab radio."""
        self._pattern_stack.setCurrentIndex(idx)
        lv = self._live_view
        radios = getattr(lv, '_pattern_radios', None) if lv is not None else None
        if radios and 0 <= idx < len(radios):
            radios[idx].setChecked(True)
            if hasattr(lv, '_on_pattern_changed'):
                lv._on_pattern_changed(idx)
        self._refresh_preview()

    def _pull_pattern_type(self):
        """Copy the Live-tab pattern selection into the local radios."""
        if not hasattr(self, '_pattern_type_group'):
            return
        lv = self._live_view
        grp = getattr(lv, '_pattern_btn_group', None) if lv is not None else None
        idx = grp.checkedId() if grp is not None else -1
        if not (0 <= idx < len(self._pattern_type_radios)):
            return
        self._pattern_type_group.blockSignals(True)
        self._pattern_type_radios[idx].setChecked(True)
        self._pattern_type_group.blockSignals(False)
        self._pattern_stack.setCurrentIndex(idx)

    def _push_pattern_type(self):
        """Write the local pattern selection back to the Live tab."""
        if not hasattr(self, '_pattern_type_group'):
            return
        idx = self._pattern_type_group.checkedId()
        lv = self._live_view
        radios = getattr(lv, '_pattern_radios', None) if lv is not None else None
        if radios and 0 <= idx < len(radios):
            radios[idx].setChecked(True)
            if hasattr(lv, '_on_pattern_changed'):
                lv._on_pattern_changed(idx)

    def selected_pattern_type(self) -> str:
        """Currently selected pattern type ('grid' / 'ring' / 'line' / 'random')."""
        idx = (self._pattern_type_group.checkedId()
               if hasattr(self, '_pattern_type_group') else self._PATTERN_DEFAULT)
        if not (0 <= idx < len(self._PATTERN_TYPES)):
            idx = self._PATTERN_DEFAULT
        return self._PATTERN_TYPES[idx]

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

        # Short labels keep the left column narrow; full descriptions are in
        # the tooltips.
        self._af_widefield_prusa = QCheckBox("Widefield AF (Prusa)")
        self._af_widefield_prusa.setToolTip(
            "Autofocus the widefield camera with the Prusa stage (once)")
        self._af_widefield_prusa.setChecked(True)
        self._af_wasatch_prusa = QCheckBox("Spectrometer Autofocus per point (Prusa)")
        self._af_wasatch_prusa.setToolTip(
            "Autofocus the Wasatch signal with the Prusa stage at each point")
        self._af_wasatch_prusa.setChecked(True)
        self._af_wasatch_nanodrive = QCheckBox("Spectrometer AF (NanoDrive)")
        self._af_wasatch_nanodrive.setToolTip(
            "Autofocus the Wasatch signal with the NanoDrive at each point")
        self._af_wasatch_nanodrive.setChecked(True)
        lay.addWidget(self._af_widefield_prusa)
        lay.addWidget(self._af_wasatch_prusa)
        lay.addWidget(self._af_wasatch_nanodrive)
        for cb in (self._af_widefield_prusa, self._af_wasatch_prusa):
            cb.toggled.connect(self._refresh_preview)
        # NanoDrive AF is mutually exclusive with Z-stack (both drive the
        # NanoDrive Z), handled in its own slot.
        self._af_wasatch_nanodrive.toggled.connect(self._on_wasatch_nano_toggled)

        # Z-stack and NanoDrive AF are mutually exclusive (both drive the
        # NanoDrive Z). Checking one unchecks the other — switchable from either.
        self._z_stack_check = QCheckBox("Z-stack at each point (NanoDrive)")
        self._z_stack_check.setToolTip(
            "Acquire a NanoDrive Z-stack at each point (uses the NanoDrive "
            "range / step). Mutually exclusive with Wasatch AF (NanoDrive).")
        self._z_stack_check.toggled.connect(self._on_z_stack_toggled)
        lay.addWidget(self._z_stack_check)

        # Optional photobleaching (laser on, no spectra saved) before measuring
        # at each point.
        self._wait_check = QCheckBox("Photobleaching with laser on before measuring")
        self._wait_check.toggled.connect(self._on_wait_toggled)
        lay.addWidget(self._wait_check)

        wait_row = QHBoxLayout()
        self._wait_time_label = QLabel("Bleaching time per point:")
        wait_row.addWidget(self._wait_time_label)
        self._wait_time_spin = QSpinBox()
        self._wait_time_spin.setRange(0, 3600)
        self._wait_time_spin.setValue(30)
        self._wait_time_spin.valueChanged.connect(self._refresh_preview)
        wait_row.addLayout(self._entry_with_unit(self._wait_time_spin, "s"))
        lay.addLayout(wait_row)

        # Reflect initial states
        self._on_z_stack_toggled()
        self._on_wait_toggled()
        return group

    def _on_acquisition_mode_changed(self):
        """Switch the middle column and the Sample Pattern panel to match the
        selected mode.

        Robotic Deposition -> droplet map + pattern-design controls.
        Current FoV        -> captured/segmented image with points + a point
                              count (the points are already fixed). The
                              stage-moved safety warning is deferred to Start
                              (see _confirm_stage_move_for_start).
        """
        current_fov = self._mode_current_fov.isChecked()
        robotic = self._mode_robotic.isChecked()

        # Middle column page.
        self._position_stack.setCurrentIndex(1 if current_fov else 0)
        self._sample_grid.set_grid_enabled(robotic)

        # Sample Pattern panel: design controls vs. point count.
        self._pattern_params_widget.setVisible(not current_fov)
        self._pattern_count_label.setVisible(current_fov)

        if current_fov:
            self._refresh_fov_view()

        self._refresh_preview()
        self._update_start_enabled()

    # --- Current FoV handling ------------------------------------------------
    def _lv_attr(self, name, default=None):
        """Safe accessor for a Live-view attribute (None-safe)."""
        lv = self._live_view
        return getattr(lv, name, default) if lv is not None else default

    def _current_fov_point_count(self) -> int:
        """Number of sampling points generated on the captured image
        (generated + manually added — they share one Live-tab list)."""
        positions = self._lv_attr('_generated_positions', None)
        return len(positions) if positions else 0

    def _points_per_droplet(self) -> Tuple[int, bool]:
        """(count, exact) sample points the current pattern yields per droplet,
        from the pattern params alone (no segmentation needed). Exact for
        line/random; an upper bound for grid (rows×cols) and ring (rays×rings),
        since only candidate points inside each droplet's shape are kept."""
        ps = (self._live_view.snapshot_pattern_settings()
              if self._live_view is not None else None)
        if not ps:
            return 1, True
        t = ps.get('type', 'ring')
        prm = ps.get(t, {})
        if t == 'grid':
            return max(1, int(prm.get('rows', 1)) * int(prm.get('cols', 1))), False
        if t == 'ring':
            return max(1, int(prm.get('rays', 1)) * int(prm.get('rings', 1))), False
        return max(1, int(prm.get('points', 1))), True  # line / random (exact)

    def _current_fov_ready(self) -> bool:
        """Current FoV needs a captured image, a segmentation mask, and at
        least one generated sample point before it can be used."""
        return (self._lv_attr('_captured_image') is not None
                and self._lv_attr('_current_mask') is not None
                and self._current_fov_point_count() > 0)

    def _update_current_fov_availability(self):
        """Enable Current FoV only once points exist; if it becomes
        unavailable while selected, fall back to Robotic Deposition."""
        ready = self._current_fov_ready()
        self._mode_current_fov.setEnabled(ready)
        if ready:
            self._mode_current_fov.setToolTip(
                "Acquire at the sample points generated on the captured image.")
        else:
            self._mode_current_fov.setToolTip(
                "Capture an image, run segmentation, and generate sample points "
                "in the Live tab to enable Current FoV.")
        if not ready and self._mode_current_fov.isChecked():
            # Triggers _on_acquisition_mode_changed via the toggled signal.
            self._mode_robotic.setChecked(True)

    def _refresh_fov_view(self):
        """Update the Current-FoV image and point count from the Live tab."""
        n = self._current_fov_point_count()

        # Sample Pattern panel count.
        if hasattr(self, '_pattern_count_label'):
            if n > 0:
                self._pattern_count_label.setText(
                    f"Total sample points: {n}\n"
                    "(in the current field of view, including manually "
                    "added points)")
            else:
                self._pattern_count_label.setText("No sample points generated")

        # Middle-column image: reuse the composite pixmap the Live tab already
        # renders (captured image + mask overlay + point markers).
        if hasattr(self, '_fov_preview'):
            pixmap = None
            cap = self._lv_attr('_capture_preview')
            if cap is not None and hasattr(cap, 'base_pixmap'):
                pixmap = cap.base_pixmap()
            self._fov_preview.set_base_pixmap(pixmap)
            # Match the Live tab's captured-image scale bar (same widefield FOV).
            if self._live_view is not None:
                self._fov_preview.set_scale_bar(
                    self._live_view.widefield_um_per_px())
            if n > 0:
                self._fov_info_label.setText(
                    f"{n} sample point{'s' if n != 1 else ''} on captured image "
                    "(including manually added)")
            else:
                self._fov_info_label.setText("No sample points generated")

        self._update_fov_edit_enabled()

    def _on_live_positions_changed(self, positions):
        """Live tab generated or edited points: re-gate Current FoV and refresh
        the image / count."""
        self._update_current_fov_availability()
        self._refresh_fov_view()
        self._refresh_preview()
        self._update_start_enabled()

    # --- Current-FoV point editing (mirrors the Live tab's Add/Remove Points) --

    def _update_fov_edit_enabled(self):
        """Gate the normal Add/Remove buttons: editing needs a captured image
        AND a segmentation mask (so points exist / can be added); Remove also
        needs at least one point. Skipped while a Confirm/Cancel edit is active
        (its buttons are hidden)."""
        if not hasattr(self, '_fov_add_btn'):
            return
        lv = self._live_view
        can_edit = bool(lv is not None and lv.can_add_points()
                        and getattr(lv, '_current_mask', None) is not None)
        has_points = self._current_fov_point_count() > 0
        self._fov_add_btn.setEnabled(can_edit)
        self._fov_remove_btn.setEnabled(can_edit and has_points)

    def _begin_fov_edit(self, mode: str):
        """Start an add/delete edit session in the Live tab. The resulting
        point_edit_mode_changed signal swaps this tab's toolbar to Confirm/
        Cancel (see _on_fov_edit_mode_changed)."""
        if self._live_view is not None:
            self._live_view.begin_point_edit(mode)

    def _confirm_fov_edit(self):
        """Keep the points edited this session (delegates to the Live tab)."""
        if self._live_view is not None:
            self._live_view.confirm_point_edit()

    def _cancel_fov_edit(self):
        """Revert the points edited this session (delegates to the Live tab)."""
        if self._live_view is not None:
            self._live_view.cancel_point_edit()

    def _on_fov_edit_mode_changed(self, mode: str):
        """Swap the FoV toolbar between the normal Add/Remove row and the
        Confirm/Cancel row, following the Live tab's edit state ('' = none)."""
        if not hasattr(self, '_fov_edit_active'):
            return
        editing = bool(mode)
        self._fov_edit_active.setVisible(editing)
        self._fov_edit_normal.setVisible(not editing)
        if editing:
            self._fov_edit_hint.setText(
                "Click the image to add a point, then Confirm." if mode == 'add'
                else "Click a point to remove it, then Confirm.")

    def _on_fov_preview_clicked(self, base_x: int, base_y: int):
        """A click on the FoV preview edits the shared point list while an edit
        session is active; otherwise it's a no-op (the preview is view/zoom only
        here — stage moves stay on the Live tab)."""
        lv = self._live_view
        mode = getattr(lv, 'point_edit_mode', None) if lv is not None else None
        if mode is None:
            return
        # The Live tab owns the points; it re-emits positions_generated, which
        # refreshes this preview via _on_live_positions_changed.
        lv.edit_point_at_base(base_x, base_y, mode)

    def _confirm_stage_move_for_start(self) -> bool:
        """Safety gate run when Start is pressed.

        If Current FoV is selected and the stage has drifted away from where the
        image was captured, the acquisition's move back to the captured points
        could be long-distance and risk crashing the objective. Warn the user
        and let them cancel to go back and adjust (e.g. raise Z).

        Returns True to proceed with the acquisition, False to cancel. Any case
        where the check can't be evaluated (no Current FoV, no capture position,
        stage not moved) returns True so normal runs are never blocked.
        """
        if not self._mode_current_fov.isChecked():
            return True
        cap = self._lv_attr('_capture_stage_position')
        provider = self._lv_attr('_position_provider')
        if cap is None or provider is None:
            return True
        try:
            cur = provider()  # cached physical position (mm); no hardware I/O
        except Exception:
            return True
        if not cur or len(cur) < 2:
            return True
        dx, dy = cur[0] - cap[0], cur[1] - cap[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= self._FOV_MOVE_WARN_MM:
            return True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Stage moved from capture position")
        box.setText(
            f"The stage is now {dist:.2f} mm (XY) away from where the image "
            "was captured.")
        box.setInformativeText(
            f"Capture position:  X={cap[0]:.3f}, Y={cap[1]:.3f} mm\n"
            f"Current position:  X={cur[0]:.3f}, Y={cur[1]:.3f} mm\n\n"
            "Starting Current FoV acquisition will move the stage back to the "
            "captured sample points, which may involve long-distance travel. "
            "Make sure the objective is raised to a SAFE Z height before "
            "continuing, to avoid crashing into the sample or stage.\n\n"
            "Proceed with acquisition?")
        proceed_btn = box.addButton("Proceed Anyway",
                                    QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(box.buttons()[-1])  # default to Cancel (safe)
        box.exec()
        return box.clickedButton() is proceed_btn

    def _on_z_stack_toggled(self):
        """Z-stack and Wasatch NanoDrive AF both drive the NanoDrive Z, so they
        are mutually exclusive. Enabling Z-stack unchecks NanoDrive AF (which
        stays clickable — clicking it switches back)."""
        if self._z_stack_check.isChecked() and self._af_wasatch_nanodrive.isChecked():
            self._af_wasatch_nanodrive.blockSignals(True)
            self._af_wasatch_nanodrive.setChecked(False)
            self._af_wasatch_nanodrive.blockSignals(False)
        self._refresh_preview()

    def _on_wasatch_nano_toggled(self):
        """Selecting Wasatch NanoDrive AF turns off the (mutually exclusive)
        Z-stack, so the user can switch back to NanoDrive AF by clicking it."""
        if self._af_wasatch_nanodrive.isChecked() and self._z_stack_check.isChecked():
            self._z_stack_check.blockSignals(True)
            self._z_stack_check.setChecked(False)
            self._z_stack_check.blockSignals(False)
        self._refresh_preview()

    def _on_wait_toggled(self):
        """The wait-time entry only matters when the wait step is enabled."""
        on = self._wait_check.isChecked()
        self._wait_time_label.setEnabled(on)
        self._wait_time_spin.setEnabled(on)
        self._refresh_preview()

    def _add_parameters_rows(self, grid, row: int) -> int:
        """Add the per-point spectrum parameter (repetitions per point) to an
        existing grid starting at ``row``; return the next free row.

        (Repeat-protocol lives in its own 'Acquisition Repeats' panel.)"""
        # Repetitions per point per sample drop
        grid.addWidget(QLabel("Repetitions per point:"), row, 0)
        self._reps_per_point_spin = QSpinBox()
        self._reps_per_point_spin.setRange(1, 1000)
        self._reps_per_point_spin.setValue(3)
        self._reps_per_point_spin.valueChanged.connect(self._refresh_preview)
        grid.addLayout(self._entry_with_unit(self._reps_per_point_spin, ""), row, 1)
        row += 1

        return row

    def _create_repeats_group(self) -> QGroupBox:
        """Acquisition Repeats: how many times to repeat the WHOLE protocol
        end-to-end over all points (not point-by-point)."""
        group = QGroupBox("Acquisition Repeats")
        grid = QGridLayout(group)
        grid.setSpacing(8)
        grid.addWidget(QLabel("Repeat protocol:"), 0, 0)
        self._repeat_sample_spin = QSpinBox()
        self._repeat_sample_spin.setRange(1, 1000)
        self._repeat_sample_spin.setValue(1)
        self._repeat_sample_spin.setToolTip(
            "Number of times to repeat the entire acquisition end-to-end over "
            "all points: point 1, point 2, …, then point 1, point 2, … again "
            "(NOT point 1, point 1, …). For back-to-back repeats at a single "
            "point, use 'Repetitions per point' instead.")
        self._repeat_sample_spin.valueChanged.connect(self._refresh_preview)
        grid.addLayout(self._entry_with_unit(self._repeat_sample_spin, ""), 0, 1)
        return group

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
        is_running = new_state in ('moving', 'focusing', 'acquiring', 'processing')
        is_paused = new_state == 'paused'

        self._stop_btn.setEnabled(is_running or is_paused)

        # The single primary button reflects the current state.
        if is_paused:
            self._start_btn.setText("Resume")
        elif is_running:
            self._start_btn.setText("Pause")
        else:
            self._start_btn.setText("Start")
        # Gate the Start button by the start prerequisites (when idle).
        self._update_start_enabled()

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
        self._update_start_enabled()

    def _on_primary_clicked(self):
        """The single primary button: Start when idle, Pause while running,
        Resume while paused."""
        if self._cfov_running:
            if self._cfov_paused:
                self.resume_requested.emit()
            else:
                self.pause_requested.emit()
            return
        self._on_start_clicked()

    def _on_start_clicked(self):
        """Handle start button click."""
        # Safety gate: if the stage has drifted from the capture position in
        # Current FoV mode, confirm before doing anything (the return move can
        # be long-distance / crash-prone). Cancel leaves everything untouched
        # so the user can go back and adjust.
        if not self._confirm_stage_move_for_start():
            self._log("Start cancelled. Adjust the stage / Z height and retry")
            return

        # Resume a paused Current-FoV run rather than starting a new one.
        if self._cfov_paused:
            self.resume_requested.emit()
            return

        # Push the mirrored parameters back to the Live tab first, so both tabs
        # stay consistent for the run.
        self.push_to_live()
        self._log("Parameters synced to Live tab")

        # Current FoV: hand off to the host window's acquisition worker, which
        # runs over the points generated on the captured image.
        if self._mode_current_fov.isChecked():
            if self._current_fov_point_count() == 0:
                self._log("No sample points generated", error=True)
                return
            self._progress_bar.setValue(0)
            self.acquisition_requested.emit(self.get_acquisition_parameters())
            return

        # Robotic Deposition: hand off to the host window's worker, which moves
        # to each selected droplet and images + segments + acquires it at run time.
        positions = self._sample_grid.get_selected_positions()
        if not positions:
            self._log("No droplets selected", error=True)
            return
        self._progress_bar.setValue(0)
        params = dict(self.get_acquisition_parameters())
        params['droplets'] = list(positions)  # [(row, col), …]
        self.robotic_requested.emit(params)

    def _on_pause_clicked(self):
        """Handle pause button click."""
        if self._cfov_running:
            if self._cfov_paused:
                self.resume_requested.emit()
            else:
                self.pause_requested.emit()
            return
        try:
            self._state_machine.pause()
            self._log("Acquisition paused")
        except Exception as e:
            self._log(f"Cannot pause: {e}", error=True)

    def _on_stop_clicked(self):
        """Handle stop button click."""
        if self._cfov_running:
            self._cfov_stopping = True
            self.stop_requested.emit()
            # Reset the progress display to a neutral state immediately; the
            # _cfov_stopping flag keeps late worker updates from overwriting it.
            self._progress_bar.setValue(0)
            self._progress_label.setText("None")
            return
        try:
            self._state_machine.reset()
            self._progress_bar.setValue(0)
            self._progress_label.setText("Stopped")
            self._log("Acquisition stopped")
            self.acquisition_stopped.emit()
        except Exception as e:
            self._log(f"Cannot stop: {e}", error=True)

    # --- host-driven Current-FoV run callbacks ------------------------------
    def on_acquisition_started(self):
        """The host window started the Current-FoV worker."""
        self._cfov_running = True
        self._cfov_paused = False
        self._progress_bar.setValue(0)
        # Start button becomes Pause; it stays enabled to toggle Pause/Resume.
        self._start_btn.setEnabled(True)
        self._start_btn.setText("Pause")
        self._stop_btn.setEnabled(True)

    def on_acquisition_paused(self, paused: bool):
        """Reflect pause/resume on the single primary button."""
        if not self._cfov_running:
            return
        self._cfov_paused = paused
        self._start_btn.setText("Resume" if paused else "Pause")

    def on_acquisition_progress(self, done: int, total: int, message: str):
        """Update the progress bar / label during a run."""
        # Once Stop is pressed, leave the reset ("None") display alone.
        if self._cfov_stopping:
            return
        if total > 0:
            self._progress_bar.setValue(int(100 * done / total))
        self._progress_label.setText(
            f"{message} ({done}/{total})" if total > 0 else message)
        if not self._cfov_stopping:
            self._highlight_step(message)

    def on_acquisition_status(self, message: str):
        """Update the status label with a non-numeric status message."""
        self._log(message)
        if not self._cfov_stopping:
            self._progress_label.setText(message)
            self._highlight_step(message)

    def on_acquisition_finished(self, aborted: bool):
        """The host window's worker finished (or was stopped)."""
        self._cfov_running = False
        self._cfov_paused = False
        # Clear the live step highlight.
        self._highlight_idx = -1
        self._render_preview()
        # Primary button reverts to Start (re-gated by prerequisites); Stop off.
        self._start_btn.setText("Start")
        self._stop_btn.setEnabled(False)
        self._update_start_enabled()
        if aborted:
            # Stopped by the user -> neutral ("None") progress display.
            self._progress_bar.setValue(0)
            self._progress_label.setText("None")
        else:
            self._progress_bar.setValue(100)
            self._progress_label.setText("Complete")
        self._cfov_stopping = False
        self.acquisition_stopped.emit()

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
            self._update_start_enabled()

    def _log(self, message: str, error: bool = False):
        """Record an activity message.

        The on-screen Activity Log was removed (the Acquisition Steps Preview
        covers what the run will do); messages now go to the application log.
        """
        if error:
            logger.error(message)
        else:
            logger.info(message)

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
            'autofocus_widefield_prusa': self._af_widefield_prusa.isChecked(),
            'autofocus_wasatch_prusa': self._af_wasatch_prusa.isChecked(),
            'autofocus_wasatch_nanodrive': self._af_wasatch_nanodrive.isChecked(),
            'z_stack': self._z_stack_check.isChecked(),
            'wait_before_measure': self._wait_check.isChecked(),
            'wait_time_s': self._wait_time_spin.value(),
            'save_path': self._save_path_edit.text(),
            'filename_prefix': self._prefix_edit.text().strip(),
            'sample_pattern': self.selected_pattern_type(),
        }
        # Mirrored Live-tab parameters
        for key in self._bindings:
            params[key] = self._binding_value(key)
        return params

    def set_state_machine(self, state_machine: AcquisitionStateMachine):
        """Set or update the state machine."""
        self._state_machine = state_machine
        self._connect_signals()
