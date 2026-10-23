"""
Stage control widget for PyQt6 with modern dark theme.

Provides UI for controlling the PRUSA 3D printer stage with:
- Reference/homing buttons with status indicators
- Position display and manual entry
- Directional movement pad
- Speed control slider
- NanoDrive controls
"""

from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QPushButton, QLabel, QLineEdit, QSlider,
    QDoubleSpinBox, QFrame, QSizePolicy, QMessageBox,
    QSpacerItem, QAbstractSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QFont

from raman_control.gui.threads.motion_thread import MotionThread
from raman_control.gui.styles import COLORS
from raman_control.hardware.nano_piezo import NanoPiezo
from raman_control.core.coordinate_manager import CoordinateManager, CameraView
from raman_control.config.hardware_config import StageConfig, NanoDriveConfig
from raman_control.config.logging_config import get_logger

logger = get_logger('stage_control_widget')


class StatusIndicator(QLabel):
    """Small circular status indicator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self._active = False
        self._update_style()

    def set_active(self, active: bool):
        self._active = active
        self._update_style()

    def _update_style(self):
        color = '#4caf50' if self._active else '#606060'
        self.setStyleSheet(f'''
            background-color: {color};
            border-radius: 6px;
            border: 1px solid #3d3d3d;
        ''')


_PAD_BTN_SIZE = 32  # Square buttons

# Per-step visual: bigger arrow + darker bg for larger steps
_STEP_LOOK = {
    'large':  {'font': 22, 'bg': '#252525', 'hover': '#353535', 'border': '#4d4d4d'},
    'medium': {'font': 16, 'bg': '#3c3c3c', 'hover': '#4c4c4c', 'border': '#4d4d4d'},
    'small':  {'font': 10, 'bg': '#585858', 'hover': '#686868', 'border': '#6d6d6d'},
}

def _btn_style(size: str) -> str:
    s = _PAD_BTN_SIZE
    lk = _STEP_LOOK[size]
    return f'''
        QPushButton {{
            background-color: {lk['bg']};
            border: 1px solid {lk['border']};
            border-radius: 3px;
            font-size: {lk['font']}px;
            color: #e0e0e0;
            padding: 0px;
            min-width: {s}px; max-width: {s}px;
            min-height: {s}px; max-height: {s}px;
        }}
        QPushButton:hover {{ background-color: {lk['hover']}; border-color: #00a0a0; }}
        QPushButton:pressed {{ background-color: #2d2d2d; }}
        QPushButton:disabled {{ color: #505050; background-color: #2a2a2a; }}
    '''

_STEP_LABEL_STYLE = 'color: #ffffff; font-size: 12px;'


class DirectionPad(QWidget):
    """Compact direction control pad for XY movement."""

    move_requested = pyqtSignal(float, float)  # dx, dy

    def __init__(self, step_sizes: dict, parent=None):
        super().__init__(parent)
        self._step_sizes = step_sizes
        self._enabled = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QGridLayout(self)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)

        bs = _PAD_BTN_SIZE

        #        col0  col1  col2  col3  col4  col5  col6
        # row 0              [lbl] Y+L
        # row 1              [lbl] Y+M
        # row 2              [lbl] Y+S
        # row 3  X-L   X-M   X-S  XY   X+S   X+M   X+L
        # row 4                    Y-S [lbl]
        # row 5                    Y-M [lbl]
        # row 6                    Y-L [lbl]

        # Y+ buttons (top, large=far, small=near) — col 3, labels left at col 2
        for i, size in enumerate(['large', 'medium', 'small']):
            btn = QPushButton('\u25b2')  # ▲
            btn.setFixedSize(bs, bs)
            btn.setStyleSheet(_btn_style(size))
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, s=size: self._emit_move(0, self._step_sizes[s]))
            layout.addWidget(btn, i, 3, Qt.AlignmentFlag.AlignCenter)
            lbl = QLabel(str(self._step_sizes[size]))
            lbl.setStyleSheet(_STEP_LABEL_STYLE)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(lbl, i, 2)
            setattr(self, f'btn_y_plus_{size}', btn)

        # Y- buttons (bottom, small=near, large=far) — col 3, labels right at col 4
        for i, size in enumerate(['small', 'medium', 'large']):
            btn = QPushButton('\u25bc')  # ▼
            btn.setFixedSize(bs, bs)
            btn.setStyleSheet(_btn_style(size))
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, s=size: self._emit_move(0, -self._step_sizes[s]))
            layout.addWidget(btn, 4 + i, 3, Qt.AlignmentFlag.AlignCenter)
            # No step label here — the 3 Y+ labels (col 2) already cover the Y
            # axis; the other set of 3 now labels the X axis (on the X+ buttons).
            setattr(self, f'btn_y_minus_{size}', btn)

        # X- buttons (left, large=far, small=near) — cols 0,1,2
        for i, size in enumerate(['large', 'medium', 'small']):
            btn = QPushButton('\u25c0')  # ◀
            btn.setFixedSize(bs, bs)
            btn.setStyleSheet(_btn_style(size))
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, s=size: self._emit_move(-self._step_sizes[s], 0))
            layout.addWidget(btn, 3, i, Qt.AlignmentFlag.AlignCenter)
            setattr(self, f'btn_x_minus_{size}', btn)

        # X+ buttons (right, small=near, large=far) — cols 4,5,6
        for i, size in enumerate(['small', 'medium', 'large']):
            btn = QPushButton('\u25b6')  # ▶
            btn.setFixedSize(bs, bs)
            btn.setStyleSheet(_btn_style(size))
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, s=size: self._emit_move(self._step_sizes[s], 0))
            layout.addWidget(btn, 3, 4 + i, Qt.AlignmentFlag.AlignCenter)
            # X-axis step label above each X+ button (the set moved off Y).
            lbl = QLabel(str(self._step_sizes[size]))
            lbl.setStyleSheet(_STEP_LABEL_STYLE)
            lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter
                             | Qt.AlignmentFlag.AlignBottom)
            layout.addWidget(lbl, 2, 4 + i)
            setattr(self, f'btn_x_plus_{size}', btn)

        # Center label — unit shown once here
        center = QLabel('XY\nmm')
        center.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.setStyleSheet('color: #ffffff; font-weight: bold; font-size: 14px;')
        layout.addWidget(center, 3, 3, Qt.AlignmentFlag.AlignCenter)

    def _emit_move(self, dx: float, dy: float):
        self.move_requested.emit(dx, dy)

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        for size in ['large', 'medium', 'small']:
            for direction in ['x_plus', 'x_minus', 'y_plus', 'y_minus']:
                btn = getattr(self, f'btn_{direction}_{size}', None)
                if btn:
                    btn.setEnabled(enabled)


class ZControlPad(QWidget):
    """Vertical Z-axis control pad."""

    move_requested = pyqtSignal(float)  # dz

    def __init__(self, step_sizes: dict, label: str = 'Z',
                 unit: str = 'mm', parent=None):
        super().__init__(parent)
        self._step_sizes = step_sizes
        self._label = label
        self._unit = unit
        self._enabled = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QGridLayout(self)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)

        bs = _PAD_BTN_SIZE

        row = 0

        # Up buttons (large to small) with step size labels
        for size in ['large', 'medium', 'small']:
            btn = QPushButton('\u25b2')  # ▲
            btn.setFixedSize(bs, bs)
            btn.setStyleSheet(_btn_style(size))
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, s=size: self.move_requested.emit(self._step_sizes[s]))
            layout.addWidget(btn, row, 0, Qt.AlignmentFlag.AlignCenter)

            step_label = QLabel(str(self._step_sizes[size]))
            step_label.setStyleSheet(_STEP_LABEL_STYLE)
            layout.addWidget(step_label, row, 1)

            setattr(self, f'btn_up_{size}', btn)
            row += 1

        # Center label — unit shown once here
        center = QLabel(f'{self._label}\n{self._unit}')
        center.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.setStyleSheet('color: #ffffff; font-weight: bold; font-size: 14px;')
        layout.addWidget(center, row, 0, Qt.AlignmentFlag.AlignCenter)
        row += 1

        # Down buttons (small to large) with step size labels
        for size in ['small', 'medium', 'large']:
            btn = QPushButton('\u25bc')  # ▼
            btn.setFixedSize(bs, bs)
            btn.setStyleSheet(_btn_style(size))
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, s=size: self.move_requested.emit(-self._step_sizes[s]))
            layout.addWidget(btn, row, 0, Qt.AlignmentFlag.AlignCenter)

            step_label = QLabel(str(self._step_sizes[size]))
            step_label.setStyleSheet(_STEP_LABEL_STYLE)
            layout.addWidget(step_label, row, 1)

            setattr(self, f'btn_down_{size}', btn)
            row += 1

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        for size in ['large', 'medium', 'small']:
            for direction in ['up', 'down']:
                btn = getattr(self, f'btn_{direction}_{size}', None)
                if btn:
                    btn.setEnabled(enabled)


class StageControlWidget(QWidget):
    """
    Stage control panel widget with dark theme.

    Signals:
        position_updated: Emitted when position changes (x, y, z)
        referenced_changed: Emitted when reference status changes
        ranges_registered: Emitted when autofocus ranges are registered. Read
            the values via get_widefield_z_limits / get_objective_z_limits /
            get_nanodrive_limits.
        registration_reset: Emitted when a previously-registered autofocus range
            is invalidated (e.g. on disconnect) and must be registered again.
    """

    position_updated = pyqtSignal(float, float, float)
    referenced_changed = pyqtSignal(dict)
    ranges_registered = pyqtSignal()
    registration_reset = pyqtSignal()

    def __init__(
        self,
        motion_thread: MotionThread,
        config: Optional[StageConfig] = None,
        nanodrive: Optional[NanoPiezo] = None,
        nanodrive_config: Optional[NanoDriveConfig] = None,
        coordinate_manager: Optional[CoordinateManager] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self._motion = motion_thread
        self._config = config or StageConfig()
        self._nanodrive = nanodrive
        self._nd_config = nanodrive_config or NanoDriveConfig()
        self._coord_mgr = coordinate_manager
        # Optional absolute-move handler (physical x, y, z) set by the main
        # window so Go moves hop over slide barriers in objective view.
        self._move_handler = None

        # Current state
        self._current_position = (0.0, 0.0, 0.0)
        self._nd_position = 0.0
        # Autofocus Z ranges are only valid once the user presses Register.
        self._ranges_registered = False
        self._referenced = {'X': False, 'Y': False, 'Z': False}

        # Build UI
        self._setup_ui()
        self._connect_signals()

        # Position update timer
        self._position_timer = QTimer(self)
        self._position_timer.timeout.connect(self._request_position_update)
        self._position_timer.start(500)

        # Initial button states (disconnected by default)
        self._update_button_states()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # === Reference Section ===
        #
        # Grid layout:
        #   row 0:  [X Y Z ND indicators]  |  [Go Home]       |  [Calibrate Home]
        #   row 1:  [Reference All btn   ]  |  (spans row 0-1) |  (spans row 0-1)
        #
        ref_group = QGroupBox("Reference & Home")
        ref_grid = QGridLayout(ref_group)
        ref_grid.setSpacing(6)
        ref_grid.setContentsMargins(4, 4, 4, 4)

        # Col 0: indicators + Reference All
        indicator_row = QHBoxLayout()
        indicator_row.setContentsMargins(0, 0, 0, 0)
        self._ref_indicators = {}
        for name in ['X', 'Y', 'Z', 'ND']:
            indicator = StatusIndicator()
            self._ref_indicators[name] = indicator

            label = QLabel(name)
            label.setStyleSheet('color: #a0a0a0; font-weight: 600;')

            pair = QHBoxLayout()
            pair.setSpacing(3)
            pair.addWidget(indicator)
            pair.addWidget(label)
            indicator_row.addLayout(pair)
            indicator_row.addStretch(1)
        ref_grid.addLayout(indicator_row, 0, 0)

        self._ref_all_btn = QPushButton("Reference All")
        # Same solid-cyan style as Go Home (transparent outline when disabled).
        self._ref_all_btn.setStyleSheet('''
            QPushButton {
                background-color: #00b4b4;
                border: 1px solid #00d4d4;
                border-radius: 5px;
                color: #1a1a1a;
                font-weight: 600;
                padding: 4px 8px;
            }
            QPushButton:hover { background-color: #00d4d4; }
            QPushButton:pressed { background-color: #008080; }
            QPushButton:disabled {
                background-color: transparent;
                border: 1px solid #4d4d4d;
                color: #505050;
            }
        ''')
        self._ref_all_btn.clicked.connect(self._on_ref_all_clicked)
        ref_grid.addWidget(self._ref_all_btn, 1, 0)

        # Col 1: Go Home (spans rows 0-1)
        # Solid cyan when enabled, transparent outline when disabled
        self._home_btn = QPushButton("Go\nHome")
        self._home_btn.setEnabled(False)
        self._home_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._home_btn.setStyleSheet('''
            QPushButton {
                background-color: #00b4b4;
                border: 1px solid #00d4d4;
                border-radius: 5px;
                color: #1a1a1a;
                font-weight: 600;
                padding: 4px 8px;
            }
            QPushButton:hover { background-color: #00d4d4; }
            QPushButton:pressed { background-color: #008080; }
            QPushButton:disabled {
                background-color: transparent;
                border: 1px solid #4d4d4d;
                color: #505050;
            }
        ''')
        self._home_btn.clicked.connect(self._on_home_clicked)
        ref_grid.addWidget(self._home_btn, 0, 1, 2, 1)

        # Col 2: Calibrate Home (spans rows 0-1, outline style)
        self._calibrate_home_btn = QPushButton("Calibrate\nHome")
        self._calibrate_home_btn.setEnabled(False)
        self._calibrate_home_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._calibrate_home_btn.setStyleSheet('''
            QPushButton {
                background-color: transparent;
                border: 1px solid #4d4d4d;
                border-radius: 5px;
                color: #606060;
                font-size: 12px;
                padding: 4px 8px;
            }
            QPushButton:enabled {
                border-color: #00b4b4;
                color: #00b4b4;
            }
            QPushButton:enabled:hover {
                background-color: rgba(0, 180, 180, 30);
                border-color: #00d4d4;
                color: #00d4d4;
            }
            QPushButton:enabled:pressed {
                background-color: rgba(0, 180, 180, 60);
            }
            QPushButton:disabled {
                border-color: #3d3d3d;
                color: #505050;
            }
        ''')
        self._calibrate_home_btn.clicked.connect(self._on_calibrate_home_clicked)
        ref_grid.addWidget(self._calibrate_home_btn, 0, 2, 2, 1)

        # Column stretch: buttons get more space, reference column stays compact
        ref_grid.setColumnStretch(0, 1)
        ref_grid.setColumnStretch(1, 2)
        ref_grid.setColumnStretch(2, 2)

        ref_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        main_layout.addWidget(ref_group)

        # === Position Display ===
        pos_group = QGroupBox("Position")
        pos_layout = QGridLayout(pos_group)
        pos_layout.setHorizontalSpacing(6)
        pos_layout.setVerticalSpacing(8)
        pos_layout.setContentsMargins(6, 8, 6, 6)

        # Column stretch: label fixed, data columns equal
        pos_layout.setColumnStretch(0, 0)
        for c in range(1, 5):
            pos_layout.setColumnStretch(c, 1)

        # Headers
        headers = ['', 'X [mm]', 'Y [mm]', 'Z [mm]', 'ND [μm]']
        for col, header in enumerate(headers):
            label = QLabel(header)
            label.setStyleSheet('color: #00b4b4; font-weight: 600;')
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pos_layout.addWidget(label, 0, col)

        # Current position
        pos_layout.addWidget(QLabel("Current:"), 1, 0)

        self._pos_labels = {}
        for col, axis in enumerate(['X', 'Y', 'Z', 'ND']):
            label = QLabel("-")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFixedHeight(26)
            label.setMinimumWidth(48)
            label.setStyleSheet('''
                font-family: "SF Mono", "Consolas", monospace;
                font-size: 12px;
            ''')
            self._pos_labels[axis] = label
            pos_layout.addWidget(label, 1, col + 1)

        # Target position (no spin arrows — clean text input)
        pos_layout.addWidget(QLabel("Target:"), 2, 0)

        target_style = '''
            QDoubleSpinBox {
                padding: 4px 2px;
                font-family: "SF Mono", "Consolas", monospace;
                font-size: 12px;
                qproperty-alignment: AlignCenter;
            }
        '''

        self._target_inputs = {}
        # Per-axis ranges from stage config (single source of truth)
        if self._config:
            _axis_ranges = {
                'X': (self._config.x_min, self._config.x_max),
                'Y': (self._config.y_min, self._config.y_max),
                'Z': (self._config.z_min, self._config.z_max),
            }
        else:
            _axis_ranges = {'X': (0, 250), 'Y': (0, 210), 'Z': (0, 200)}

        for col, axis in enumerate(['X', 'Y', 'Z']):
            spin = QDoubleSpinBox()
            spin.setRange(*_axis_ranges[axis])
            spin.setDecimals(2)
            spin.setSingleStep(1.0)
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setFixedHeight(26)
            spin.setMinimumWidth(48)
            spin.setStyleSheet(target_style)
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
            spin.setValue(getattr(self._config, f'home_{axis.lower()}', 0.0))
            self._target_inputs[axis] = spin
            pos_layout.addWidget(spin, 2, col + 1)

        # NanoDrive target
        self._nd_target = QDoubleSpinBox()
        self._nd_target.setRange(self._nd_config.min_position, self._nd_config.max_position)
        self._nd_target.setDecimals(1)
        self._nd_target.setValue(self._nd_config.initial_position)
        self._nd_target.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._nd_target.setFixedHeight(26)
        self._nd_target.setMinimumWidth(48)
        self._nd_target.setStyleSheet(target_style)
        self._nd_target.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pos_layout.addWidget(self._nd_target, 2, 4)

        # Go button — solid cyan when enabled, transparent outline when disabled
        self._go_btn = QPushButton("Go to Target")
        self._go_btn.setEnabled(False)
        self._go_btn.setStyleSheet('''
            QPushButton {
                background-color: #00b4b4;
                border: 1px solid #00d4d4;
                border-radius: 5px;
                color: #1a1a1a;
                font-weight: 600;
                padding: 5px 12px;
                min-height: 18px;
            }
            QPushButton:hover { background-color: #00d4d4; }
            QPushButton:pressed { background-color: #008080; }
            QPushButton:disabled {
                background-color: transparent;
                border: 1px solid #4d4d4d;
                color: #505050;
            }
        ''')
        self._go_btn.clicked.connect(self._on_go_clicked)
        self._go_btn.setMinimumWidth(96)
        # Sit to the RIGHT of the ND column, level with the Target entry row.
        pos_layout.addWidget(self._go_btn, 2, 5)
        pos_layout.setColumnStretch(5, 0)

        # Spacer between the position rows and the speed slider
        pos_layout.setRowMinimumHeight(3, 8)

        # Speed slider row
        speed_row = QHBoxLayout()
        speed_row.setSpacing(6)

        slow_label = QLabel("Slow")
        slow_label.setFixedWidth(30)
        speed_row.addWidget(slow_label)

        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(10, self._config.default_speed)
        self._speed_slider.setValue(self._config.default_speed)
        self._speed_slider.valueChanged.connect(self._on_speed_label_update)
        self._speed_slider.sliderReleased.connect(self._on_speed_committed)
        speed_row.addWidget(self._speed_slider, 1)

        fast_label = QLabel("Fast")
        fast_label.setFixedWidth(28)
        speed_row.addWidget(fast_label)

        self._speed_label = QLabel(f"{self._config.default_speed} mm/min")
        self._speed_label.setFixedWidth(90)
        self._speed_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._speed_label.setStyleSheet('color: #00b4b4;')
        speed_row.addWidget(self._speed_label)

        pos_layout.addLayout(speed_row, 4, 0, 1, 6)

        pos_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        main_layout.addWidget(pos_group)

        # === Manual Movement ===
        move_group = QGroupBox("Manual Movement")
        move_layout = QHBoxLayout(move_group)
        move_layout.setSpacing(8)

        xy_steps = {
            'small': self._config.small_step_xy,
            'medium': self._config.medium_step_xy,
            'large': self._config.large_step_xy
        }
        self._xy_pad = DirectionPad(xy_steps)
        self._xy_pad.move_requested.connect(
            lambda dx, dy: self._motion.request_move_by(x=dx, y=dy)
        )
        move_layout.addWidget(self._xy_pad)
        move_layout.addStretch()

        z_steps = {
            'small': self._config.small_step_z,
            'medium': self._config.medium_step_z,
            'large': self._config.large_step_z
        }
        self._z_pad = ZControlPad(z_steps, 'Z', unit='mm')
        self._z_pad.move_requested.connect(
            lambda dz: self._motion.request_move_by(z=dz)
        )
        move_layout.addWidget(self._z_pad)
        move_layout.addStretch()

        nd_steps = {'small': 1.0, 'medium': 5.0, 'large': 10.0}
        self._nd_pad = ZControlPad(nd_steps, 'ND', unit='\u03bcm')
        self._nd_pad.move_requested.connect(self._on_nd_move_requested)
        move_layout.addWidget(self._nd_pad)

        move_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        main_layout.addWidget(move_group)

        # === Autofocus Section ===
        af_group = QGroupBox("Autofocus")
        af_layout = QVBoxLayout(af_group)
        af_layout.setSpacing(6)

        # Z limits row — ranges and defaults from stage config
        if self._config:
            _af_z_min, _af_z_max = self._config.z_min, self._config.z_max
            _af_upper, _af_lower = self._config.af_z_upper, self._config.af_z_lower
        else:
            _af_z_min, _af_z_max = 0.0, 200.0
            _af_upper, _af_lower = 40.0, 30.0

        # Prusa Z autofocus ranges in two camera frames — Widefield (WF) and
        # Objective (Obj) — plus the NanoDrive fine range. Registering in one
        # view fills the other via the wide->objective Z offset, and only the
        # active view's Prusa boxes are editable (noodlepy model).
        _nd_min, _nd_max = self._nd_config.min_position, self._nd_config.max_position
        _off_z = self._coord_mgr.offset_z if self._coord_mgr else 0.0

        limits_grid = QGridLayout()
        limits_grid.setHorizontalSpacing(6)
        limits_grid.setVerticalSpacing(6)

        for col, text in ((1, "WF [mm]"), (2, "Obj [mm]"), (3, "ND [µm]")):
            hdr = QLabel(text)
            hdr.setStyleSheet(f'color: {COLORS["accent_primary"]}; font-weight: 600;')
            hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
            limits_grid.addWidget(hdr, 0, col)

        def _af_spin(low, high, val):
            sb = QDoubleSpinBox()
            sb.setRange(low, high)
            sb.setValue(val)
            sb.setDecimals(1)
            sb.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            sb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sb.setFixedWidth(64)
            return sb

        def _clamp(v):
            return max(_af_z_min, min(_af_z_max, v))

        # Row 1: highest — WF | Obj | NanoDrive
        limits_grid.addWidget(QLabel("Highest:"), 1, 0)
        self._wf_z_upper = _af_spin(_af_z_min, _af_z_max, _af_upper)
        limits_grid.addWidget(self._wf_z_upper, 1, 1)
        self._obj_z_upper = _af_spin(_af_z_min, _af_z_max, _clamp(_af_upper + _off_z))
        limits_grid.addWidget(self._obj_z_upper, 1, 2)
        self._nano_upper = _af_spin(_nd_min, _nd_max, min(90.0, _nd_max))
        limits_grid.addWidget(self._nano_upper, 1, 3)

        # Row 2: lowest
        limits_grid.addWidget(QLabel("Lowest:"), 2, 0)
        self._wf_z_lower = _af_spin(_af_z_min, _af_z_max, _af_lower)
        limits_grid.addWidget(self._wf_z_lower, 2, 1)
        self._obj_z_lower = _af_spin(_af_z_min, _af_z_max, _clamp(_af_lower + _off_z))
        limits_grid.addWidget(self._obj_z_lower, 2, 2)
        self._nano_lower = _af_spin(_nd_min, _nd_max, max(10.0, _nd_min))
        limits_grid.addWidget(self._nano_lower, 2, 3)

        # Register all ranges (spans the two limit rows).
        self._register_btn = QPushButton("Register")
        self._register_btn.setStyleSheet(f'''
            QPushButton {{
                background-color: {COLORS['accent_primary']};
                border: 1px solid {COLORS['accent_hover']};
                border-radius: 5px;
                color: {COLORS['bg_dark']};
                font-weight: 600;
                padding: 5px 10px;
            }}
            QPushButton:hover {{ background-color: {COLORS['accent_hover']}; }}
            QPushButton:disabled {{
                background-color: transparent;
                border: 1px solid #4d4d4d;
                color: #505050;
            }}
        ''')
        # Spans the two limit rows and FILLS them vertically (Expanding), so its
        # top aligns with the Highest entry box and its bottom with the Lowest
        # entry box.
        self._register_btn.setSizePolicy(QSizePolicy.Policy.Preferred,
                                         QSizePolicy.Policy.Expanding)
        self._register_btn.clicked.connect(self._on_register_ranges)
        limits_grid.addWidget(self._register_btn, 1, 4, 2, 1)

        # Best-focus Z from each autofocus, stacked to the right of Register.
        focus_box = QVBoxLayout()
        focus_box.setSpacing(2)
        focus_box.setContentsMargins(0, 0, 0, 0)
        self._focus_lbls = {}
        for kind, text in (('cam_prusa', 'Cam Prusa'),
                           ('spec_prusa', 'Spec Prusa'),
                           ('spec_nano', 'Spec ND')):
            lbl = QLabel(f'{text} Focus: N/A')
            lbl.setStyleSheet('font-size: 10px; color: #ffffff;')
            self._focus_lbls[kind] = (lbl, text)
            focus_box.addWidget(lbl)
        limits_grid.addLayout(focus_box, 0, 5, 3, 1)

        limits_grid.setColumnStretch(6, 1)  # absorb extra space past the readouts
        af_layout.addLayout(limits_grid)

        # Only the active camera view's Prusa boxes are editable.
        self._update_af_editability()

        af_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        main_layout.addWidget(af_group)

    def _connect_signals(self):
        """Connect motion thread signals."""
        self._motion.position_changed.connect(self._on_position_changed)
        self._motion.axis_referenced.connect(self._on_axis_referenced)
        self._motion.movement_complete.connect(self._on_movement_complete)
        self._motion.connected.connect(self._on_connected)
        self._motion.disconnected.connect(self._on_disconnected)
        self._motion.error.connect(self._on_error)

    def _on_position_changed(self, x: float, y: float, z: float):
        # Show the REAL physical stage position (no sample-centric conversion).
        self._current_position = (x, y, z)
        self._pos_labels['X'].setText(f"{x:.2f}")
        self._pos_labels['Y'].setText(f"{y:.2f}")
        self._pos_labels['Z'].setText(f"{z:.2f}")
        self.position_updated.emit(x, y, z)

    def _on_axis_referenced(self, axis: str):
        if axis in self._referenced:
            self._referenced[axis] = True
            if axis in self._ref_indicators:
                self._ref_indicators[axis].set_active(True)

        self._update_button_states()
        self.referenced_changed.emit(self._referenced)
        logger.info(f"Axis {axis} referenced")

        # Resume position polling once all axes are referenced
        if all(self._referenced.values()) and not self._position_timer.isActive():
            self._position_timer.start(500)

    def _on_movement_complete(self):
        self._request_position_update()

    def _on_connected(self):
        logger.info("Stage connected")
        self._update_button_states()

    def _on_disconnected(self):
        logger.warning("Stage disconnected")
        self._referenced = {'X': False, 'Y': False, 'Z': False}
        for indicator in self._ref_indicators.values():
            indicator.set_active(False)
        # The autofocus range is anchored to the (now-lost) home position, so it
        # is invalidated too: the user must re-register after re-referencing.
        was_registered = self._ranges_registered
        self._ranges_registered = False
        self._update_button_states()
        # The home reference is lost on disconnect — tell the rest of the GUI so
        # reference-gated features (FOV switch, sample grid, move-to-point) lock
        # again until the stage is re-referenced.
        self.referenced_changed.emit(self._referenced)
        if was_registered:
            # Re-lock the autofocus-range-gated buttons (Switch to Objective,
            # camera autofocus, acquisition Start) until re-registration.
            self.registration_reset.emit()

    def _on_error(self, message: str):
        logger.error(f"Motion error: {message}")
        # Don't show dialog for "Stage not connected" - this is expected
        # when hardware isn't present. Use status indicator instead.
        if "not connected" not in message.lower():
            QMessageBox.warning(self, "Motion Error", message)

    def _on_ref_all_clicked(self):
        if not self._motion.is_connected:
            self._show_not_connected_warning("Stage")
            return
        reply = QMessageBox.question(
            self,
            "Reference All Axes",
            "This will home all axes (X, Y, Z) and the NanoDrive.\n"
            "The stage will move.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            # Pause position polling during homing to avoid M114 commands
            # racing with the G28/M400 sequence (causes glitches & bare 'ok')
            self._position_timer.stop()
            self._motion.request_home('XYZ')

            # Reference NanoDrive (move to 50 um resting position)
            if self._nanodrive and self._nanodrive.is_connected:
                self._nanodrive.home()
                self._ref_indicators['ND'].set_active(True)
                self._update_nd_position()
                logger.info("NanoDrive referenced (moved to resting position)")

    def _show_not_connected_warning(self, device: str):
        """Show a brief warning that device is not connected."""
        logger.warning(f"{device} not connected - action blocked")
        # Use a non-blocking approach - just update the status indicator
        # The user can see the connection status in the UI

    def _on_calibrate_home_clicked(self):
        """Save the current position as the home position."""
        x, y, z = self._current_position
        self._config.home_x = x
        self._config.home_y = y
        self._config.home_z = z

        # Update target inputs to reflect new home
        self._target_inputs['X'].setValue(x)
        self._target_inputs['Y'].setValue(y)
        self._target_inputs['Z'].setValue(z)

        logger.info(f"Home position calibrated to X={x:.2f} Y={y:.2f} Z={z:.2f}")

    def _on_home_clicked(self):
        if not self._motion.is_connected:
            self._show_not_connected_warning("Stage")
            return
        if self._motion.all_referenced:
            if self._coord_mgr:
                # Convert logical home to physical coords for current view
                px, py, pz = self._coord_mgr.logical_to_physical(
                    self._config.home_x, self._config.home_y, self._config.home_z
                )
                self._motion.request_go_home(x=px, y=py, z=pz)
            else:
                self._motion.request_go_home()

    def _on_go_clicked(self):
        if not self._motion.is_connected:
            self._show_not_connected_warning("Stage")
            return
        try:
            x = self._target_inputs['X'].value()
            y = self._target_inputs['Y'].value()
            z = self._target_inputs['Z'].value()
            # Target inputs are in logical coords — convert to physical
            if self._coord_mgr:
                x, y, z = self._coord_mgr.logical_to_physical(x, y, z)
            if self._move_handler is not None:
                self._move_handler(x, y, z)
            else:
                self._motion.request_move_to(x=x, y=y, z=z)

            # Also move nanodrive to its target
            nd_target = self._nd_target.value()
            if self._nanodrive and self._nanodrive.is_connected:
                self._nanodrive.move_to(z=nd_target)
                self._update_nd_position()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _on_speed_label_update(self, value: int):
        """Update the display label while dragging (no hardware command)."""
        self._speed_label.setText(f"{value} mm/min")

    def _on_speed_committed(self):
        """Set the actual speed only when slider is released."""
        value = self._speed_slider.value()
        self._motion.speed = value

    def _on_nd_move_requested(self, dz: float):
        """Handle NanoDrive pad button press."""
        if self._nanodrive and self._nanodrive.is_connected:
            self._nanodrive.move_by(z=dz)
            self._update_nd_position()
        else:
            self._show_not_connected_warning("NanoDrive")

    def _update_nd_position(self):
        """Read nanodrive position and update the display label."""
        if self._nanodrive and self._nanodrive.is_connected:
            pos = self._nanodrive.get_position()
            self._nd_position = pos[0]
            self._pos_labels['ND'].setText(f"{self._nd_position:.1f}")

    def _request_position_update(self):
        if self._motion.is_connected and self._motion.all_referenced:
            self._motion.request_position_update()
        self._update_nd_position()

    def _update_button_states(self):
        all_ref = all(self._referenced.values())
        connected = self._motion.is_connected
        nd_connected = self._nanodrive.is_connected if self._nanodrive else False

        self._ref_all_btn.setEnabled(connected)

        self._home_btn.setEnabled(all_ref and connected)
        self._calibrate_home_btn.setEnabled(all_ref and connected)
        self._go_btn.setEnabled(all_ref and connected)
        # Registering autofocus ranges needs a referenced stage.
        self._register_btn.setEnabled(all_ref and connected)

        self._xy_pad.set_enabled(all_ref and connected)
        self._z_pad.set_enabled(all_ref and connected)
        self._nd_pad.set_enabled(all_ref and connected and nd_connected)

    def _on_register_ranges(self):
        """Register the autofocus ranges.

        The active camera view's Prusa range is read; the other view's range
        is computed from the wide->objective Z offset, and both are filled in.
        """
        off_z = self._coord_mgr.offset_z if self._coord_mgr else 0.0
        obj_view = self._is_objective_view()

        if obj_view:
            obj_up, obj_lo = self._obj_z_upper.value(), self._obj_z_lower.value()
            wf_up, wf_lo = round(obj_up - off_z, 2), round(obj_lo - off_z, 2)
        else:
            wf_up, wf_lo = self._wf_z_upper.value(), self._wf_z_lower.value()
            obj_up, obj_lo = round(wf_up + off_z, 2), round(wf_lo + off_z, 2)

        if wf_lo >= wf_up:
            QMessageBox.warning(self, "Invalid Range",
                                "Prusa Z: highest must be greater than lowest.")
            return
        n_up, n_lo = self._nano_upper.value(), self._nano_lower.value()
        if n_lo >= n_up:
            QMessageBox.warning(self, "Invalid Range",
                                "NanoDrive: highest must be greater than lowest.")
            return

        # Fill in both views' boxes.
        self._wf_z_upper.setValue(wf_up)
        self._wf_z_lower.setValue(wf_lo)
        self._obj_z_upper.setValue(obj_up)
        self._obj_z_lower.setValue(obj_lo)

        logger.info(f"Registered autofocus ranges: WF Z [{wf_lo}, {wf_up}] mm, "
                    f"Obj Z [{obj_lo}, {obj_up}] mm (offset {off_z}), "
                    f"NanoDrive [{n_lo}, {n_up}] um")
        self._ranges_registered = True
        self.ranges_registered.emit()

    def _is_objective_view(self) -> bool:
        return (self._coord_mgr is not None
                and self._coord_mgr.current_view == CameraView.OBJECTIVE)

    def _update_af_editability(self):
        """Only the active view's Prusa boxes are editable; the other view's
        are shown disabled (they hold the offset-computed values)."""
        obj_view = self._is_objective_view()
        self._wf_z_upper.setEnabled(not obj_view)
        self._wf_z_lower.setEnabled(not obj_view)
        self._obj_z_upper.setEnabled(obj_view)
        self._obj_z_lower.setEnabled(obj_view)

    def on_camera_view_changed(self):
        """Called by the main window when the camera view switches."""
        self._update_af_editability()

    def is_registered(self) -> bool:
        """True once the user has pressed Register (autofocus Z ranges valid)."""
        return self._ranges_registered

    def get_widefield_z_limits(self) -> tuple:
        """Widefield Prusa Z autofocus limits (upper, lower) in mm."""
        return (self._wf_z_upper.value(), self._wf_z_lower.value())

    def set_move_handler(self, handler) -> None:
        """Route Go-button moves through ``handler(phys_x, phys_y, phys_z)``
        instead of a direct request_move_to (see MainWindow._manual_move)."""
        self._move_handler = handler

    def get_objective_z_limits(self) -> tuple:
        """Objective Prusa Z autofocus limits (upper, lower) in mm."""
        return (self._obj_z_upper.value(), self._obj_z_lower.value())

    def get_nanodrive_limits(self) -> tuple:
        """Get current NanoDrive fine-focus limits (upper, lower) in µm."""
        return (self._nano_upper.value(), self._nano_lower.value())

    def set_focus_result(self, kind: str, value: float):
        """Show the best-focus Z from an autofocus run next to Register.

        kind: 'cam_prusa' / 'spec_prusa' (Prusa Z, mm) or 'spec_nano'
        (NanoDrive, µm).
        """
        entry = self._focus_lbls.get(kind)
        if entry is None:
            return
        lbl, text = entry
        # Match the live XYZ / ND position readouts: 2 decimals for Prusa Z (mm),
        # 1 decimal for the NanoDrive (µm).
        if kind == 'spec_nano':
            lbl.setText(f'{text} Focus: {value:.1f} µm')
        else:
            lbl.setText(f'{text} Focus: {value:.2f} mm')
