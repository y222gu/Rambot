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
    QSpacerItem
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QFont

from raman_control.gui.threads.motion_thread import MotionThread
from raman_control.config.hardware_config import StageConfig
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
        layout.setSpacing(3)
        layout.setContentsMargins(0, 0, 0, 0)

        # Button style for direction buttons
        btn_style = '''
            QPushButton {
                background-color: #363636;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
                color: #e0e0e0;
            }
            QPushButton:hover { background-color: #4d4d4d; border-color: #00a0a0; }
            QPushButton:pressed { background-color: #2d2d2d; }
            QPushButton:disabled { color: #505050; background-color: #2a2a2a; }
        '''

        # Create 3x3 grid for each step size
        # Row 0: Y+ (large, medium, small)
        # Row 1: X-, center, X+
        # Row 2: Y-

        sizes = ['large', 'medium', 'small']
        arrows = {
            'up': '▲', 'down': '▼', 'left': '◀', 'right': '▶'
        }

        # Y+ buttons (top)
        for i, size in enumerate(sizes):
            btn = QPushButton(arrows['up'])
            btn.setFixedSize(36, 36)
            btn.setStyleSheet(btn_style)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, s=size: self._emit_move(0, self._step_sizes[s]))
            layout.addWidget(btn, i, 1)
            setattr(self, f'btn_y_plus_{size}', btn)

        # X- buttons (left)
        for i, size in enumerate(sizes):
            btn = QPushButton(arrows['left'])
            btn.setFixedSize(36, 36)
            btn.setStyleSheet(btn_style)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, s=size: self._emit_move(-self._step_sizes[s], 0))
            layout.addWidget(btn, 3, 2 - i)
            setattr(self, f'btn_x_minus_{size}', btn)

        # X+ buttons (right)
        for i, size in enumerate(sizes):
            btn = QPushButton(arrows['right'])
            btn.setFixedSize(36, 36)
            btn.setStyleSheet(btn_style)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, s=size: self._emit_move(self._step_sizes[s], 0))
            layout.addWidget(btn, 3, 4 + i)
            setattr(self, f'btn_x_plus_{size}', btn)

        # Y- buttons (bottom)
        for i, size in enumerate(sizes):
            btn = QPushButton(arrows['down'])
            btn.setFixedSize(36, 36)
            btn.setStyleSheet(btn_style)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, s=size: self._emit_move(0, -self._step_sizes[s]))
            layout.addWidget(btn, 4 + i, 1)
            setattr(self, f'btn_y_minus_{size}', btn)

        # Center indicator
        center = QLabel('XY')
        center.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.setStyleSheet('''
            background-color: #2d2d2d;
            border-radius: 4px;
            color: #808080;
            font-weight: bold;
        ''')
        center.setFixedSize(36, 36)
        layout.addWidget(center, 3, 1)

        # Step size labels
        for i, size in enumerate(sizes):
            label = QLabel(f'{self._step_sizes[size]}')
            label.setStyleSheet('color: #808080; font-size: 10px;')
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label, i, 2)

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

    def __init__(self, step_sizes: dict, label: str = 'Z', parent=None):
        super().__init__(parent)
        self._step_sizes = step_sizes
        self._label = label
        self._enabled = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(3)
        layout.setContentsMargins(0, 0, 0, 0)

        btn_style = '''
            QPushButton {
                background-color: #363636;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
                color: #e0e0e0;
            }
            QPushButton:hover { background-color: #4d4d4d; border-color: #00a0a0; }
            QPushButton:pressed { background-color: #2d2d2d; }
            QPushButton:disabled { color: #505050; background-color: #2a2a2a; }
        '''

        # Up label
        up_label = QLabel(f'{self._label} ▲')
        up_label.setStyleSheet('color: #808080; font-size: 11px;')
        up_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(up_label)

        # Up buttons (large to small)
        for size in ['large', 'medium', 'small']:
            btn = QPushButton('▲')
            btn.setFixedSize(44, 32)
            btn.setStyleSheet(btn_style)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, s=size: self.move_requested.emit(self._step_sizes[s]))
            layout.addWidget(btn)
            setattr(self, f'btn_up_{size}', btn)

        # Spacer
        layout.addSpacing(8)

        # Down buttons (small to large)
        for size in ['small', 'medium', 'large']:
            btn = QPushButton('▼')
            btn.setFixedSize(44, 32)
            btn.setStyleSheet(btn_style)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, s=size: self.move_requested.emit(-self._step_sizes[s]))
            layout.addWidget(btn)
            setattr(self, f'btn_down_{size}', btn)

        # Down label
        down_label = QLabel(f'{self._label} ▼')
        down_label.setStyleSheet('color: #808080; font-size: 11px;')
        down_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(down_label)

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
        autofocus_requested: Emitted when autofocus is requested (type, z_upper, z_lower)
    """

    position_updated = pyqtSignal(float, float, float)
    referenced_changed = pyqtSignal(dict)
    autofocus_requested = pyqtSignal(str, float, float)  # type, z_upper, z_lower

    def __init__(
        self,
        motion_thread: MotionThread,
        config: Optional[StageConfig] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self._motion = motion_thread
        self._config = config or StageConfig()

        # Current state
        self._current_position = (0.0, 0.0, 0.0)
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
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)

        # === Reference Section ===
        ref_group = QGroupBox("Reference & Home")
        ref_layout = QGridLayout(ref_group)
        ref_layout.setSpacing(8)

        # Reference buttons with status indicators
        ref_buttons = [
            ('X', 0, 0), ('Y', 0, 1), ('Z', 0, 2), ('NP', 0, 3)
        ]

        self._ref_indicators = {}
        self._ref_buttons = {}

        for name, row, col in ref_buttons:
            # Container for button + indicator
            container = QHBoxLayout()
            container.setSpacing(4)

            indicator = StatusIndicator()
            self._ref_indicators[name] = indicator

            btn = QPushButton(f"Ref {name}")
            btn.setFixedWidth(70)
            btn.clicked.connect(lambda _, n=name: self._on_ref_clicked(n))
            self._ref_buttons[name] = btn

            container.addWidget(indicator)
            container.addWidget(btn)
            ref_layout.addLayout(container, row, col)

        # Ref All button
        self._ref_all_btn = QPushButton("Reference All")
        self._ref_all_btn.setProperty('primary', True)
        self._ref_all_btn.clicked.connect(self._on_ref_all_clicked)
        ref_layout.addWidget(self._ref_all_btn, 1, 0, 1, 2)

        # Home button
        self._home_btn = QPushButton("⌂ Go Home")
        self._home_btn.setEnabled(False)
        self._home_btn.clicked.connect(self._on_home_clicked)
        ref_layout.addWidget(self._home_btn, 1, 2, 1, 2)

        main_layout.addWidget(ref_group)

        # === Position Display ===
        pos_group = QGroupBox("Position")
        pos_layout = QGridLayout(pos_group)
        pos_layout.setSpacing(8)

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
            label = QLabel("—")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet('''
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 6px;
                font-family: "SF Mono", "Consolas", monospace;
                font-size: 14px;
            ''')
            self._pos_labels[axis] = label
            pos_layout.addWidget(label, 1, col + 1)

        # Target position
        pos_layout.addWidget(QLabel("Target:"), 2, 0)

        self._target_inputs = {}
        for col, axis in enumerate(['X', 'Y', 'Z']):
            spin = QDoubleSpinBox()
            spin.setRange(-500, 500)
            spin.setDecimals(2)
            spin.setSingleStep(1.0)
            spin.setValue(getattr(self._config, f'home_{axis.lower()}', 100.0))
            self._target_inputs[axis] = spin
            pos_layout.addWidget(spin, 2, col + 1)

        # NanoDrive target
        self._nd_target = QDoubleSpinBox()
        self._nd_target.setRange(0, 100)
        self._nd_target.setDecimals(1)
        self._nd_target.setValue(50.0)
        self._nd_target.setSuffix(' μm')
        pos_layout.addWidget(self._nd_target, 2, 4)

        # Go button
        self._go_btn = QPushButton("Go to Target")
        self._go_btn.setProperty('primary', True)
        self._go_btn.setEnabled(False)
        self._go_btn.clicked.connect(self._on_go_clicked)
        pos_layout.addWidget(self._go_btn, 3, 1, 1, 4)

        main_layout.addWidget(pos_group)

        # === Speed Control ===
        speed_group = QGroupBox("Movement Speed")
        speed_layout = QHBoxLayout(speed_group)

        speed_layout.addWidget(QLabel("Slow"))

        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(10, 3000)
        self._speed_slider.setValue(self._config.default_speed)
        self._speed_slider.valueChanged.connect(self._on_speed_changed)
        speed_layout.addWidget(self._speed_slider, 1)

        speed_layout.addWidget(QLabel("Fast"))

        self._speed_label = QLabel(f"{self._config.default_speed} mm/min")
        self._speed_label.setFixedWidth(90)
        self._speed_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._speed_label.setStyleSheet('color: #00b4b4;')
        speed_layout.addWidget(self._speed_label)

        main_layout.addWidget(speed_group)

        # === Manual Movement ===
        move_group = QGroupBox("Manual Movement")
        move_layout = QHBoxLayout(move_group)
        move_layout.setSpacing(20)

        # XY Pad
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

        # Spacer
        move_layout.addSpacing(20)

        # Z Pad
        z_steps = {
            'small': self._config.small_step_z,
            'medium': self._config.medium_step_z,
            'large': self._config.large_step_z
        }
        self._z_pad = ZControlPad(z_steps, 'Z')
        self._z_pad.move_requested.connect(
            lambda dz: self._motion.request_move_by(z=dz)
        )
        move_layout.addWidget(self._z_pad)

        # NanoDrive Pad
        nd_steps = {'small': 0.5, 'medium': 1.0, 'large': 5.0}
        self._nd_pad = ZControlPad(nd_steps, 'ND')
        # TODO: Connect to nanodrive
        move_layout.addWidget(self._nd_pad)

        move_layout.addStretch()

        main_layout.addWidget(move_group)

        # === Autofocus Section ===
        af_group = QGroupBox("Autofocus")
        af_layout = QVBoxLayout(af_group)
        af_layout.setSpacing(6)

        # Z limits row
        limits_row = QHBoxLayout()
        limits_row.addWidget(QLabel("Z Upper:"))
        self._z_upper = QDoubleSpinBox()
        self._z_upper.setRange(0, 100)
        self._z_upper.setValue(40.0)
        self._z_upper.setSuffix(" mm")
        self._z_upper.setDecimals(1)
        limits_row.addWidget(self._z_upper)
        limits_row.addWidget(QLabel("Lower:"))
        self._z_lower = QDoubleSpinBox()
        self._z_lower.setRange(0, 100)
        self._z_lower.setValue(30.0)
        self._z_lower.setSuffix(" mm")
        self._z_lower.setDecimals(1)
        limits_row.addWidget(self._z_lower)
        af_layout.addLayout(limits_row)

        # Set from current buttons
        set_row = QHBoxLayout()
        self._set_upper_btn = QPushButton("Set Upper")
        self._set_upper_btn.setFixedHeight(24)
        self._set_upper_btn.clicked.connect(lambda: self._z_upper.setValue(self._current_position[2]))
        set_row.addWidget(self._set_upper_btn)
        self._set_lower_btn = QPushButton("Set Lower")
        self._set_lower_btn.setFixedHeight(24)
        self._set_lower_btn.clicked.connect(lambda: self._z_lower.setValue(self._current_position[2]))
        set_row.addWidget(self._set_lower_btn)
        af_layout.addLayout(set_row)

        # Run buttons row
        run_row = QHBoxLayout()
        self._coarse_af_btn = QPushButton("Coarse AF")
        self._coarse_af_btn.setEnabled(False)
        self._coarse_af_btn.clicked.connect(self._on_coarse_autofocus)
        run_row.addWidget(self._coarse_af_btn)
        self._fine_af_btn = QPushButton("Fine AF")
        self._fine_af_btn.setEnabled(False)
        self._fine_af_btn.clicked.connect(self._on_fine_autofocus)
        run_row.addWidget(self._fine_af_btn)
        af_layout.addLayout(run_row)

        main_layout.addWidget(af_group)

        # === Emergency Stop ===
        self._stop_btn = QPushButton("⚠ EMERGENCY STOP")
        self._stop_btn.setProperty('danger', True)
        self._stop_btn.setMinimumHeight(40)
        self._stop_btn.setStyleSheet('''
            QPushButton {
                background-color: #c62828;
                border: 2px solid #f44336;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover { background-color: #d32f2f; }
            QPushButton:pressed { background-color: #b71c1c; }
        ''')
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        main_layout.addWidget(self._stop_btn)

    def _connect_signals(self):
        """Connect motion thread signals."""
        self._motion.position_changed.connect(self._on_position_changed)
        self._motion.axis_referenced.connect(self._on_axis_referenced)
        self._motion.movement_complete.connect(self._on_movement_complete)
        self._motion.connected.connect(self._on_connected)
        self._motion.disconnected.connect(self._on_disconnected)
        self._motion.error.connect(self._on_error)

    def _on_position_changed(self, x: float, y: float, z: float):
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
        self._update_button_states()

    def _on_error(self, message: str):
        logger.error(f"Motion error: {message}")
        # Don't show dialog for "Stage not connected" - this is expected
        # when hardware isn't present. Use status indicator instead.
        if "not connected" not in message.lower():
            QMessageBox.warning(self, "Motion Error", message)

    def _on_ref_clicked(self, axis: str):
        if not self._motion.is_connected:
            self._show_not_connected_warning("Stage")
            return
        if axis == 'NP':
            # TODO: Reference nanodrive
            pass
        else:
            self._motion.request_home(axis)

    def _on_ref_all_clicked(self):
        if not self._motion.is_connected:
            self._show_not_connected_warning("Stage")
            return
        reply = QMessageBox.question(
            self,
            "Reference All Axes",
            "This will home all axes (X, Y, Z). The stage will move.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._motion.request_home('XYZ')

    def _show_not_connected_warning(self, device: str):
        """Show a brief warning that device is not connected."""
        logger.warning(f"{device} not connected - action blocked")
        # Use a non-blocking approach - just update the status indicator
        # The user can see the connection status in the UI

    def _on_home_clicked(self):
        if not self._motion.is_connected:
            self._show_not_connected_warning("Stage")
            return
        if self._motion.all_referenced:
            self._motion.request_go_home()

    def _on_go_clicked(self):
        if not self._motion.is_connected:
            self._show_not_connected_warning("Stage")
            return
        try:
            x = self._target_inputs['X'].value()
            y = self._target_inputs['Y'].value()
            z = self._target_inputs['Z'].value()
            self._motion.request_move_to(x=x, y=y, z=z)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _on_speed_changed(self, value: int):
        self._motion.speed = value
        self._speed_label.setText(f"{value} mm/min")

    def _on_stop_clicked(self):
        self._motion.request_stop()
        logger.warning("Emergency stop triggered by user")

    def _request_position_update(self):
        if self._motion.is_connected:
            self._motion.request_position_update()

    def _update_button_states(self):
        all_ref = all(self._referenced.values())
        connected = self._motion.is_connected

        # Reference buttons only enabled when connected
        for btn in self._ref_buttons.values():
            btn.setEnabled(connected)
        self._ref_all_btn.setEnabled(connected)

        self._home_btn.setEnabled(all_ref and connected)
        self._go_btn.setEnabled(all_ref and connected)

        self._xy_pad.set_enabled(all_ref and connected)
        self._z_pad.set_enabled(all_ref and connected)
        self._nd_pad.set_enabled(connected)  # ND doesn't need stage ref

        # Autofocus buttons
        self._coarse_af_btn.setEnabled(all_ref and connected)
        self._fine_af_btn.setEnabled(connected)

    def _on_coarse_autofocus(self):
        """Run coarse autofocus using PRUSA stepper."""
        z_upper = self._z_upper.value()
        z_lower = self._z_lower.value()
        if z_lower >= z_upper:
            QMessageBox.warning(self, "Invalid Range", "Upper limit must be greater than lower limit")
            return
        self.autofocus_requested.emit('coarse', z_upper, z_lower)

    def _on_fine_autofocus(self):
        """Run fine autofocus using NanoDrive piezo."""
        z_upper = self._z_upper.value()
        z_lower = self._z_lower.value()
        self.autofocus_requested.emit('fine', z_upper, z_lower)

    def get_z_limits(self) -> tuple:
        """Get current Z limits."""
        return (self._z_upper.value(), self._z_lower.value())

    def set_z_limits(self, upper: float, lower: float):
        """Set Z limits."""
        self._z_upper.setValue(upper)
        self._z_lower.setValue(lower)
