"""
Hardware settings dialog for viewing and editing runtime configuration.

Provides a tabbed interface matching the HardwareConfig hierarchy with
appropriate input widgets for each field type.
"""

from dataclasses import fields as dc_fields
from typing import Any, Dict, List, Tuple

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QFormLayout, QGroupBox, QScrollArea, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QCheckBox, QLineEdit, QFileDialog,
    QDialogButtonBox, QSizePolicy,
)
from PyQt6.QtCore import Qt

from raman_control.config.hardware_config import (
    HardwareConfig, StageConfig, SerialConfig, CameraConfig,
    NanoDriveConfig, SpectrometerConfig, SampleDropConfig,
)
from raman_control.config.logging_config import get_logger
from raman_control.gui.styles import COLORS

logger = get_logger('settings_dialog')

# Mapping of field names to human-readable labels and tooltips.
# Fields not listed here get an auto-generated label from the field name.
_FIELD_META: Dict[str, Tuple[str, str]] = {
    # Stage
    'small_step_xy': ("Small step XY (mm)", ""),
    'medium_step_xy': ("Medium step XY (mm)", ""),
    'large_step_xy': ("Large step XY (mm)", ""),
    'small_step_z': ("Small step Z (mm)", ""),
    'medium_step_z': ("Medium step Z (mm)", ""),
    'large_step_z': ("Large step Z (mm)", ""),
    'default_speed': ("Default speed (mm/min)", ""),
    'homing_speed': ("Homing speed (mm/min)", ""),
    'safety_height': ("Safety height (mm)", "Z height for safe lateral moves"),
    'home_x': ("Home X (mm)", ""),
    'home_y': ("Home Y (mm)", ""),
    'home_z': ("Home Z (mm)", ""),
    'calib_wide_to_obj_x': ("Wide-to-Obj offset X (mm)", "Calibration offset"),
    'calib_wide_to_obj_y': ("Wide-to-Obj offset Y (mm)", "Calibration offset"),
    'calib_wide_to_obj_z': ("Wide-to-Obj offset Z (mm)", "Calibration offset"),
    'movement_stabilization_time': ("Stabilization time (s)", "Wait after movement"),
    'keepalive_interval': ("Keep-alive interval (s)", ""),
    # Serial
    'baudrate': ("Baud rate", ""),
    'timeout': ("Read timeout (s)", ""),
    'write_timeout': ("Write timeout (s)", ""),
    'command_delay': ("Command delay (s)", "Delay between serial commands"),
    'max_retries': ("Max retries", ""),
    'ack_timeout': ("ACK timeout (s)", "Time to wait for acknowledgment"),
    # Camera
    'serial_number': ("Serial number", "Camera hardware serial"),
    'exposure_time': ("Exposure time (s)", ""),
    'gain': ("Gain", ""),
    'frame_rate': ("Frame rate (Hz)", ""),
    'image_poll_timeout_ms': ("Image poll timeout (ms)", "0 = non-blocking"),
    'pixel_size_x_um': ("Pixel size X (um)", "Micrometers per pixel"),
    'pixel_size_y_um': ("Pixel size Y (um)", "Micrometers per pixel"),
    'flip_horizontal': ("Flip horizontal", ""),
    'rotate_minus90': ("Rotate -90 deg", ""),
    'crop_enabled': ("Crop enabled", ""),
    'crop_center_x': ("Crop center X (px)", ""),
    'crop_center_y': ("Crop center Y (px)", ""),
    'crop_width': ("Crop width (px)", ""),
    'crop_height': ("Crop height (px)", ""),
    'center_max_repeats': ("Auto-center max repeats", "Max re-center attempts for the Center button"),
    # NanoDrive
    'dll_path': ("DLL path", "Path to Madlib.dll"),
    'axis': ("Axis", "Piezo axis number"),
    'initial_position': ("Initial position (um)", ""),
    'step_size': ("Step size (um)", ""),
    'min_position': ("Min position (um)", ""),
    'max_position': ("Max position (um)", "Hardware safe max"),
    'stabilization_time': ("Stabilization time (s)", "Settling time after move"),
    # Spectrometer
    'integration_time': ("Integration time (ms)", ""),
    'laser_power': ("Laser power (%)", ""),
    'laser_power_mw': ("Laser power (mW)", ""),
    'averaging': ("Averaging", "Number of spectra to average"),
    # Stage safety limits
    'x_min': ("X min limit (mm)", "Minimum stage X position"),
    'x_max': ("X max limit (mm)", "Maximum stage X position"),
    'y_min': ("Y min limit (mm)", "Minimum stage Y position"),
    'y_max': ("Y max limit (mm)", "Maximum stage Y position"),
    'z_min': ("Z min limit (mm)", "Minimum stage Z position"),
    'z_max': ("Z max limit (mm)", "Maximum stage Z position"),
    'z_safe': ("Z safe height (mm)", "Safe Z for camera switching"),
    'af_z_upper': ("AF Z upper (mm)", "Upper autofocus search limit"),
    'af_z_lower': ("AF Z lower (mm)", "Lower autofocus search limit"),
}

# Which float fields need high precision
_HIGH_PRECISION_FIELDS = {
    'pixel_size_x_um', 'pixel_size_y_um', 'stabilization_time',
    'command_delay', 'movement_stabilization_time', 'step_size',
}


def _label_for(name: str) -> str:
    """Return a human-readable label for a field name."""
    if name in _FIELD_META:
        return _FIELD_META[name][0]
    return name.replace('_', ' ').capitalize()


def _tooltip_for(name: str) -> str:
    if name in _FIELD_META:
        return _FIELD_META[name][1]
    return ""


class SettingsDialog(QDialog):
    """Modal dialog for editing HardwareConfig at runtime."""

    def __init__(
        self,
        config: HardwareConfig,
        parent=None,
    ):
        super().__init__(parent)
        self._config = config

        # Stores (dotted_key, widget) pairs so we can read values back
        self._widgets: List[Tuple[str, Any]] = []

        self._init_ui()
        self._populate_from_config()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        self.setWindowTitle("Hardware Settings")
        self.setMinimumSize(560, 620)
        self.resize(620, 720)

        root = QVBoxLayout(self)
        root.setSpacing(8)

        # Note label
        note = QLabel(
            "Some changes (serial numbers, DLL paths) require reconnecting devices to take effect."
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            f"color: {COLORS['warning']}; font-size: 11px; padding: 4px 8px;"
        )
        root.addWidget(note)

        # Tab widget
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, stretch=1)

        self._build_stage_tab()
        self._build_cameras_tab()
        self._build_nanodrive_tab()
        self._build_spectrometer_tab()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self._load_btn = QPushButton("Load from File")
        self._save_btn = QPushButton("Save to File")
        self._load_btn.clicked.connect(self._on_load)
        self._save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(self._load_btn)
        btn_layout.addWidget(self._save_btn)

        btn_layout.addStretch()

        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setProperty("primary", True)
        self._apply_btn.clicked.connect(self._on_apply)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(self._apply_btn)
        btn_layout.addWidget(self._cancel_btn)

        root.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_stage_tab(self):
        tab = self._make_scroll_tab()
        layout = tab.layout()

        # Serial group
        serial_group = self._make_group("Serial Communication")
        serial_form = QFormLayout()
        serial_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._add_dataclass_fields(serial_form, self._config.stage.serial, "stage.serial")
        serial_group.setLayout(serial_form)
        layout.addWidget(serial_group)

        # Home position
        home_group = self._make_group("Home Position")
        home_form = QFormLayout()
        home_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for name in ('home_x', 'home_y', 'home_z'):
            self._add_field(home_form, name, getattr(self._config.stage, name), f"stage.{name}")
        home_group.setLayout(home_form)
        layout.addWidget(home_group)

        # Calibration offsets
        calib_group = self._make_group("Calibration Offsets (Wide FOV -> Objective)")
        calib_form = QFormLayout()
        calib_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for name in ('calib_wide_to_obj_x', 'calib_wide_to_obj_y', 'calib_wide_to_obj_z'):
            self._add_field(calib_form, name, getattr(self._config.stage, name), f"stage.{name}")
        calib_group.setLayout(calib_form)
        layout.addWidget(calib_group)

        # Step sizes
        step_group = self._make_group("Step Sizes")
        step_form = QFormLayout()
        step_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for name in ('small_step_xy', 'medium_step_xy', 'large_step_xy',
                      'small_step_z', 'medium_step_z', 'large_step_z'):
            self._add_field(step_form, name, getattr(self._config.stage, name), f"stage.{name}")
        step_group.setLayout(step_form)
        layout.addWidget(step_group)

        # Speed / timing
        misc_group = self._make_group("Speed && Timing")
        misc_form = QFormLayout()
        misc_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for name in ('default_speed', 'homing_speed', 'safety_height',
                      'movement_stabilization_time', 'keepalive_interval'):
            self._add_field(misc_form, name, getattr(self._config.stage, name), f"stage.{name}")
        misc_group.setLayout(misc_form)
        layout.addWidget(misc_group)

        # Safety limits
        safety_group = self._make_group("Safety Limits")
        safety_form = QFormLayout()
        safety_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for name in ('x_min', 'x_max', 'y_min', 'y_max',
                      'z_min', 'z_max', 'z_safe',
                      'af_z_upper', 'af_z_lower'):
            self._add_field(safety_form, name, getattr(self._config.stage, name), f"stage.{name}")
        safety_group.setLayout(safety_form)
        layout.addWidget(safety_group)

        layout.addStretch()
        self._tabs.addTab(tab, "Stage")

    def _build_cameras_tab(self):
        tab = self._make_scroll_tab()
        layout = tab.layout()

        for section_label, cfg, prefix in [
            ("Wide FOV Camera", self._config.widefield_camera, "widefield_camera"),
            ("Objective Camera", self._config.objective_camera, "objective_camera"),
        ]:
            group = self._make_group(section_label)
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            # Auto-center repeats is a widefield-only behavior, so only show
            # it under the Wide FOV camera.
            exclude = None if prefix == "widefield_camera" else {"center_max_repeats"}
            self._add_dataclass_fields(form, cfg, prefix, exclude=exclude)
            group.setLayout(form)
            layout.addWidget(group)

        layout.addStretch()
        self._tabs.addTab(tab, "Cameras")

    def _build_nanodrive_tab(self):
        tab = self._make_scroll_tab()
        layout = tab.layout()

        group = self._make_group("NanoDrive (MadCity Labs)")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._add_dataclass_fields(form, self._config.nanodrive, "nanodrive")
        group.setLayout(form)
        layout.addWidget(group)

        layout.addStretch()
        self._tabs.addTab(tab, "NanoDrive")

    def _build_spectrometer_tab(self):
        tab = self._make_scroll_tab()
        layout = tab.layout()

        group = self._make_group("Spectrometer (Wasatch Photonics)")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._add_dataclass_fields(form, self._config.spectrometer, "spectrometer")
        group.setLayout(form)
        layout.addWidget(group)

        layout.addStretch()
        self._tabs.addTab(tab, "Spectrometer")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_scroll_tab(self) -> QWidget:
        """Create a scrollable tab page with a vertical layout."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(8, 8, 8, 8)
        inner_layout.setSpacing(10)
        scroll.setWidget(inner)
        # Attach inner reference so callers can do tab.layout()
        scroll._inner = inner
        scroll.layout = inner.layout  # type: ignore[assignment]
        return scroll

    @staticmethod
    def _make_group(title: str) -> QGroupBox:
        group = QGroupBox(title)
        return group

    def _add_dataclass_fields(self, form: QFormLayout, dc_instance, prefix: str,
                              exclude=None):
        """Add all fields of a dataclass instance to a form layout."""
        exclude = exclude or set()
        for f in dc_fields(type(dc_instance)):
            if f.name in exclude:
                continue
            # Skip nested dataclass fields (handled separately for stage.serial)
            val = getattr(dc_instance, f.name)
            if hasattr(val, '__dataclass_fields__'):
                continue
            self._add_field(form, f.name, val, f"{prefix}.{f.name}")

    def _add_field(self, form: QFormLayout, name: str, value: Any, dotted_key: str):
        """Add a single field row to a form layout and register the widget."""
        label_text = _label_for(name)
        tooltip = _tooltip_for(name)
        widget: QWidget

        if isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
            if tooltip:
                widget.setToolTip(tooltip)
        elif isinstance(value, float):
            widget = QDoubleSpinBox()
            decimals = 3 if name in _HIGH_PRECISION_FIELDS else 2
            widget.setDecimals(decimals)
            widget.setRange(-99999.0, 99999.0)
            widget.setSingleStep(0.01 if decimals == 3 else 0.1)
            widget.setValue(value)
            if tooltip:
                widget.setToolTip(tooltip)
        elif isinstance(value, int):
            widget = QSpinBox()
            widget.setRange(-999999, 999999)
            widget.setValue(value)
            if tooltip:
                widget.setToolTip(tooltip)
        elif isinstance(value, str):
            widget = QLineEdit(value)
            if tooltip:
                widget.setToolTip(tooltip)
        else:
            # Fallback: read-only string representation (e.g. tuple)
            widget = QLineEdit(str(value))
            widget.setToolTip("Read-only (complex type)")
            widget.setReadOnly(True)

        widget.setMinimumWidth(180)
        form.addRow(label_text, widget)
        self._widgets.append((dotted_key, widget))

    # ------------------------------------------------------------------
    # Read widget values
    # ------------------------------------------------------------------

    def _read_all_widgets(self) -> Dict[str, Any]:
        """Read current values from all widgets as a dotted-key dict."""
        result: Dict[str, Any] = {}
        for dotted_key, widget in self._widgets:
            if isinstance(widget, QCheckBox):
                result[dotted_key] = widget.isChecked()
            elif isinstance(widget, QDoubleSpinBox):
                result[dotted_key] = widget.value()
            elif isinstance(widget, QSpinBox):
                result[dotted_key] = widget.value()
            elif isinstance(widget, QLineEdit):
                if widget.isReadOnly():
                    continue  # skip complex types
                result[dotted_key] = widget.text()
        return result

    # ------------------------------------------------------------------
    # Populate widgets from config
    # ------------------------------------------------------------------

    def _populate_from_config(self):
        """Set widget values from the current config object."""
        for dotted_key, widget in self._widgets:
            parts = dotted_key.split('.')
            obj: Any = self._config
            try:
                for part in parts:
                    obj = getattr(obj, part)
            except AttributeError:
                continue

            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(obj))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(obj))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(obj))
            elif isinstance(widget, QLineEdit) and not widget.isReadOnly():
                widget.setText(str(obj))

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_apply(self):
        """Write widget values back into the live config.

        CoordinateManager reads directly from StageConfig, so no
        manual sync is needed — changes take effect immediately.
        """
        values = self._read_all_widgets()
        self._config.update_from_dict(values)
        logger.info("Settings applied to live config")
        self.accept()

    def _on_save(self):
        """Save config to a JSON file (values are read from widgets first)."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Hardware Config", "hardware_config.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not filepath:
            return

        # Apply widget values to config before saving
        values = self._read_all_widgets()
        self._config.update_from_dict(values)
        self._config.save_to_file(filepath)
        logger.info(f"Config saved to {filepath}")

    def _on_load(self):
        """Load config from a JSON file and repopulate widgets."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Hardware Config", "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not filepath:
            return

        try:
            loaded = HardwareConfig.load_from_file(filepath)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return

        # Copy loaded values into the live config object's sub-objects
        # so the same object references stay valid for threads/widgets.
        for f in dc_fields(HardwareConfig):
            setattr(self._config, f.name, getattr(loaded, f.name))

        self._populate_from_config()
        logger.info(f"Config loaded from {filepath}")
