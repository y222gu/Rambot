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
    QDialogButtonBox, QSizePolicy, QComboBox,
)
from PyQt6.QtCore import Qt

from raman_control.config.hardware_config import (
    HardwareConfig, StageConfig, SerialConfig, CameraConfig,
    NanoDriveConfig, SpectrometerConfig, SampleDropConfig, KymeraConfig,
)
from raman_control.hardware.spectrometer_base import BACKEND_CHOICES, BACKEND_LABELS
from raman_control.config.logging_config import get_logger
from raman_control.config.offset_calibration import get_offset_calibration
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
    'use_offset_calibration_curve': (
        "Calibration curve",
        "Checked: the Y offset is read from the linear fit of "
        "config/offset_calibration.csv at the current widefield X "
        "(the offset drifts along X).\n"
        "Unchecked: the fixed Wide-to-Obj offset Y above is used everywhere.",
    ),
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
    'backend': (
        "Spectrometer",
        "Which spectrometer collects the Raman signal. Auto tries the Wasatch "
        "Photonics unit first and falls back to the Andor Kymera + Newton if no "
        "Wasatch is detected. The excitation laser is always the Wasatch laser. "
        "Applies at once: the connected spectrometer is released and the "
        "selected one connected; if it is absent the spectrum panel shows "
        "'No spectrometer detected' and selecting a unit here again (or "
        "Connect) retries."),
    'integration_time': ("Integration time (ms)", ""),
    'laser_power': ("Laser power (%)", ""),
    'laser_power_mw': ("Laser power (mW)", ""),
    'averaging': ("Averaging", "Number of spectra to average"),
    'excitation_wavelength_nm': ("Excitation wavelength (nm)",
                                 "Laser line used to convert nm to Raman shift"),
    'af_metric': (
        "Focus metric",
        "Score the spectrometer autofocus minimises. Ratio: signal band / "
        "reference band (bands below). Entropy: Shannon entropy of the entropy "
        "band (sharp peaks = low entropy), the method of the Kymera notebooks. "
        "Auto: ratio on the Wasatch, entropy on the Kymera."),
    'af_entropy_min_nm': ("Entropy band min (nm)",
                          "Lower edge of the band the entropy metric is computed over"),
    'af_entropy_max_nm': ("Entropy band max (nm)",
                          "Upper edge of the band the entropy metric is computed over"),
    # Kymera / Newton
    'grating': ("Grating", "Kymera turret position (1-based), lab default 3"),
    'center_wavelength_nm': ("Center wavelength (nm)",
                             "Grating centre wavelength, lab default 853 nm"),
    'cooler_on': ("Cooler on", "Start the Newton CCD cooler on connect"),
    'target_temperature_c': ("Target temperature (C)", "CCD set point, lab default -60"),
    'keep_cooler_on_shutdown': (
        "Keep cooler on at shutdown",
        "Keep the CCD at its set point when the camera is shut down instead "
        "of letting it warm uncontrolled"),
    'bin_row_start': (
        "Bin rows: first row",
        "First detector row (1-based, inclusive) summed into the spectrum. "
        "Newton DU920P: 256 rows, 1 = top. Wasatch: the unit's vertical ROI, "
        "0 = the factory ROI from its EEPROM; only changeable on firmware "
        "that supports a settable vertical ROI (Series-XS, or microcontroller "
        ">= 10.0.0.47), otherwise ignored. Use the Spectrograph switch on the "
        "spectrum plot to see where the signal lands. Applies live."),
    'bin_row_end': (
        "Bin rows: last row",
        "Last detector row (1-based, inclusive) summed into the spectrum. "
        "Newton: 0 = the last row (full vertical binning, the default). "
        "Wasatch: 0 = the factory ROI. Applies live."),
    'acquisition_timeout_margin_s': (
        "Acquisition timeout margin (s)",
        "One exposure is abandoned after 2 x exposure + this margin"),
    'sdk_dir': ("Andor SDK DLL folder",
                "Optional folder with atmcd64d.dll / atspectrograph.dll. Empty = "
                "the DLLs bundled with the pyAndorSDK2 packages"),
    'af_prusa_signal_min_nm': (
        "Signal band min (nm)",
        "Lower edge of the SIGNAL band (numerator) for the Prusa coarse "
        "autofocus. Default 808 nm (quartz peak)."),
    'af_prusa_signal_max_nm': (
        "Signal band max (nm)",
        "Upper edge of the SIGNAL band (numerator) for the Prusa coarse "
        "autofocus. Default 815 nm (quartz peak)."),
    'af_prusa_reference_min_nm': (
        "Reference band min (nm)",
        "Lower edge of the REFERENCE band (denominator) for the Prusa coarse "
        "autofocus. Default 790 nm (reflected laser line)."),
    'af_prusa_reference_max_nm': (
        "Reference band max (nm)",
        "Upper edge of the REFERENCE band (denominator) for the Prusa coarse "
        "autofocus. Default 795 nm (reflected laser line)."),
    'af_nano_signal_min_nm': (
        "Signal band min (nm)",
        "Lower edge of the SIGNAL band (numerator) for the NanoDrive fine "
        "autofocus. Default 808 nm (quartz peak)."),
    'af_nano_signal_max_nm': (
        "Signal band max (nm)",
        "Upper edge of the SIGNAL band (numerator) for the NanoDrive fine "
        "autofocus. Default 815 nm (quartz peak)."),
    'af_nano_reference_min_nm': (
        "Reference band min (nm)",
        "Lower edge of the REFERENCE band (denominator) for the NanoDrive fine "
        "autofocus. Default 790 nm (reflected laser line)."),
    'af_nano_reference_max_nm': (
        "Reference band max (nm)",
        "Upper edge of the REFERENCE band (denominator) for the NanoDrive fine "
        "autofocus. Default 795 nm (reflected laser line)."),
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

# Fields that must NOT get the default +/-99999 spin-box range. The
# spectrometer exposure has no software cap: long integrations are a valid way
# to pull signal out of a weak sample, so only the driver/hardware limits it.
_UNCAPPED_FIELDS = {'integration_time'}

# Spectrometer autofocus focus-score bands. Pulled out of the generic
# Spectrometer group so they can be shown together under their own heading.
_AF_BAND_FIELDS = {
    'af_prusa_signal_min_nm', 'af_prusa_signal_max_nm',
    'af_prusa_reference_min_nm', 'af_prusa_reference_max_nm',
    'af_nano_signal_min_nm', 'af_nano_signal_max_nm',
    'af_nano_reference_min_nm', 'af_nano_reference_max_nm',
    'af_metric', 'af_entropy_min_nm', 'af_entropy_max_nm',
}

# String fields edited with a drop-down instead of a free-text box:
# field name -> list of (stored value, label).
_COMBO_FIELDS: Dict[str, List[Tuple[str, str]]] = {
    'backend': [(b, BACKEND_LABELS[b]) for b in BACKEND_CHOICES],
    'af_metric': [
        ('auto', "Auto (ratio on Wasatch, entropy on Kymera)"),
        ('ratio', "Band ratio (signal / reference)"),
        ('entropy', "Spectral entropy"),
    ],
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
        calib_widgets = {}
        for name in ('calib_wide_to_obj_x', 'calib_wide_to_obj_y', 'calib_wide_to_obj_z',
                     'use_offset_calibration_curve'):
            calib_widgets[name] = self._add_field(
                calib_form, name, getattr(self._config.stage, name), f"stage.{name}")

        # The Y offset box is only used when the calibration curve is OFF;
        # with the curve on, Y is computed from the CSV fit at the current
        # widefield X, so the manual entry is greyed out to make that obvious.
        self._calib_y_spin = calib_widgets['calib_wide_to_obj_y']
        self._calib_y_label = calib_form.labelForField(self._calib_y_spin)
        self._calib_curve_check = calib_widgets['use_offset_calibration_curve']
        self._calib_curve_check.toggled.connect(self._on_calib_curve_toggled)

        # Status line under the checkbox: what the fitted curve looks like, so
        # the user can tell at a glance whether the CSV was picked up.
        calib = get_offset_calibration()
        if calib.is_valid:
            curve_text = f"Curve: {calib.describe()}"
            curve_color = COLORS.get('text_secondary', '#a0a0a0')
        else:
            curve_text = f"Curve unavailable: {calib.error}"
            curve_color = COLORS['warning']
        curve_label = QLabel(curve_text)
        curve_label.setWordWrap(True)
        curve_label.setStyleSheet(f"color: {curve_color}; font-size: 11px;")
        curve_label.setToolTip(f"Source: {calib.filepath}")
        calib_form.addRow("", curve_label)

        self._on_calib_curve_toggled(self._calib_curve_check.isChecked())

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

    def _on_calib_curve_toggled(self, checked: bool):
        """Grey out the manual Y offset while the calibration curve is on.

        With the curve enabled the Y offset is computed from the fit at the
        current widefield X, so the typed value has no effect. The value is
        still read back and kept in the config, so unchecking the box
        restores whatever was last entered.
        """
        self._calib_y_spin.setEnabled(not checked)
        if self._calib_y_label is not None:
            self._calib_y_label.setEnabled(not checked)

        if checked:
            self._calib_y_spin.setToolTip(
                "Ignored while 'Calibration curve' is checked - the Y offset "
                "is computed from config/offset_calibration.csv at the current "
                "widefield X. Uncheck to use this fixed value."
            )
        else:
            self._calib_y_spin.setToolTip(_tooltip_for('calib_wide_to_obj_y'))

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

        group = self._make_group("Spectrometer")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        # Which instrument collects the spectra (drop-down), then the shared
        # acquisition / laser settings.
        self._add_field(form, 'backend', self._config.spectrometer.backend,
                        "spectrometer.backend")
        self._add_dataclass_fields(form, self._config.spectrometer, "spectrometer",
                                   exclude=_AF_BAND_FIELDS | {'backend'})
        group.setLayout(form)
        layout.addWidget(group)

        laser_note = QLabel(
            "The excitation laser is always the one on the Wasatch "
            "spectrometer-laser unit. With the Kymera selected the Wasatch is "
            "opened for laser control only. Changing the spectrometer "
            "reconnects at once; only spectra from the selected unit are shown.")
        laser_note.setWordWrap(True)
        laser_note.setStyleSheet(
            f"color: {COLORS.get('text_secondary', '#a0a0a0')}; font-size: 11px;"
            " padding: 2px 4px;")
        layout.addWidget(laser_note)

        # Wasatch Photonics settings (used when that backend connects).
        w_group = self._make_group("Wasatch Photonics")
        w_form = QFormLayout()
        w_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._add_dataclass_fields(w_form, self._config.spectrometer.wasatch,
                                   "spectrometer.wasatch")
        w_group.setLayout(w_form)
        layout.addWidget(w_group)

        # Andor Kymera + Newton settings (used when that backend connects).
        k_group = self._make_group("Andor Kymera 328i + Newton DU920P")
        k_form = QFormLayout()
        k_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._add_dataclass_fields(k_form, self._config.spectrometer.kymera,
                                   "spectrometer.kymera")
        k_group.setLayout(k_form)
        layout.addWidget(k_group)

        # Focus metric selection + entropy band.
        m_group = self._make_group("Autofocus Focus Metric")
        m_form = QFormLayout()
        m_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for name in ('af_metric', 'af_entropy_min_nm', 'af_entropy_max_nm'):
            self._add_field(m_form, name, getattr(self._config.spectrometer, name),
                            f"spectrometer.{name}")
        m_group.setLayout(m_form)
        layout.addWidget(m_group)

        # Focus-score bands get their own groups: they define the metric each
        # spectrometer autofocus minimises, not an acquisition setting. The
        # coarse (Prusa) and fine (NanoDrive) scans are tuned separately.
        note = QLabel(
            "Ratio focus score = summed intensity of the signal band / summed "
            "intensity of the reference band. Each scan keeps the Z with the "
            "LOWEST score. Bands are inclusive, in nanometres. The two scans "
            "have independent bands; set them the same to score alike. "
            "Their exposure times live in the Live tab's Autofocus panel."
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            f"color: {COLORS.get('text_secondary', '#a0a0a0')}; font-size: 11px;"
            " padding: 2px 4px;")
        layout.addWidget(note)

        for title, names in (
            ("Prusa Autofocus Score (coarse)",
             ('af_prusa_signal_min_nm', 'af_prusa_signal_max_nm',
              'af_prusa_reference_min_nm', 'af_prusa_reference_max_nm')),
            ("NanoDrive Autofocus Score (fine)",
             ('af_nano_signal_min_nm', 'af_nano_signal_max_nm',
              'af_nano_reference_min_nm', 'af_nano_reference_max_nm')),
        ):
            af_group = self._make_group(title)
            af_form = QFormLayout()
            af_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            for name in names:
                self._add_field(af_form, name,
                                getattr(self._config.spectrometer, name),
                                f"spectrometer.{name}")
            af_group.setLayout(af_form)
            layout.addWidget(af_group)

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
        """Add a single field row to a form layout and register the widget.

        Returns the created widget so callers can wire up extra behavior
        (e.g. enabling/disabling one field from another).
        """
        label_text = _label_for(name)
        tooltip = _tooltip_for(name)
        widget: QWidget

        if name in _COMBO_FIELDS:
            widget = QComboBox()
            for stored, text in _COMBO_FIELDS[name]:
                widget.addItem(text, stored)
            idx = widget.findData(value)
            widget.setCurrentIndex(idx if idx >= 0 else 0)
            if tooltip:
                widget.setToolTip(tooltip)
        elif isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
            if tooltip:
                widget.setToolTip(tooltip)
        elif isinstance(value, float):
            widget = QDoubleSpinBox()
            decimals = 3 if name in _HIGH_PRECISION_FIELDS else 2
            widget.setDecimals(decimals)
            if name in _UNCAPPED_FIELDS:
                widget.setRange(0.0, 1e12)
            else:
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
        return widget

    # ------------------------------------------------------------------
    # Read widget values
    # ------------------------------------------------------------------

    def _read_all_widgets(self) -> Dict[str, Any]:
        """Read current values from all widgets as a dotted-key dict."""
        result: Dict[str, Any] = {}
        for dotted_key, widget in self._widgets:
            if isinstance(widget, QComboBox):
                result[dotted_key] = widget.currentData()
            elif isinstance(widget, QCheckBox):
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

            if isinstance(widget, QComboBox):
                idx = widget.findData(obj)
                widget.setCurrentIndex(idx if idx >= 0 else 0)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(obj))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(obj))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(obj))
            elif isinstance(widget, QLineEdit) and not widget.isReadOnly():
                widget.setText(str(obj))

        # setChecked only emits toggled on an actual change, so re-sync the
        # dependent enable state explicitly after repopulating.
        self._on_calib_curve_toggled(self._calib_curve_check.isChecked())

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
