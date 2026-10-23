"""
Real-time spectrum viewer widget using PyQtGraph.

Features:
- High-performance real-time plotting
- Dark theme styling
- Interactive zoom/pan
- Peak markers
- Integration time and laser controls
"""

import os
from datetime import datetime
from typing import Optional, List, Tuple, Callable
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QSizePolicy, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPoint, QSize, QEvent
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QBrush, QPolygon

import pyqtgraph as pg

from raman_control.gui.threads.spectrum_thread import SpectrumThread
from raman_control.gui.styles import COLORS
from raman_control.config.hardware_config import SpectrometerConfig
from raman_control.config.logging_config import get_logger

logger = get_logger('spectrum_viewer')

# No software cap on the spectrometer exposure time: long integrations are a
# legitimate way to get signal out of a weak sample, so the only limit is what
# the driver/hardware itself accepts. QSpinBox stores a 32-bit int, so its
# maximum is the highest value the widget can physically hold.
MAX_EXPOSURE_MS = 2147483647

# Configure PyQtGraph for performance + dark theme
pg.setConfigOptions(
    antialias=False,  # Disable antialiasing for better performance
    background=COLORS['bg_medium'],
    foreground=COLORS['text_primary'],
    useOpenGL=False,  # OpenGL can cause issues on some systems
)


class SpectrumViewerWidget(QWidget):
    """
    Real-time spectrum display widget.

    Features:
    - PyQtGraph plot with dark theme
    - Live spectrum updates at high frame rate
    - Integration time control
    - Laser power control
    - Multi-spectrum capture to nm-format CSV
    """

    # Capture progress for the status-bar progress bar (percent, message).
    capture_progress = pyqtSignal(int, str)
    # Spectrometer autofocus requests, carrying a dict of parameters
    # (steps, exposure_ms, power_mw, reps). One per focus stage.
    prusa_autofocus_requested = pyqtSignal(dict)       # coarse (Prusa Z)
    nanodrive_autofocus_requested = pyqtSignal(dict)   # fine (NanoDrive)

    def __init__(
        self,
        spectrum_thread: Optional[SpectrumThread] = None,
        config: Optional[SpectrometerConfig] = None,
        metadata_provider: Optional[Callable] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self._spectrum_thread = spectrum_thread
        self._config = config or SpectrometerConfig()
        # Returns a dict of extra capture metadata (stage/nanodrive position).
        self._metadata_provider = metadata_provider

        # Data storage. _wavelengths always holds Raman shift in cm⁻¹; the
        # x axis can be displayed as wavelength (nm) via the unit toggle.
        self._wavelengths: Optional[np.ndarray] = None
        self._current_spectrum: Optional[np.ndarray] = None
        self._x_in_nm = False  # False = Raman shift (cm⁻¹), True = wavelength (nm)
        self._crosshair_x: Optional[float] = None  # crosshair x in display units

        # Multi-spectrum capture state.
        self._capturing = False
        self._capture_target = 0
        self._capture_buffer: List[np.ndarray] = []
        self._capture_path: Optional[str] = None

        # Frame rate limiting for plot updates
        self._last_plot_update = 0.0
        self._min_plot_interval = 0.033  # ~30 fps max

        # Build UI
        self._setup_ui()

        # Connect signals if thread provided
        if self._spectrum_thread:
            self._connect_signals()

    def _setup_ui(self):
        """Build the user interface."""
        # Outer row: spectrum content (left) + autofocus panel (right). The
        # top margin adds a gap between the divider above and both the plot
        # and the autofocus panel.
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 12, 0, 0)
        outer.setSpacing(0)

        content = QWidget()
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(8, 0, 8, 0)
        main_layout.setSpacing(8)

        # === Plot Area ===
        self._setup_plot()
        main_layout.addWidget(self._plot_widget, 1)

        # === Controls (no surrounding frame) ===
        controls_frame = QWidget()
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(12, 8, 12, 0)
        controls_layout.setSpacing(20)

        # Acquisition + laser controls (fields on the left, play/laser
        # buttons on the right). The Crosshair / Auto Scale toggles live as an
        # overlay in the plot's top-right corner.
        controls_layout.addWidget(self._create_controls_grid())

        controls_layout.addStretch()

        # Crosshair x/y readout is drawn on the plot itself (orange, top-left),
        # not in the control row — see self._crosshair_text in _setup_plot.

        main_layout.addWidget(controls_frame)

        outer.addWidget(content, 1)  # content fills the left

        # Fixed Wasatch-autofocus panel column docked on the right.
        self._create_autofocus_panel(outer)

    _AF_PANEL_WIDTH = 330

    def _create_autofocus_panel(self, outer: QHBoxLayout):
        """Fixed Wasatch-autofocus panel column docked on the spectrum block's
        right side.

        Holds the Wasatch (spectrometer) autofocus parameters, a focus-score
        vs Z curve (as in the old noodlepy software), and the Prusa (coarse)
        and NanoDrive (fine) focus buttons.
        """
        self._af_zs = {'prusa': [], 'nano': []}
        self._af_scores = {'prusa': [], 'nano': []}

        # --- Titled frame, matching the "Captured Image" group box ---
        self._af_panel = QGroupBox("Spectrometer Autofocus")
        self._af_panel.setFixedWidth(self._AF_PANEL_WIDTH)
        lay = QVBoxLayout(self._af_panel)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        # --- Focus-score vs Z curves: Prusa (top) + NanoDrive (bottom),
        #     stacked in one canvas. Latest point highlighted in red. ---
        self._af_glw = pg.GraphicsLayoutWidget()
        self._af_glw.setBackground(COLORS['bg_light'])
        self._af_glw.setMinimumHeight(180)
        self._af_curves = {}
        self._af_latests = {}
        for row, (stage, label) in enumerate([('prusa', 'Prusa'),
                                              ('nano', 'NanoDrive')]):
            pl = self._af_glw.addPlot(row=row, col=0)
            pl.setTitle(label, color=COLORS['text_secondary'], size='9pt')
            pl.showGrid(x=True, y=True, alpha=0.3)
            pl.setLabel('left', 'Score')
            for ax in ('bottom', 'left'):
                pl.getAxis(ax).setPen(pg.mkPen(COLORS['border_light']))
                pl.getAxis(ax).setTextPen(pg.mkPen(COLORS['text_secondary']))
                pl.getAxis(ax).enableAutoSIPrefix(False)
            self._af_curves[stage] = pl.plot(
                pen=pg.mkPen(COLORS['accent_primary'], width=2),
                symbol='o', symbolSize=5,
                symbolBrush=COLORS['accent_primary'], symbolPen=None)
            self._af_latests[stage] = pl.plot(
                pen=None, symbol='o', symbolSize=10,
                symbolBrush=COLORS['error'], symbolPen=None)
        lay.addWidget(self._af_glw, 1)  # stretch — fills available height

        # --- Parameters (AF-only exposure/power, reps, step counts) ---
        def _spin(lo, hi, val, suffix="", width=56):
            sb = QSpinBox()
            sb.setRange(lo, hi)
            sb.setValue(val)
            sb.setFixedWidth(width)
            if suffix:
                sb.setSuffix(suffix)
            return sb

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        # AF acquisition settings (used only during focusing, not acquisition).
        # Exposure is per scan stage: the coarse Prusa sweep is usually run
        # short and fast, while the fine NanoDrive sweep can afford longer
        # integrations for a cleaner score.
        self._af_prusa_expo = _spin(1, MAX_EXPOSURE_MS, 200, width=76)
        self._af_prusa_expo.setToolTip(
            "Exposure time used during the Prusa (coarse) autofocus (ms)")
        self._af_nano_expo = _spin(1, MAX_EXPOSURE_MS, 200, width=76)
        self._af_nano_expo.setToolTip(
            "Exposure time used during the NanoDrive (fine) autofocus (ms)")
        self._af_power = _spin(0, 450, 450)
        self._af_power.setToolTip("Laser power used during autofocus (mW)")
        self._af_reps = _spin(1, 50, 3)
        self._af_reps.setToolTip("Spectra per Z step; the median is taken to reject cosmic rays")
        self._af_prusa_steps = _spin(2, 200, 20, width=46)
        self._af_prusa_steps.setToolTip("Prusa Z steps across the coarse range (left Autofocus panel)")
        self._af_nano_steps = _spin(2, 500, 10, width=46)
        self._af_nano_steps.setToolTip("NanoDrive steps across the fine range (left Autofocus panel)")

        # Labels right-aligned so they hug their boxes (no gap). Left group:
        # Expo/Power with unit; right group: the step boxes. Column 3 is a
        # spacer that pushes the step group to the right edge.
        _R = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        grid.addWidget(QLabel("Prusa expo:"), 0, 0, alignment=_R)
        grid.addWidget(self._af_prusa_expo, 0, 1)
        grid.addWidget(QLabel("ms"), 0, 2)
        grid.addWidget(QLabel("Prusa steps:"), 0, 4, alignment=_R)
        grid.addWidget(self._af_prusa_steps, 0, 5)

        grid.addWidget(QLabel("Nano expo:"), 1, 0, alignment=_R)
        grid.addWidget(self._af_nano_expo, 1, 1)
        grid.addWidget(QLabel("ms"), 1, 2)
        grid.addWidget(QLabel("Nano steps:"), 1, 4, alignment=_R)
        grid.addWidget(self._af_nano_steps, 1, 5)

        grid.addWidget(QLabel("Power:"), 2, 0, alignment=_R)
        grid.addWidget(self._af_power, 2, 1)
        grid.addWidget(QLabel("mW"), 2, 2)

        grid.setColumnStretch(3, 1)  # gap between the two field groups
        grid.setColumnMinimumWidth(3, 26)  # keep the unit clear of the steps
        lay.addLayout(grid)

        # Reps centered across the panel (not crammed into the first column).
        reps_row = QHBoxLayout()
        reps_row.setSpacing(6)
        reps_row.addStretch()
        reps_row.addWidget(QLabel("Reps per Step"))
        reps_row.addWidget(self._af_reps)
        reps_row.addStretch()
        lay.addLayout(reps_row)

        # --- Focus buttons (side by side) ---
        _af_btn_style = f'''
            QPushButton {{
                background-color: {COLORS['accent_primary']};
                border: 1px solid {COLORS['accent_hover']};
                border-radius: 5px;
                color: {COLORS['bg_dark']};
                font-weight: 600;
                font-size: 11px;
                padding: 6px 4px;
            }}
            QPushButton:hover {{ background-color: {COLORS['accent_hover']}; }}
            QPushButton:disabled {{
                background-color: transparent;
                border: 1px solid #4d4d4d;
                color: #505050;
            }}
        '''
        self._af_prusa_btn = QPushButton("Focus with Prusa")
        self._af_prusa_btn.setStyleSheet(_af_btn_style)
        self._af_prusa_btn.setEnabled(False)  # until a range is registered
        self._af_prusa_btn.setToolTip(
            "Coarse autofocus: step the Prusa Z across the range and pick the "
            "best focus.\nOnly available in objective view (the objective must "
            "be over the sample). Switch to Objective first.")
        self._af_prusa_btn.clicked.connect(self._on_prusa_autofocus_clicked)

        self._af_nano_btn = QPushButton("Focus with Nanodrive")
        self._af_nano_btn.setStyleSheet(_af_btn_style)
        self._af_nano_btn.setEnabled(False)  # until a range is registered
        self._af_nano_btn.setToolTip(
            "Fine autofocus: step the NanoDrive across the range and pick the "
            "best focus.\nOnly available in objective view (the objective must "
            "be over the sample). Switch to Objective first.")
        self._af_nano_btn.clicked.connect(self._on_nano_autofocus_clicked)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.addWidget(self._af_prusa_btn)
        btn_row.addWidget(self._af_nano_btn)
        lay.addLayout(btn_row)

        # Dock the panel as a fixed column on the right of the spectrum block.
        outer.addWidget(self._af_panel)

    def _af_params(self, stage: str) -> dict:
        """Current Wasatch-autofocus parameters from the panel.

        ``stage`` is 'prusa' (coarse) or 'nano' (fine); ``exposure_ms`` is that
        stage's own exposure, since the two scans are tuned separately.
        """
        expo = (self._af_nano_expo if stage == 'nano'
                else self._af_prusa_expo).value()
        return {
            "stage": stage,
            "exposure_ms": expo,
            "power_mw": self._af_power.value(),
            "reps": self._af_reps.value(),
            "coarse_steps": self._af_prusa_steps.value(),
            "fine_steps": self._af_nano_steps.value(),
        }

    def clear_focus_plot(self, stage: str):
        """Reset the focus-score curve for a stage ('prusa' or 'nano')."""
        self._af_zs[stage] = []
        self._af_scores[stage] = []
        self._af_curves[stage].setData([], [])
        self._af_latests[stage].setData([], [])

    def add_focus_point(self, stage: str, z: float, score: float):
        """Append a focus score at a Z step for a stage and redraw its curve."""
        self._af_zs[stage].append(float(z))
        self._af_scores[stage].append(float(score))
        self._af_curves[stage].setData(self._af_zs[stage], self._af_scores[stage])
        self._af_latests[stage].setData([float(z)], [float(score)])  # red latest

    def set_autofocus_enabled(self, enabled: bool):
        """Enable/disable the Prusa and NanoDrive focus buttons."""
        self._af_prusa_btn.setEnabled(enabled)
        self._af_nano_btn.setEnabled(enabled)

    def _on_prusa_autofocus_clicked(self):
        """Request spectrometer coarse autofocus by moving the Prusa in Z."""
        self.clear_focus_plot('prusa')
        self.prusa_autofocus_requested.emit(self._af_params('prusa'))

    def _on_nano_autofocus_clicked(self):
        """Request spectrometer fine autofocus by moving the NanoDrive."""
        self.clear_focus_plot('nano')
        self.nanodrive_autofocus_requested.emit(self._af_params('nano'))

    def _setup_plot(self):
        """Create the PyQtGraph plot widget."""
        self._plot_widget = pg.PlotWidget()

        # Style the plot
        self._plot_widget.setBackground(COLORS['bg_light'])
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Configure axes
        self._axis_label_style = {'color': COLORS['text_primary'], 'font-size': '12px'}
        self._set_axis_label('left', 'Intensity', 'a.u.')
        self._apply_x_axis()  # x label depends on the unit toggle

        # Set axis pen colors
        axis_pen = pg.mkPen(color=COLORS['border_light'], width=1)
        for axis in ['bottom', 'left', 'top', 'right']:
            self._plot_widget.getAxis(axis).setPen(axis_pen)
            self._plot_widget.getAxis(axis).setTextPen(pg.mkPen(COLORS['text_secondary']))

        # Create plot curve
        self._spectrum_curve = self._plot_widget.plot(
            pen=pg.mkPen(color=COLORS['accent_primary'], width=2),
            name='Spectrum'
        )

        # Create baseline curve (optional)
        self._baseline_curve = self._plot_widget.plot(
            pen=pg.mkPen(color=COLORS['warning'], width=1, style=Qt.PenStyle.DashLine),
            name='Baseline'
        )
        self._baseline_curve.hide()

        # Add crosshair for cursor position (hidden by default for performance)
        self._vline = pg.InfiniteLine(angle=90, movable=False,
            pen=pg.mkPen(COLORS['text_disabled'], width=1, style=Qt.PenStyle.DashLine))
        self._hline = pg.InfiniteLine(angle=0, movable=False,
            pen=pg.mkPen(COLORS['text_disabled'], width=1, style=Qt.PenStyle.DashLine))
        self._plot_widget.addItem(self._vline, ignoreBounds=True)
        self._plot_widget.addItem(self._hline, ignoreBounds=True)
        self._vline.hide()
        self._hline.hide()

        # Crosshair X/Y readout drawn in the plot's top-LEFT corner (orange), so
        # it doesn't take space in the control bar. anchor=(0,0) puts the text's
        # top-left corner at the placement point; we re-pin it to the current
        # view's top-left on every range change (sigRangeChanged), so it stays
        # put even while autoscale moves the Y axis.
        self._crosshair_text = pg.TextItem(text="", color=(255, 165, 0),
                                           anchor=(0, 0))
        self._crosshair_text.setZValue(100)
        self._plot_widget.addItem(self._crosshair_text, ignoreBounds=True)
        self._crosshair_text.hide()
        self._plot_widget.plotItem.vb.sigRangeChanged.connect(
            self._position_crosshair_text)

        # Mouse tracking - DISABLED by default for performance
        # Only enabled when crosshair checkbox is checked
        self._mouse_tracking_enabled = False

        # Cursor x/y readout — added to the bottom control row in _setup_ui.
        self._cursor_label = QLabel("")
        self._cursor_label.setStyleSheet(f'''
            color: {COLORS['text_secondary']};
            font-family: "SF Mono", monospace;
            font-size: 11px;
        ''')

        # Display toggles floating at the plot's top-right corner.
        self._create_plot_overlay()

    def _create_plot_overlay(self):
        """Show Peaks / Crosshair / Auto Scale floating at the plot top-right."""
        vp = self._plot_widget.viewport()

        self._plot_overlay = QWidget(vp)
        self._plot_overlay.setObjectName('plotOverlay')
        self._plot_overlay.setStyleSheet(f'''
            QWidget#plotOverlay {{
                background-color: rgba(20, 20, 20, 170);
                border: 1px solid {COLORS['border_dark']};
                border-radius: 6px;
            }}
        ''')
        ov = QHBoxLayout(self._plot_overlay)
        ov.setContentsMargins(8, 4, 8, 4)
        ov.setSpacing(10)

        self._xunit_check = QCheckBox("Wavelength (nm)")
        self._xunit_check.setToolTip(
            "Show the x axis as wavelength (nm) instead of Raman shift (cm⁻¹)")
        self._xunit_check.toggled.connect(self._on_xunit_toggled)
        ov.addWidget(self._xunit_check)

        self._crosshair_check = QCheckBox("Crosshair")
        self._crosshair_check.toggled.connect(self._on_crosshair_toggled)
        ov.addWidget(self._crosshair_check)

        self._autoscale_btn = QPushButton("Auto Scale")
        self._autoscale_btn.clicked.connect(self._on_autoscale)
        ov.addWidget(self._autoscale_btn)

        # Reposition the overlays whenever the plot is resized/shown.
        vp.installEventFilter(self)
        self._reposition_plot_overlay()

    def _reposition_plot_overlay(self):
        """Anchor the Crosshair / Auto Scale overlay to the plot's top-right."""
        if not hasattr(self, '_plot_overlay'):
            return
        vp = self._plot_widget.viewport()
        margin = 8
        self._plot_overlay.adjustSize()
        self._plot_overlay.move(
            max(margin, vp.width() - self._plot_overlay.width() - margin),
            margin,
        )
        self._plot_overlay.raise_()

    def eventFilter(self, obj, event):
        if (obj is self._plot_widget.viewport()
                and event.type() in (QEvent.Type.Resize, QEvent.Type.Show)):
            self._reposition_plot_overlay()
        return super().eventFilter(obj, event)

    def _create_controls_grid(self) -> QWidget:
        """Acquisition + laser controls.

        Left → right: Expo. Time field, Power field, then the play and laser
        buttons.
        """
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        # --- Play button styles (cyan outline idle / filled running),
        #     matching the live camera's acquisition button. ---
        self._acq_btn_outline = f'''
            QPushButton {{
                background-color: transparent;
                border: 2px solid {COLORS['accent_primary']};
                border-radius: 6px;
                color: {COLORS['accent_primary']};
                font-weight: 600;
                padding: 5px 12px;
            }}
            QPushButton:hover {{ background-color: rgba(0, 180, 180, 30); }}
            QPushButton:disabled {{
                background-color: transparent;
                border: 2px solid #4d4d4d;
                color: #505050;
            }}
        '''
        self._acq_btn_filled = f'''
            QPushButton {{
                background-color: {COLORS['accent_primary']};
                border: 2px solid {COLORS['accent_primary']};
                border-radius: 6px;
                color: {COLORS['bg_dark']};
                font-weight: 600;
                padding: 5px 12px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
                border-color: {COLORS['accent_hover']};
            }}
            QPushButton:disabled {{
                background-color: transparent;
                border: 2px solid #4d4d4d;
                color: #505050;
            }}
        '''

        # --- Widgets ---
        self._acq_btn = QPushButton()
        self._acq_btn.setCheckable(True)
        self._acq_btn.setFixedWidth(48)
        # Expand vertically so it spans both grid rows.
        self._acq_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._acq_btn.setIconSize(QSize(24, 24))      # match live camera icon size
        self._acq_btn.setIcon(self._make_play_icon(COLORS['accent_primary']))
        self._acq_btn.setStyleSheet(self._acq_btn_outline)
        self._acq_btn.clicked.connect(self._on_acquisition_toggled)

        self._laser_btn = QPushButton("Turn Laser On")
        self._laser_btn.setCheckable(True)
        self._laser_btn.setMinimumWidth(130)
        # Expand vertically so it spans both grid rows.
        self._laser_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._laser_btn.clicked.connect(self._on_laser_toggled)
        self._laser_btn.setStyleSheet('''
            QPushButton { background-color: #363636; }
            QPushButton:checked { background-color: #c62828; color: white; }
        ''')

        self._integration_spin = QSpinBox()
        # Uncapped on purpose (see MAX_EXPOSURE_MS): long integrations are a
        # valid way to pull signal out of a weak sample.
        self._integration_spin.setRange(1, MAX_EXPOSURE_MS)
        self._integration_spin.setValue(int(self._config.integration_time))
        # Width just enough for the number (text + spin arrows + frame padding).
        _int_w = self._integration_spin.fontMetrics().horizontalAdvance("1000000")
        self._integration_spin.setFixedWidth(_int_w + 32)
        self._integration_spin.valueChanged.connect(self._on_integration_changed)

        self._power_spin = QSpinBox()
        self._power_spin.setRange(0, int(self._config.laser_power_mw * 5))
        self._power_spin.setValue(int(self._config.laser_power_mw))
        _pow_w = self._power_spin.fontMetrics().horizontalAdvance("250.5")
        self._power_spin.setFixedWidth(_pow_w + 32)
        self._power_spin.valueChanged.connect(self._on_power_changed)

        # --- Expo. Time | Power ---
        grid.addWidget(QLabel("Expo. Time"), 0, 0)
        grid.addWidget(self._integration_spin, 0, 1)
        grid.addWidget(QLabel("ms"), 0, 2)
        grid.addWidget(QLabel("Power"), 0, 3)
        grid.addWidget(self._power_spin, 0, 4)
        grid.addWidget(QLabel("mW"), 0, 5)

        # --- Capture N spectra → save in native wavelength (nm) format ---
        self._num_spectra_spin = QSpinBox()
        self._num_spectra_spin.setRange(1, 100000)
        self._num_spectra_spin.setValue(10)
        self._num_spectra_spin.setToolTip(
            "Number of spectra to capture at the current exposure time and laser power")
        _ns_w = self._num_spectra_spin.fontMetrics().horizontalAdvance("100000")
        self._num_spectra_spin.setFixedWidth(_ns_w + 32)

        self._capture_btn = QPushButton("Capture")
        self._capture_btn.setToolTip(
            "Capture the set number of spectra and save them in wavelength (nm) format")
        self._capture_btn.setStyleSheet(f'''
            QPushButton {{
                background-color: {COLORS['accent_primary']};
                border: 1px solid {COLORS['accent_hover']};
                border-radius: 5px;
                color: {COLORS['bg_dark']};
                font-weight: 600;
                padding: 5px 12px;
            }}
            QPushButton:hover {{ background-color: {COLORS['accent_hover']}; }}
            QPushButton:disabled {{
                background-color: transparent;
                border: 1px solid #4d4d4d;
                color: #505050;
            }}
        ''')
        self._capture_btn.setEnabled(False)  # until the spectrometer connects
        self._capture_btn.clicked.connect(self._on_capture_clicked)

        # --- Buttons / capture controls, on the right of the field columns ---
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addWidget(self._acq_btn)
        btn_row.addWidget(self._laser_btn)
        # Capture controls sit to the right of the "Turn Laser On" button.
        btn_row.addWidget(QLabel("# Spectra:"))
        btn_row.addWidget(self._num_spectra_spin)
        btn_row.addWidget(self._capture_btn)
        grid.addLayout(btn_row, 0, 6)

        return container

    @staticmethod
    def _make_play_icon(color: str, size: int = 20) -> QIcon:
        """Draw a play triangle with the given color (matches live camera)."""
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(0, 0, 0, 0))
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(QColor(color)))
        p.setPen(Qt.PenStyle.NoPen)
        margin = size // 5
        triangle = QPolygon([
            QPoint(margin, margin),
            QPoint(size - margin, size // 2),
            QPoint(margin, size - margin),
        ])
        p.drawPolygon(triangle)
        p.end()
        return QIcon(pixmap)

    @staticmethod
    def _make_pause_icon(color: str, size: int = 20) -> QIcon:
        """Draw a pause icon (two vertical bars) with the given color."""
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(0, 0, 0, 0))
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(QColor(color)))
        p.setPen(Qt.PenStyle.NoPen)
        margin = size // 5
        bar_w = size // 5
        gap = size // 10
        x1 = (size - 2 * bar_w - gap) // 2
        x2 = x1 + bar_w + gap
        p.drawRoundedRect(x1, margin, bar_w, size - 2 * margin, 1, 1)
        p.drawRoundedRect(x2, margin, bar_w, size - 2 * margin, 1, 1)
        p.end()
        return QIcon(pixmap)

    def _update_acq_btn(self, running: bool):
        """Update acquisition button icon and style (play/pause)."""
        if running:
            self._acq_btn.setIcon(self._make_pause_icon(COLORS['bg_dark']))
            self._acq_btn.setStyleSheet(self._acq_btn_filled)
        else:
            self._acq_btn.setIcon(self._make_play_icon(COLORS['accent_primary']))
            self._acq_btn.setStyleSheet(self._acq_btn_outline)

    def _connect_signals(self):
        """Connect spectrum thread signals."""
        self._spectrum_thread.spectrum_ready.connect(self._on_spectrum_ready)
        self._spectrum_thread.connected.connect(self._on_connected)
        self._spectrum_thread.disconnected.connect(self._on_disconnected)
        self._spectrum_thread.laser_state_changed.connect(self._on_laser_state_changed)

    # --- X axis units (Raman shift cm⁻¹ <-> wavelength nm) -------------------

    def _wavenumber_to_wavelength(self, wn):
        """Convert Raman shift (cm⁻¹) to absolute (Stokes) wavelength (nm)."""
        lam0 = self._config.excitation_wavelength_nm or 785.0
        wn0 = 1e7 / lam0                                    # laser wavenumber
        scattered = wn0 - np.asarray(wn, dtype=np.float64)  # Stokes side
        scattered = np.where(scattered == 0, np.nan, scattered)
        return 1e7 / scattered

    def _current_x(self):
        """The x-axis array in the currently displayed unit, or None."""
        if self._wavelengths is None:
            return None
        if self._x_in_nm:
            return self._wavenumber_to_wavelength(self._wavelengths)
        return self._wavelengths

    def _set_axis_label(self, axis_name: str, text: str, units: str):
        """Label an axis with a fixed unit — no pyqtgraph SI prefix.

        pyqtgraph otherwise auto-prepends an SI prefix to the unit when the
        range is large (e.g. "knm", "kcm⁻¹"), which is meaningless here. We
        disable that and force an empty prefix / unit-scale of 1.
        """
        axis = self._plot_widget.getAxis(axis_name)
        axis.autoSIPrefix = False
        axis.autoSIPrefixScale = 1.0
        self._plot_widget.setLabel(axis_name, text, units=units, unitPrefix='',
                                   **self._axis_label_style)

    def _apply_x_axis(self):
        """Set the x-axis label/units to match the current unit toggle."""
        if self._x_in_nm:
            self._set_axis_label('bottom', 'Wavelength', 'nm')
        else:
            self._set_axis_label('bottom', 'Raman Shift', 'cm⁻¹')

    def _on_xunit_toggled(self, checked: bool):
        """Switch the x axis between Raman shift (cm⁻¹) and wavelength (nm)."""
        self._x_in_nm = checked
        self._apply_x_axis()
        x = self._current_x()
        if x is not None and self._current_spectrum is not None:
            self._spectrum_curve.setData(x, self._current_spectrum)
            self._plot_widget.enableAutoRange(axis='x')

    def _on_spectrum_ready(self, wavelengths: np.ndarray, intensities: np.ndarray, timestamp: float):
        """Handle new spectrum data with frame rate limiting."""
        import time

        self._wavelengths = wavelengths
        self._current_spectrum = intensities

        # Frame rate limiting - skip update if too soon after last one
        current_time = time.time()
        time_since_last = current_time - self._last_plot_update
        if time_since_last < self._min_plot_interval:
            return

        self._last_plot_update = current_time

        # Update plot (x in the currently selected unit)
        self._spectrum_curve.setData(self._current_x(), intensities)

        # Keep the crosshair intensity readout in sync with the live spectrum.
        self._update_crosshair_readout()

    def _on_connected(self):
        """Handle spectrometer connection."""
        self._acq_btn.setEnabled(True)
        self._laser_btn.setEnabled(True)
        self._capture_btn.setEnabled(True)
        # Auto-start acquisition on connect and reflect it on the play button
        # (checked + pause icon / filled style).
        if self._spectrum_thread:
            self._spectrum_thread.start_acquisition()
        self._acq_btn.setChecked(True)
        self._update_acq_btn(True)
        logger.info("Spectrometer connected")

    def _on_disconnected(self):
        """Handle spectrometer disconnection."""
        if self._capturing:
            self._finish_capture(save=False)  # abort an in-progress capture
        self._acq_btn.setChecked(False)
        self._update_acq_btn(False)
        self._acq_btn.setEnabled(False)
        self._laser_btn.setChecked(False)
        self._laser_btn.setEnabled(False)
        self._capture_btn.setEnabled(False)
        logger.info("Spectrometer disconnected")

    def _on_laser_state_changed(self, is_on: bool):
        """Handle laser state change."""
        self._laser_btn.setChecked(is_on)
        self._laser_btn.setText("Turn Laser Off" if is_on else "Turn Laser On")

    def _on_acquisition_toggled(self, checked: bool):
        """Handle acquisition start/stop."""
        if self._spectrum_thread and self._spectrum_thread.is_connected:
            if checked:
                self._spectrum_thread.start_acquisition()
                self._update_acq_btn(True)
            else:
                self._spectrum_thread.stop_acquisition()
                self._update_acq_btn(False)
        else:
            # Not connected - reset button state
            self._acq_btn.setChecked(False)

    def _on_laser_toggled(self, checked: bool):
        """Handle laser on/off."""
        if self._spectrum_thread and self._spectrum_thread.is_connected:
            self._spectrum_thread.set_laser(checked)
        else:
            # Not connected - reset button state
            self._laser_btn.setChecked(False)

    def acquisition_exposure_ms(self) -> float:
        """The ACQUISITION exposure (ms) shown in the spectrometer panel.

        This is the value the device must sit at whenever nothing is actively
        overriding it, so the autofocus restores to this rather than to a
        snapshot taken when the scan started.
        """
        return float(self._integration_spin.value())

    def acquisition_power_mw(self) -> float:
        """The ACQUISITION laser power (mW) shown in the spectrometer panel."""
        return float(self._power_spin.value())

    def _on_integration_changed(self, value: int):
        """Handle integration time change."""
        if self._spectrum_thread:
            self._spectrum_thread.set_integration_time(float(value))

    def _on_power_changed(self, value: int):
        """Handle laser power change (in mW)."""
        if self._spectrum_thread:
            self._spectrum_thread.set_laser_power_mw(float(value))

    # --- Multi-spectrum capture -------------------------------------------

    def _on_capture_clicked(self):
        """Capture N spectra at the current settings, saved in nm format."""
        if not (self._spectrum_thread and self._spectrum_thread.is_connected):
            logger.warning("Capture requested but spectrometer is not connected")
            return
        if self._capturing:
            return

        n = self._num_spectra_spin.value()
        default_name = datetime.now().strftime("%Y_%m_%d_capture_spectra.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Captured Spectra", default_name,
            "CSV files (*.csv);;All files (*)")
        if not path:
            return

        # Spectra stream in only while acquiring — make sure it's running.
        if not self._spectrum_thread.is_acquiring:
            self._spectrum_thread.start_acquisition()
            self._acq_btn.setChecked(True)
            self._update_acq_btn(True)

        self._capturing = True
        self._capture_target = n
        self._capture_buffer = []
        self._capture_path = path
        self._capture_btn.setEnabled(False)
        self._num_spectra_spin.setEnabled(False)
        self._capture_btn.setText(f"0/{n}")
        self.capture_progress.emit(0, f"Capturing 0/{n}")
        self._spectrum_thread.spectrum_ready.connect(self._on_capture_spectrum)
        logger.info(f"Capturing {n} spectra -> {path}")

    def _on_capture_spectrum(self, wavelengths: np.ndarray,
                             intensities: np.ndarray, timestamp: float):
        """Collect one streamed spectrum toward the capture target."""
        if not self._capturing:
            return
        self._capture_buffer.append(np.asarray(intensities, dtype=np.float64))
        done = len(self._capture_buffer)
        target = self._capture_target
        self._capture_btn.setText(f"{done}/{target}")
        self.capture_progress.emit(
            int(done / target * 100) if target else 100,
            f"Capturing {done}/{target}")
        if done >= target:
            self._finish_capture(save=True)

    def _finish_capture(self, save: bool):
        """Stop collecting and optionally write the captured spectra."""
        if self._spectrum_thread is not None:
            try:
                self._spectrum_thread.spectrum_ready.disconnect(self._on_capture_spectrum)
            except TypeError:
                pass  # already disconnected
        if save and self._capture_buffer and self._capture_path:
            self._save_capture(self._capture_path, self._capture_buffer)
        # Refresh the progress bar back to 0% once done (or aborted).
        self.capture_progress.emit(0, "")
        self._capturing = False
        self._capture_buffer = []
        self._capture_path = None
        self._capture_btn.setText("Capture")
        self._capture_btn.setEnabled(True)
        self._num_spectra_spin.setEnabled(True)

    def _save_capture(self, path: str, spectra: List[np.ndarray]):
        """Write captured spectra to CSV in native wavelength (nm) format.

        Column layout: wavelength_nm, spectrum_1, spectrum_2, ...
        """
        x_nm = self._spectrum_thread.wavelengths_nm
        # Be defensive about any length mismatch between axis and spectra.
        n_px = min([len(x_nm)] + [len(s) for s in spectra])
        cols = [np.asarray(x_nm[:n_px])] + [s[:n_px] for s in spectra]
        data = np.column_stack(cols)
        header = "wavelength_nm," + ",".join(
            f"spectrum_{i + 1}" for i in range(len(spectra)))
        try:
            np.savetxt(path, data, delimiter=",", header=header, comments="")
            logger.info(f"Saved {len(spectra)} spectra ({n_px} px) to {path}")
        except OSError as e:
            logger.error(f"Failed to save spectra to {path}: {e}")
            return

        self._save_metadata(path, len(spectra))

    def _save_metadata(self, spectra_path: str, n_spectra: int):
        """Write a sidecar metadata file next to the captured-spectra CSV."""
        meta = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "number_of_spectra": n_spectra,
            "exposure_time_ms": self._integration_spin.value(),
            "laser_power_mW": self._power_spin.value(),
            "excitation_wavelength_nm": self._config.excitation_wavelength_nm,
        }
        # Stage / nanodrive position, supplied by the host window.
        if self._metadata_provider is not None:
            try:
                meta.update(self._metadata_provider() or {})
            except Exception as e:  # noqa: BLE001 - never block saving
                logger.error(f"Capture metadata provider failed: {e}")
        meta["spectra_file"] = os.path.basename(spectra_path)

        meta_path = os.path.splitext(spectra_path)[0] + "_metadata.txt"
        try:
            with open(meta_path, "w") as f:
                f.write("Raman Capture Metadata\n")
                f.write("======================\n")
                for key, value in meta.items():
                    f.write(f"{key}: {value}\n")
            logger.info(f"Saved capture metadata to {meta_path}")
        except OSError as e:
            logger.error(f"Failed to save metadata to {meta_path}: {e}")

    def _on_crosshair_toggled(self, checked: bool):
        """Handle crosshair visibility toggle."""
        if checked:
            self._vline.show()
            self._hline.show()
            # Enable mouse tracking only when crosshair is visible
            if not self._mouse_tracking_enabled:
                self._plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
                self._mouse_tracking_enabled = True
        else:
            self._vline.hide()
            self._hline.hide()
            self._crosshair_text.hide()
            self._crosshair_x = None
            # Disable mouse tracking when crosshair is hidden for better performance
            if self._mouse_tracking_enabled:
                try:
                    self._plot_widget.scene().sigMouseMoved.disconnect(self._on_mouse_moved)
                except TypeError:
                    pass  # Already disconnected
                self._mouse_tracking_enabled = False

    def _on_autoscale(self):
        """Auto-scale the plot to fit data."""
        self._plot_widget.enableAutoRange()

    def _on_mouse_moved(self, pos):
        """Move the crosshair to the cursor's wavenumber/wavelength."""
        if not self._mouse_tracking_enabled:
            return
        try:
            if not self._plot_widget.sceneBoundingRect().contains(pos):
                return
            mouse_point = self._plot_widget.plotItem.vb.mapSceneToView(pos)
            self._crosshair_x = mouse_point.x()
            self._vline.setPos(self._crosshair_x)
            self._update_crosshair_readout(fallback_y=mouse_point.y())
        except RuntimeError:
            pass  # Widget was deleted

    def _update_crosshair_readout(self, fallback_y: Optional[float] = None):
        """Snap the horizontal line / label to the spectrum intensity at the
        crosshair x. Called on mouse move AND on each new spectrum, so the
        reported intensity tracks the live data instead of freezing."""
        if not self._mouse_tracking_enabled or self._crosshair_x is None:
            return
        x = self._crosshair_x
        disp_x = self._current_x()
        if (disp_x is not None and self._current_spectrum is not None
                and len(disp_x) > 0):
            idx = int(np.argmin(np.abs(disp_x - x)))
            xv = float(disp_x[idx])
            yv = float(self._current_spectrum[idx])
            self._hline.setPos(yv)
            unit = "nm" if self._x_in_nm else "cm⁻¹"
            xname = "Wavelength" if self._x_in_nm else "Raman Shift"
            xfmt = f"{xv:.1f}" if self._x_in_nm else f"{xv:.0f}"
            self._crosshair_text.setText(
                f"{xname} = {xfmt} {unit}\nIntensity = {yv:.0f}")
            self._crosshair_text.show()
            self._position_crosshair_text()
        elif fallback_y is not None:
            self._hline.setPos(fallback_y)
            self._crosshair_text.setText(f"Intensity = {fallback_y:.0f}")
            self._crosshair_text.show()
            self._position_crosshair_text()

    def _position_crosshair_text(self, *args):
        """Pin the crosshair readout to the top-left corner of the current
        view. Called on each readout update AND on every view-range change
        (e.g. autoscale), so it never drifts with the Y axis."""
        (xmin, _xmax), (_ymin, ymax) = self._plot_widget.plotItem.vb.viewRange()
        self._crosshair_text.setPos(xmin, ymax)

    def set_spectrum_thread(self, thread: SpectrumThread):
        """Set or update the spectrum thread."""
        self._spectrum_thread = thread
        self._connect_signals()

    def update_spectrum(self, wavelengths: np.ndarray, intensities: np.ndarray):
        """Manually update spectrum (for use without thread)."""
        self._wavelengths = wavelengths
        self._current_spectrum = intensities
        self._spectrum_curve.setData(self._current_x(), intensities)
