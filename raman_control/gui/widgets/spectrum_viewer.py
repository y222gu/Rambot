"""
Real-time spectrum viewer widget using PyQtGraph.

Features:
- High-performance real-time plotting
- Dark theme styling
- Interactive zoom/pan
- Peak markers
- Integration time and laser controls
"""

from typing import Optional, List, Tuple
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

import pyqtgraph as pg

from raman_control.gui.threads.spectrum_thread import SpectrumThread
from raman_control.gui.styles import COLORS
from raman_control.config.hardware_config import SpectrometerConfig
from raman_control.config.logging_config import get_logger

logger = get_logger('spectrum_viewer')

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
    - Peak detection markers
    """

    def __init__(
        self,
        spectrum_thread: Optional[SpectrumThread] = None,
        config: Optional[SpectrometerConfig] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self._spectrum_thread = spectrum_thread
        self._config = config or SpectrometerConfig()

        # Data storage
        self._wavelengths: Optional[np.ndarray] = None
        self._current_spectrum: Optional[np.ndarray] = None
        self._accumulated_spectra: List[np.ndarray] = []

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
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # === Plot Area ===
        self._setup_plot()
        main_layout.addWidget(self._plot_widget, 1)

        # === Controls ===
        controls_frame = QFrame()
        controls_frame.setStyleSheet(f'''
            QFrame {{
                background-color: {COLORS['bg_medium']};
                border: 1px solid {COLORS['border_dark']};
                border-radius: 6px;
            }}
        ''')
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(12, 8, 12, 8)
        controls_layout.setSpacing(20)

        # Acquisition controls
        acq_group = self._create_acquisition_controls()
        controls_layout.addWidget(acq_group)

        # Laser controls
        laser_group = self._create_laser_controls()
        controls_layout.addWidget(laser_group)

        # Display controls
        display_group = self._create_display_controls()
        controls_layout.addWidget(display_group)

        controls_layout.addStretch()

        main_layout.addWidget(controls_frame)

    def _setup_plot(self):
        """Create the PyQtGraph plot widget."""
        self._plot_widget = pg.PlotWidget()

        # Style the plot
        self._plot_widget.setBackground(COLORS['bg_light'])
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Configure axes
        self._plot_widget.setLabel('bottom', 'Wavenumber', units='cm⁻¹')
        self._plot_widget.setLabel('left', 'Intensity', units='a.u.')

        # Style axis labels
        label_style = {'color': COLORS['text_primary'], 'font-size': '12px'}
        self._plot_widget.getAxis('bottom').setLabel(**label_style)
        self._plot_widget.getAxis('left').setLabel(**label_style)

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

        # Peak markers
        self._peak_scatter = pg.ScatterPlotItem(
            pen=pg.mkPen(None),
            brush=pg.mkBrush(COLORS['error']),
            size=10,
            symbol='o'
        )
        self._plot_widget.addItem(self._peak_scatter)
        self._peak_scatter.hide()

        # Add crosshair for cursor position (hidden by default for performance)
        self._vline = pg.InfiniteLine(angle=90, movable=False,
            pen=pg.mkPen(COLORS['text_disabled'], width=1, style=Qt.PenStyle.DashLine))
        self._hline = pg.InfiniteLine(angle=0, movable=False,
            pen=pg.mkPen(COLORS['text_disabled'], width=1, style=Qt.PenStyle.DashLine))
        self._plot_widget.addItem(self._vline, ignoreBounds=True)
        self._plot_widget.addItem(self._hline, ignoreBounds=True)
        self._vline.hide()
        self._hline.hide()

        # Mouse tracking - DISABLED by default for performance
        # Only enabled when crosshair checkbox is checked
        self._mouse_tracking_enabled = False

        # Cursor position label
        self._cursor_label = QLabel("")
        self._cursor_label.setStyleSheet(f'''
            color: {COLORS['text_secondary']};
            font-family: "SF Mono", monospace;
            font-size: 11px;
            padding: 4px;
        ''')

    def _create_acquisition_controls(self) -> QWidget:
        """Create acquisition control widgets."""
        group = QWidget()
        layout = QHBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Start/Stop button
        self._acq_btn = QPushButton("▶ Start")
        self._acq_btn.setCheckable(True)
        self._acq_btn.setMinimumWidth(100)
        self._acq_btn.clicked.connect(self._on_acquisition_toggled)
        layout.addWidget(self._acq_btn)

        # Integration time
        layout.addWidget(QLabel("Integration:"))
        self._integration_spin = QSpinBox()
        self._integration_spin.setRange(1, 10000)
        self._integration_spin.setValue(int(self._config.integration_time))
        self._integration_spin.setSuffix(" ms")
        self._integration_spin.valueChanged.connect(self._on_integration_changed)
        layout.addWidget(self._integration_spin)

        # Averaging
        layout.addWidget(QLabel("Avg:"))
        self._averaging_spin = QSpinBox()
        self._averaging_spin.setRange(1, 100)
        self._averaging_spin.setValue(self._config.averaging)
        self._averaging_spin.valueChanged.connect(self._on_averaging_changed)
        layout.addWidget(self._averaging_spin)

        return group

    def _create_laser_controls(self) -> QWidget:
        """Create laser control widgets."""
        group = QWidget()
        layout = QHBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Laser on/off
        self._laser_btn = QPushButton("🔴 Laser OFF")
        self._laser_btn.setCheckable(True)
        self._laser_btn.setMinimumWidth(120)
        self._laser_btn.clicked.connect(self._on_laser_toggled)
        self._laser_btn.setStyleSheet('''
            QPushButton { background-color: #363636; }
            QPushButton:checked { background-color: #c62828; color: white; }
        ''')
        layout.addWidget(self._laser_btn)

        # Laser power in mW (typical range 0-500mW)
        layout.addWidget(QLabel("Power:"))
        self._power_spin = QSpinBox()
        self._power_spin.setRange(0, 500)
        self._power_spin.setValue(100)  # Default 100mW
        self._power_spin.setSuffix(" mW")
        self._power_spin.setFixedWidth(100)
        self._power_spin.valueChanged.connect(self._on_power_changed)
        layout.addWidget(self._power_spin)

        return group

    def _create_display_controls(self) -> QWidget:
        """Create display control widgets."""
        group = QWidget()
        layout = QHBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Show peaks checkbox
        self._peaks_check = QCheckBox("Show Peaks")
        self._peaks_check.toggled.connect(self._on_peaks_toggled)
        layout.addWidget(self._peaks_check)

        # Show crosshair checkbox
        self._crosshair_check = QCheckBox("Crosshair")
        self._crosshair_check.toggled.connect(self._on_crosshair_toggled)
        layout.addWidget(self._crosshair_check)

        # Auto-scale button
        self._autoscale_btn = QPushButton("Auto Scale")
        self._autoscale_btn.clicked.connect(self._on_autoscale)
        layout.addWidget(self._autoscale_btn)

        # Cursor position
        layout.addWidget(self._cursor_label)

        return group

    def _connect_signals(self):
        """Connect spectrum thread signals."""
        self._spectrum_thread.spectrum_ready.connect(self._on_spectrum_ready)
        self._spectrum_thread.connected.connect(self._on_connected)
        self._spectrum_thread.disconnected.connect(self._on_disconnected)
        self._spectrum_thread.laser_state_changed.connect(self._on_laser_state_changed)

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

        # Update plot
        self._spectrum_curve.setData(wavelengths, intensities)

        # Update peak markers if enabled (but not too frequently)
        if self._peaks_check.isChecked() and time_since_last > 0.2:
            self._update_peak_markers()

    def _on_connected(self):
        """Handle spectrometer connection."""
        self._acq_btn.setEnabled(True)
        self._laser_btn.setEnabled(True)
        logger.info("Spectrometer connected")

    def _on_disconnected(self):
        """Handle spectrometer disconnection."""
        self._acq_btn.setChecked(False)
        self._acq_btn.setEnabled(False)
        self._laser_btn.setChecked(False)
        self._laser_btn.setEnabled(False)
        logger.info("Spectrometer disconnected")

    def _on_laser_state_changed(self, is_on: bool):
        """Handle laser state change."""
        self._laser_btn.setChecked(is_on)
        self._laser_btn.setText("🔴 Laser ON" if is_on else "🔴 Laser OFF")

    def _on_acquisition_toggled(self, checked: bool):
        """Handle acquisition start/stop."""
        if self._spectrum_thread and self._spectrum_thread.is_connected:
            if checked:
                self._spectrum_thread.start_acquisition()
                self._acq_btn.setText("⏹ Stop")
            else:
                self._spectrum_thread.stop_acquisition()
                self._acq_btn.setText("▶ Start")
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

    def _on_integration_changed(self, value: int):
        """Handle integration time change."""
        if self._spectrum_thread:
            self._spectrum_thread.set_integration_time(float(value))

    def _on_averaging_changed(self, value: int):
        """Handle averaging change."""
        if self._spectrum_thread:
            self._spectrum_thread.set_averaging(value)

    def _on_power_changed(self, value: int):
        """Handle laser power change (in mW)."""
        if self._spectrum_thread:
            self._spectrum_thread.set_laser_power_mw(float(value))

    def _on_peaks_toggled(self, checked: bool):
        """Handle peaks visibility toggle."""
        if checked:
            self._peak_scatter.show()
            self._update_peak_markers()
        else:
            self._peak_scatter.hide()

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
            self._cursor_label.setText("")
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
        """Update crosshair position on mouse move (throttled)."""
        if not self._mouse_tracking_enabled:
            return
        try:
            if self._plot_widget.sceneBoundingRect().contains(pos):
                mouse_point = self._plot_widget.plotItem.vb.mapSceneToView(pos)
                self._vline.setPos(mouse_point.x())
                self._hline.setPos(mouse_point.y())
                self._cursor_label.setText(f"({mouse_point.x():.0f}, {mouse_point.y():.0f})")
        except RuntimeError:
            pass  # Widget was deleted

    def _update_peak_markers(self):
        """Find and mark peaks in the current spectrum."""
        if self._current_spectrum is None or self._wavelengths is None:
            return

        # Simple peak detection
        from scipy.signal import find_peaks
        try:
            peaks, _ = find_peaks(
                self._current_spectrum,
                height=np.mean(self._current_spectrum) + np.std(self._current_spectrum),
                distance=20
            )

            if len(peaks) > 0:
                peak_x = self._wavelengths[peaks]
                peak_y = self._current_spectrum[peaks]
                self._peak_scatter.setData(peak_x, peak_y)
        except ImportError:
            # scipy not available, disable peak detection
            pass

    def set_spectrum_thread(self, thread: SpectrumThread):
        """Set or update the spectrum thread."""
        self._spectrum_thread = thread
        self._connect_signals()

    def update_spectrum(self, wavelengths: np.ndarray, intensities: np.ndarray):
        """Manually update spectrum (for use without thread)."""
        self._wavelengths = wavelengths
        self._current_spectrum = intensities
        self._spectrum_curve.setData(wavelengths, intensities)
