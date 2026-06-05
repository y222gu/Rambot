"""
Autofocus panel widget for coarse and fine focus control.

Features:
- Coarse autofocus using PRUSA stepper
- Fine autofocus using NanoDrive piezo
- Z limit configuration
- Focus metric selection
- Real-time focus quality display
"""

from typing import Optional
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QProgressBar, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal

from raman_control.core.autofocus import AutofocusEngine, AutofocusConfig, FocusMetric
from raman_control.gui.styles import COLORS
from raman_control.config.logging_config import get_logger

logger = get_logger('autofocus_panel')


class AutofocusPanelWidget(QWidget):
    """
    Panel for autofocus control.

    Signals:
        autofocus_started: Emitted when autofocus begins
        autofocus_complete: Emitted when autofocus finishes (success, best_z)
        z_limits_changed: Emitted when Z limits are updated (upper, lower)
    """

    autofocus_started = pyqtSignal()
    autofocus_complete = pyqtSignal(bool, float)
    z_limits_changed = pyqtSignal(float, float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._autofocus = AutofocusEngine()
        self._running = False

        # Callbacks set by main window
        self._move_z_callback = None
        self._move_piezo_callback = None
        self._get_image_callback = None
        self._get_spectrum_callback = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)

        # === Z Limits ===
        limits_group = QGroupBox("Autofocus Z Limits")
        limits_layout = QVBoxLayout(limits_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Upper Limit:"))
        self._z_upper_spin = QDoubleSpinBox()
        self._z_upper_spin.setRange(0, 100)
        self._z_upper_spin.setValue(40.0)
        self._z_upper_spin.setSuffix(" mm")
        self._z_upper_spin.setDecimals(1)
        self._z_upper_spin.valueChanged.connect(self._on_limits_changed)
        row1.addWidget(self._z_upper_spin)

        row1.addWidget(QLabel("Lower Limit:"))
        self._z_lower_spin = QDoubleSpinBox()
        self._z_lower_spin.setRange(0, 100)
        self._z_lower_spin.setValue(30.0)
        self._z_lower_spin.setSuffix(" mm")
        self._z_lower_spin.setDecimals(1)
        self._z_lower_spin.valueChanged.connect(self._on_limits_changed)
        row1.addWidget(self._z_lower_spin)

        limits_layout.addLayout(row1)

        # Set from current position buttons
        row2 = QHBoxLayout()
        self._set_upper_btn = QPushButton("Set Upper from Current")
        self._set_upper_btn.clicked.connect(self._on_set_upper)
        row2.addWidget(self._set_upper_btn)

        self._set_lower_btn = QPushButton("Set Lower from Current")
        self._set_lower_btn.clicked.connect(self._on_set_lower)
        row2.addWidget(self._set_lower_btn)

        limits_layout.addLayout(row2)

        # Current Z display
        self._current_z_label = QLabel("Current Z: - mm")
        self._current_z_label.setStyleSheet(f'''
            font-family: "SF Mono", monospace;
            font-size: 14px;
            color: {COLORS['accent_primary']};
            padding: 8px;
            background-color: {COLORS['bg_light']};
            border-radius: 4px;
        ''')
        self._current_z_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        limits_layout.addWidget(self._current_z_label)

        main_layout.addWidget(limits_group)

        # === Coarse Autofocus ===
        coarse_group = QGroupBox("Coarse Autofocus (PRUSA Stepper)")
        coarse_layout = QVBoxLayout(coarse_group)

        params_row = QHBoxLayout()

        params_row.addWidget(QLabel("Step Size:"))
        self._coarse_step_spin = QSpinBox()
        self._coarse_step_spin.setRange(10, 500)
        self._coarse_step_spin.setValue(50)
        self._coarse_step_spin.setSuffix(" μm")
        params_row.addWidget(self._coarse_step_spin)

        params_row.addWidget(QLabel("Metric:"))
        self._coarse_metric_combo = QComboBox()
        self._coarse_metric_combo.addItems(["Contrast", "Gradient"])
        params_row.addWidget(self._coarse_metric_combo)

        coarse_layout.addLayout(params_row)

        self._coarse_btn = QPushButton("▶ Run Coarse Autofocus")
        self._coarse_btn.setMinimumHeight(36)
        self._coarse_btn.clicked.connect(self._on_run_coarse)
        coarse_layout.addWidget(self._coarse_btn)

        main_layout.addWidget(coarse_group)

        # === Fine Autofocus ===
        fine_group = QGroupBox("Fine Autofocus (NanoDrive Piezo)")
        fine_layout = QVBoxLayout(fine_group)

        fine_params = QHBoxLayout()

        fine_params.addWidget(QLabel("Step Size:"))
        self._fine_step_spin = QDoubleSpinBox()
        self._fine_step_spin.setRange(0.1, 10.0)
        self._fine_step_spin.setValue(1.0)
        self._fine_step_spin.setSuffix(" μm")
        self._fine_step_spin.setDecimals(1)
        fine_params.addWidget(self._fine_step_spin)

        fine_params.addWidget(QLabel("Range:"))
        self._fine_range_spin = QSpinBox()
        self._fine_range_spin.setRange(5, 100)
        self._fine_range_spin.setValue(50)
        self._fine_range_spin.setSuffix(" μm")
        fine_params.addWidget(self._fine_range_spin)

        fine_params.addWidget(QLabel("Metric:"))
        self._fine_metric_combo = QComboBox()
        self._fine_metric_combo.addItems(["Spectrometer Intensity", "Contrast"])
        fine_params.addWidget(self._fine_metric_combo)

        fine_layout.addLayout(fine_params)

        self._fine_btn = QPushButton("▶ Run Fine Autofocus")
        self._fine_btn.setMinimumHeight(36)
        self._fine_btn.clicked.connect(self._on_run_fine)
        fine_layout.addWidget(self._fine_btn)

        main_layout.addWidget(fine_group)

        # === Progress ===
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        progress_layout.addWidget(self._progress_bar)

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet(f'color: {COLORS["text_secondary"]};')
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self._status_label)

        # Focus quality indicator
        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Focus Quality:"))
        self._quality_bar = QProgressBar()
        self._quality_bar.setMaximum(100)
        self._quality_bar.setTextVisible(False)
        self._quality_bar.setStyleSheet(f'''
            QProgressBar {{
                border: 1px solid {COLORS['border_dark']};
                border-radius: 4px;
                background-color: {COLORS['bg_light']};
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['success']};
            }}
        ''')
        quality_row.addWidget(self._quality_bar, 1)
        progress_layout.addLayout(quality_row)

        # Abort button
        self._abort_btn = QPushButton("⏹ Abort")
        self._abort_btn.setEnabled(False)
        self._abort_btn.clicked.connect(self._on_abort)
        progress_layout.addWidget(self._abort_btn)

        main_layout.addWidget(progress_group)

        main_layout.addStretch()

    def _connect_signals(self):
        """Connect autofocus engine signals."""
        self._autofocus.progress.connect(self._on_progress)
        self._autofocus.completed.connect(self._on_completed)
        self._autofocus.error.connect(self._on_error)

    def set_callbacks(
        self,
        move_z=None,
        move_piezo=None,
        get_image=None,
        get_spectrum=None
    ):
        """Set callback functions for hardware control."""
        self._move_z_callback = move_z
        self._move_piezo_callback = move_piezo
        self._get_image_callback = get_image
        self._get_spectrum_callback = get_spectrum

    def update_current_z(self, z: float):
        """Update the current Z position display."""
        self._current_z_label.setText(f"Current Z: {z:.2f} mm")
        self._current_z = z

    def _on_limits_changed(self):
        """Handle Z limit changes."""
        upper = self._z_upper_spin.value()
        lower = self._z_lower_spin.value()

        # Ensure upper > lower
        if lower >= upper:
            self._z_lower_spin.setValue(upper - 1.0)
            lower = upper - 1.0

        self._autofocus.set_z_limits(upper, lower)
        self.z_limits_changed.emit(upper, lower)

    def _on_set_upper(self):
        """Set upper limit from current Z."""
        if hasattr(self, '_current_z'):
            self._z_upper_spin.setValue(self._current_z)

    def _on_set_lower(self):
        """Set lower limit from current Z."""
        if hasattr(self, '_current_z'):
            self._z_lower_spin.setValue(self._current_z)

    def _on_run_coarse(self):
        """Run coarse autofocus."""
        if self._move_z_callback is None:
            self._status_label.setText("Error: Stage not connected")
            return

        if self._get_image_callback is None:
            self._status_label.setText("Error: Camera not connected")
            return

        self._running = True
        self._update_ui_running(True)

        # Update config
        self._autofocus.config.coarse_step_um = self._coarse_step_spin.value()
        self._autofocus.config.z_upper_mm = self._z_upper_spin.value()
        self._autofocus.config.z_lower_mm = self._z_lower_spin.value()

        # Get metric function
        metric_idx = self._coarse_metric_combo.currentIndex()

        def get_metric():
            image = self._get_image_callback()
            if image is None:
                return 0.0
            if metric_idx == 0:
                return AutofocusEngine.calculate_contrast(image)
            else:
                return AutofocusEngine.calculate_gradient(image)

        self.autofocus_started.emit()

        # Run in separate thread (autofocus handles this internally)
        import threading
        thread = threading.Thread(
            target=self._autofocus.start_coarse_focus,
            args=(self._move_z_callback, get_metric)
        )
        thread.start()

    def _on_run_fine(self):
        """Run fine autofocus."""
        if self._move_piezo_callback is None:
            self._status_label.setText("Error: Piezo not connected")
            return

        self._running = True
        self._update_ui_running(True)

        # Update config
        self._autofocus.config.fine_step_um = self._fine_step_spin.value()
        self._autofocus.config.fine_range_um = self._fine_range_spin.value()

        # Get metric function
        metric_idx = self._fine_metric_combo.currentIndex()

        def get_metric():
            if metric_idx == 0 and self._get_spectrum_callback:
                spectrum = self._get_spectrum_callback()
                return AutofocusEngine.calculate_intensity(spectrum)
            elif self._get_image_callback:
                image = self._get_image_callback()
                return AutofocusEngine.calculate_contrast(image)
            return 0.0

        self.autofocus_started.emit()

        # Run in separate thread
        import threading
        thread = threading.Thread(
            target=self._autofocus.start_fine_focus,
            args=(self._move_piezo_callback, get_metric)
        )
        thread.start()

    def _on_abort(self):
        """Abort current autofocus."""
        self._autofocus.abort()

    def _on_progress(self, step: int, total: int, z: float, metric: float):
        """Update progress display."""
        percent = int(100 * step / total) if total > 0 else 0
        self._progress_bar.setValue(percent)
        self._status_label.setText(f"Step {step}/{total} | Z={z:.2f} | Metric={metric:.2f}")

        # Update quality bar (normalize metric)
        quality = min(100, int(metric / 10))  # Rough normalization
        self._quality_bar.setValue(quality)

    def _on_completed(self, success: bool, best_z: float, best_metric: float):
        """Handle autofocus completion."""
        self._running = False
        self._update_ui_running(False)

        if success:
            self._status_label.setText(f"Complete: Z={best_z:.3f}, Quality={best_metric:.2f}")
            self._progress_bar.setValue(100)
        else:
            self._status_label.setText("Autofocus aborted or failed")

        self.autofocus_complete.emit(success, best_z)

    def _on_error(self, message: str):
        """Handle autofocus error."""
        self._running = False
        self._update_ui_running(False)
        self._status_label.setText(f"Error: {message}")
        logger.error(f"Autofocus error: {message}")

    def _update_ui_running(self, running: bool):
        """Update UI based on running state."""
        self._coarse_btn.setEnabled(not running)
        self._fine_btn.setEnabled(not running)
        self._abort_btn.setEnabled(running)
        self._z_upper_spin.setEnabled(not running)
        self._z_lower_spin.setEnabled(not running)

    def get_z_limits(self) -> tuple:
        """Get current Z limits."""
        return (self._z_upper_spin.value(), self._z_lower_spin.value())
