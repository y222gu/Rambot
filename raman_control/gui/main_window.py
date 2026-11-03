"""
Main window for the Raman microscopy control application.

Modern dark theme PyQt6 application with:
- Stage control panel (includes autofocus)
- Live view with integrated segmentation
- Spectrum display
- Protocol control
"""

import sys
import time
from typing import Optional
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QMessageBox, QFrame, QLabel, QSplitter,
    QProgressBar, QPushButton
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent

from raman_control.config.logging_config import setup_logging, get_logger
from raman_control.config.hardware_config import HardwareConfig
from raman_control.gui.styles import DARK_STYLESHEET, COLORS, apply_dark_theme
from raman_control.gui.wheel_filter import install_wheel_guard
from raman_control.gui.tooltip_style import install_fast_tooltips
from raman_control.gui.threads.motion_thread import MotionThread
from raman_control.gui.threads.camera_thread import CameraThread
from raman_control.gui.threads.spectrum_thread import SpectrumThread
from raman_control.gui.threads.laser_thread import LaserThread
from raman_control.gui.threads.autofocus_thread import AutofocusWorker
from raman_control.gui.threads.acquisition_thread import (
    AcquisitionWorker, AcquisitionContext)
from raman_control.gui.threads.robotic_thread import RoboticDepositionWorker
from raman_control.gui.threads.led_thread import LEDThread
from raman_control.gui.widgets.stage_control import StageControlWidget
from raman_control.gui.widgets.led_control import LEDControlWidget
from raman_control.gui.widgets.live_view import LiveViewWidget
from raman_control.gui.widgets.spectrum_viewer import SpectrumViewerWidget
from raman_control.gui.widgets.protocol_panel import ProtocolPanelWidget
from raman_control.gui.widgets.sample_grid import SampleGridWidget
from raman_control.gui.widgets.calibration_panel import CalibrationPanelWidget
from raman_control.core.state_machine import AcquisitionStateMachine
from raman_control.core.autofocus import AutofocusEngine
from raman_control.core.coordinate_manager import CoordinateManager, CameraView
from raman_control.core.slide_layout import plan_slide_hop, slide_at_position
from raman_control.hardware.nano_piezo import NanoPiezo
from raman_control.gui.dialogs.settings_dialog import SettingsDialog


logger = get_logger('main_window')


class DeviceIndicator(QWidget):
    """Single device connection indicator with LED-style light."""

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._name = name

        self._led = QLabel()
        self._led.setFixedSize(10, 10)
        self._set_status(False)

        self._label = QLabel(name)
        self._label.setTextFormat(Qt.TextFormat.RichText)
        self._label.setStyleSheet(f'color: {COLORS["text_secondary"]}; font-size: 11px;')

        layout.addWidget(self._led)
        layout.addWidget(self._label)
        self._connected = False
        self._simulated = False
        self._detail = ""  # e.g. the spectrometer model / CCD temperature

    def _set_status(self, connected: bool):
        color = COLORS['success'] if connected else '#5c5c5c'
        border = '#2e7d32' if connected else '#3d3d3d'
        self._led.setStyleSheet(f'''
            background-color: {color};
            border: 1px solid {border};
            border-radius: 5px;
        ''')

    def set_connected(self, connected: bool, simulated: bool = False):
        self._connected = connected
        self._simulated = simulated
        if not connected:
            self._detail = ""
        self._set_status(connected)
        self._render()

    def set_detail(self, detail: str):
        """Extra text shown after the name while connected, e.g.
        'Spectrometer (Kymera -60°C)'."""
        self._detail = detail or ""
        self._render()

    def _render(self):
        connected, simulated = self._connected, self._simulated
        name = self._name
        if connected and self._detail:
            name = (f'{self._name} <span style="color: {COLORS["text_secondary"]};">'
                    f'({self._detail})</span>')
        if connected and simulated:
            # Show "(SIM)" in orange next to device name
            self._label.setText(f'{name} <span style="color: {COLORS["warning"]};">(SIM)</span>')
            self._label.setStyleSheet(f'color: {COLORS["text_primary"]}; font-size: 11px;')
        else:
            self._label.setText(name)
            color = COLORS['text_primary'] if connected else COLORS['text_secondary']
            self._label.setStyleSheet(f'color: {color}; font-size: 11px;')


class StatusBarWidget(QWidget):
    """Custom status bar with device indicators and single progress bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 8, 4)  # no left margin (LED sits flush)
        layout.setSpacing(12)

        # Status message (far left)
        self._message_label = QLabel("")
        self._message_label.setStyleSheet(f'color: {COLORS["text_secondary"]};')
        layout.addWidget(self._message_label)

        layout.addStretch()

        # Label for the global progress bar
        progress_label = QLabel("Progress of Current Operation:")
        progress_label.setStyleSheet(f'color: {COLORS["text_secondary"]};')
        layout.addWidget(progress_label)

        # Global progress bar (left of the device indicators)
        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(200)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%p%")
        self._progress_bar.setStyleSheet(f'''
            QProgressBar {{
                border: 1px solid {COLORS['border_dark']};
                border-radius: 4px;
                background-color: {COLORS['bg_light']};
                text-align: center;
                color: {COLORS['text_primary']};
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['accent_primary']};
                border-radius: 3px;
            }}
        ''')
        layout.addWidget(self._progress_bar)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f'background-color: {COLORS["border_dark"]};')
        layout.addWidget(sep)

        # Device indicators (rightmost)
        devices_layout = QHBoxLayout()
        devices_layout.setSpacing(16)
        self._device_indicators = {}
        for device in ['Stage', 'Wide Cam', 'Obj Cam', 'NanoDrive', 'Spectrometer']:
            indicator = DeviceIndicator(device)
            self._device_indicators[device] = indicator
            devices_layout.addWidget(indicator)
        layout.addLayout(devices_layout)

    def insert_left_widget(self, widget):
        """Insert a widget at the far-left of the status bar (before the
        status message)."""
        self.layout().insertWidget(0, widget)

    def add_right_widget(self, widget, separator: bool = False):
        """Append a widget at the far-right of the status bar (after the
        device indicators). Optionally precede it with a vertical separator."""
        if separator:
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.VLine)
            sep.setStyleSheet(f'background-color: {COLORS["border_dark"]};')
            self.layout().addWidget(sep)
        self.layout().addWidget(widget)

    def set_device_connected(self, device: str, connected: bool, simulated: bool = False):
        if device in self._device_indicators:
            self._device_indicators[device].set_connected(connected, simulated)

    def set_device_detail(self, device: str, detail: str):
        """Show extra text next to a device name (model, temperature...)."""
        if device in self._device_indicators:
            self._device_indicators[device].set_detail(detail)

    def set_progress(self, value: int, text: str = ""):
        self._progress_bar.setValue(value)
        if text:
            self._progress_bar.setFormat(text)
        else:
            self._progress_bar.setFormat("%p%")

    def show_message(self, message: str, timeout_ms: int = 3000):
        self._message_label.setText(message)
        if timeout_ms > 0:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(timeout_ms, lambda: self._message_label.setText(""))


class MainWindow(QMainWindow):
    """Main application window for Raman microscopy control."""

    def __init__(self, config: Optional[HardwareConfig] = None):
        super().__init__()

        self._config = config or HardwareConfig()
        self._motion_thread: Optional[MotionThread] = None
        self._nanodrive: Optional[NanoPiezo] = None
        self._coordinate_manager = CoordinateManager(self._config.stage)
        self._current_position = (0.0, 0.0, 0.0)  # Cached from position_changed signal
        # Slide of the last manual objective-view move target. None = unknown
        # (startup, after a view switch / acquisition), which forces the next
        # manual objective move to hop over the slide barrier. See _manual_move.
        self._obj_target_slide = None

        # Autofocus: buttons are gated until a Z range is registered, and only
        # one scan runs at a time.
        self._af_registered = False
        self._af_worker: Optional[AutofocusWorker] = None

        # True while a Calibration-tab motion test runs (the UI is locked and
        # acquisition / autofocus / manual moves are refused meanwhile).
        self._calibration_running = False

        # Current-FoV acquisition worker (one run at a time).
        self._acq_worker: Optional[AcquisitionWorker] = None
        # Snapshot of widget enabled-states while the UI is locked mid-run.
        self._ui_lock_snapshot: dict = {}
        # Hardware shutdown runs at most once (closeEvent and aboutToQuit both
        # call it; the flag makes the second call a no-op).
        self._shutdown_done = False

        self._setup_window()
        self._setup_central_widget()
        self._setup_statusbar()
        self._init_threads()

        # Backstop: make sure hardware (esp. the LED serial port) is shut down
        # cleanly even on exit paths that bypass closeEvent (e.g. app.quit()).
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown_hardware)

        # Auto-connect to all devices on startup, deferred until the event loop
        # is running so the window paints first and the connect stays responsive.
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._on_connect_all)

        logger.info("Main window initialized")

    def _setup_window(self):
        self.setWindowTitle("Raman Microscopy Control")
        self.setMinimumSize(1750, 1100)
        self.setStyleSheet(DARK_STYLESHEET)

    def _setup_central_widget(self):
        # Hardware threads / shared objects
        self._motion_thread = MotionThread(self._config.stage, self)
        self._nanodrive = NanoPiezo(self._config.nanodrive)
        self._widefield_camera = CameraThread(
            camera_id='widefield',
            config=self._config.widefield_camera,
            dll_dir=self._config.dll_dir,
        )
        self._objective_camera = CameraThread(
            camera_id='objective',
            config=self._config.objective_camera,
            dll_dir=self._config.dll_dir,
        )
        # Excitation laser in laser-only mode: used when the active
        # spectrometer is the Andor Kymera (no laser of its own) and the
        # Wasatch unit only provides the 785 nm excitation. The spectrum thread
        # starts it when (and only when) its connected backend has no laser.
        self._laser_thread = LaserThread(self._config.spectrometer)
        self._spectrometer = SpectrumThread(
            config=self._config.spectrometer,
            laser_thread=self._laser_thread,
        )
        # LED illumination thread — its control widget lives at the far-left of
        # the status bar (added in _setup_statusbar).
        self._led_thread = LEDThread(self._config.led, self)
        # State machine
        self._state_machine = AcquisitionStateMachine()

        # === Tab widget spans the whole window; switching tabs swaps the
        # entire content (stage column included). The native tab bar is shown
        # at the top of the content area. ===
        self._tab_widget = QTabWidget()
        self._tab_widget.setDocumentMode(True)
        self._tab_widget.setStyleSheet(self._TAB_WIDGET_STYLE)
        self.setCentralWidget(self._tab_widget)

        # === Tab 1: Live ====================================================
        # Left column (stage control + sample grid) on the left, live view +
        # spectrometer on the right.
        live_tab = QWidget()
        live_tab_layout = QHBoxLayout(live_tab)
        live_tab_layout.setContentsMargins(0, 0, 0, 0)
        live_tab_layout.setSpacing(0)

        # --- Left column: Stage Control + Sample Grid ---
        left_panel = QFrame()
        left_panel.setObjectName("leftPanel")
        left_panel.setStyleSheet(f'''
            #leftPanel {{
                background-color: {COLORS['bg_dark']};
                border-right: 1px solid {COLORS['border_dark']};
            }}
        ''')
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._stage_control = StageControlWidget(
            self._motion_thread,
            self._config.stage,
            nanodrive=self._nanodrive,
            nanodrive_config=self._config.nanodrive,
            coordinate_manager=self._coordinate_manager,
        )
        self._stage_control.set_move_handler(self._manual_move)
        left_layout.addWidget(self._stage_control)

        # Sample Grid — below stage control (autofocus section)
        self._sample_grid = SampleGridWidget()
        self._sample_grid.set_grid_enabled(False)  # disabled until stage is referenced
        left_layout.addWidget(self._sample_grid, 1)  # stretch=1 so grid takes remaining space

        left_panel.setFixedWidth(512)
        live_tab_layout.addWidget(left_panel)

        # --- Right side: Live View (top) + Spectrometer (bottom) ---
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        # Extra left margin adds breathing room between the left column's
        # divider and the live camera / spectrometer.
        right_layout.setContentsMargins(20, 8, 8, 8)

        self._live_view = LiveViewWidget(
            widefield_thread=self._widefield_camera,
            objective_thread=self._objective_camera,
            position_provider=lambda: self._current_position,
            looking_at_provider=lambda: self._coordinate_manager.physical_to_logical(
                *self._current_position),
            widefield_config=self._config.widefield_camera,
            objective_config=self._config.objective_camera,
        )
        self._spectrum_viewer = SpectrumViewerWidget(
            spectrum_thread=self._spectrometer,
            config=self._config.spectrometer,
            metadata_provider=self._capture_metadata,
        )

        view_splitter = QSplitter(Qt.Orientation.Vertical)
        view_splitter.addWidget(self._live_view)
        view_splitter.addWidget(self._spectrum_viewer)
        # Extra space (e.g. when the window is maximized) goes to the
        # spectrometer block; the live-view block keeps its content height.
        view_splitter.setStretchFactor(0, 0)  # live view: no stretch
        view_splitter.setStretchFactor(1, 1)  # spectrometer absorbs extra
        view_splitter.setChildrenCollapsible(False)
        self._view_splitter = view_splitter
        # Default the split once the window is shown so the live-view block
        # is tall enough to show everything (see _apply_default_split).
        self._default_split_applied = False
        right_layout.addWidget(view_splitter)

        live_tab_layout.addWidget(right_panel, 1)
        self._tab_widget.addTab(live_tab, "Live")

        # === Tab 2: Auto Acquisition =========================================
        # Pass the Live-tab widgets so the panel can mirror their parameters
        # (and push edits back on Start to keep the two tabs consistent).
        self._protocol_panel = ProtocolPanelWidget(
            state_machine=self._state_machine,
            stage_control=self._stage_control,
            live_view=self._live_view,
            spectrum_viewer=self._spectrum_viewer,
        )
        self._tab_widget.addTab(self._protocol_panel, "Auto Acquisition")

        # === Tab 3: Calibration (unit 1 motion test) ========================
        # Self-contained: its moves are atomic relative blocks on the shared
        # motion queue and its images come from an opt-in raw-frame tap, so the
        # Live / Auto Acquisition behaviour is unchanged.
        self._calibration_panel = CalibrationPanelWidget(
            motion_thread=self._motion_thread,
            camera=self._widefield_camera,
            stage_config=self._config.stage,
            busy_reason=self._calibration_busy_reason,
        )
        self._tab_widget.addTab(self._calibration_panel, "Calibration")

    def _setup_statusbar(self):
        self._status_widget = StatusBarWidget()
        # No margins / size grip so the LED control sits flush in the far-left
        # corner of the window.
        self.statusBar().setContentsMargins(0, 0, 0, 0)
        self.statusBar().setSizeGripEnabled(False)
        # addWidget (not addPermanentWidget) fills from the LEFT edge, so the
        # LED control sits flush in the far-left corner.
        self.statusBar().addWidget(self._status_widget, 1)
        # LED illumination control at the far-left of the status bar.
        self._led_control = LEDControlWidget(
            self._led_thread, self._config.led.default_brightness)
        self._status_widget.insert_left_widget(self._led_control)

        # --- Right side of the status bar ---
        # Connect/Disconnect sits just right of the device indicators; Settings
        # and About are pushed to the far-right corner.
        # Connect/Disconnect stays in the status bar, just right of the device
        # indicators.
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setMinimumHeight(28)
        self._connect_btn.setMinimumWidth(110)
        self._connect_btn.clicked.connect(self._on_connect_toggle)
        self._devices_connected = False
        self._update_connect_button_state()  # start grey (disconnected)
        self._status_widget.add_right_widget(self._connect_btn)

        self._settings_btn = QPushButton("Settings")
        self._settings_btn.setMinimumHeight(28)
        self._settings_btn.setStyleSheet(self._TOOLBAR_BTN_CONNECT)
        self._settings_btn.clicked.connect(self._on_open_settings)

        # About is a flat, borderless element (matches the old toolbar action
        # look) rather than a bordered button.
        self._about_btn = QPushButton("ℹ About")
        self._about_btn.setMinimumHeight(28)
        self._about_btn.setStyleSheet(self._TOOLBAR_BTN_FLAT)
        self._about_btn.clicked.connect(self._show_about)

        # Settings + About live in the top-right corner of the tab bar (same row
        # as the tabs), About to the right of Settings.
        _tab_corner = QWidget()
        _tab_corner_lay = QHBoxLayout(_tab_corner)
        _tab_corner_lay.setContentsMargins(0, 0, 6, 0)
        _tab_corner_lay.setSpacing(4)
        _tab_corner_lay.addWidget(self._settings_btn)
        _tab_corner_lay.addWidget(self._about_btn)
        self._tab_widget.setCornerWidget(_tab_corner, Qt.Corner.TopRightCorner)

        self.statusBar().setStyleSheet(f'''
            QStatusBar {{
                background-color: {COLORS['bg_medium']};
                border-top: 1px solid {COLORS['border_dark']};
            }}
        ''')

    def _init_threads(self):
        # Stage signals
        self._motion_thread.connected.connect(self._on_stage_connected)
        self._motion_thread.disconnected.connect(self._on_stage_disconnected)
        self._motion_thread.position_changed.connect(self._on_position_changed)
        self._motion_thread.error.connect(self._on_stage_error)

        # Widefield camera
        self._widefield_camera.connected.connect(self._on_widefield_connected)
        self._widefield_camera.disconnected.connect(
            lambda: self._status_widget.set_device_connected('Wide Cam', False))

        # Objective camera
        self._objective_camera.connected.connect(self._on_objective_connected)
        self._objective_camera.disconnected.connect(
            lambda: self._status_widget.set_device_connected('Obj Cam', False))

        # Spectrometer
        self._spectrometer.connected.connect(self._on_spectrometer_connected)
        self._spectrometer.disconnected.connect(
            lambda: self._status_widget.set_device_connected('Spectrometer', False))
        self._spectrometer.disconnected.connect(
            lambda: self._status_widget.set_device_detail('Spectrometer', ''))
        self._spectrometer.status_changed.connect(self._on_spectrometer_status)
        self._spectrometer.laser_available_changed.connect(
            self._on_laser_available_changed)
        self._spectrometer.error.connect(
            lambda msg: self._status_widget.show_message(f"⚠ Spectrometer: {msg}", 6000))

        # Live view camera switch
        self._live_view.camera_switch_requested.connect(self._on_camera_switch_requested)

        # Stage reference status → enable/disable FOV switch buttons
        self._stage_control.referenced_changed.connect(self._on_referenced_changed)

        # Autofocus scan ranges registered from the stage panel
        self._stage_control.ranges_registered.connect(self._on_ranges_registered)
        # ...and invalidated (e.g. on disconnect): re-lock the AF-gated buttons.
        self._stage_control.registration_reset.connect(self._on_registration_reset)

        # Camera autofocus requested from the live view control bar
        self._live_view.camera_autofocus_requested.connect(self._on_camera_autofocus)

        # Sample points from live view to grid
        self._live_view.positions_generated.connect(self._on_positions_generated)

        # Click a generated pattern point → move stage to it
        self._live_view.move_to_point_requested.connect(self._on_move_to_point)

        # Drive the live view's auto-center loop on move completion
        self._motion_thread.movement_complete.connect(
            self._live_view.on_stage_move_finished)

        # Segmentation progress → status-bar progress bar
        self._live_view.segmentation_progress.connect(
            self._status_widget.set_progress)

        # Spectrum capture progress → status-bar progress bar
        self._spectrum_viewer.capture_progress.connect(
            self._status_widget.set_progress)

        # Spectrometer autofocus (Prusa coarse / NanoDrive fine) from the tab
        self._spectrum_viewer.prusa_autofocus_requested.connect(
            self._on_prusa_autofocus)
        self._spectrum_viewer.nanodrive_autofocus_requested.connect(
            self._on_nanodrive_autofocus)

        # Sample grid well click → move stage
        self._sample_grid.position_selected.connect(self._on_well_clicked)

        # Keep the Live-tab and Auto-tab droplet maps in sync: loading a config
        # in one applies the same sample colours to the other. apply_config does
        # not re-emit, so there's no feedback loop.
        auto_grid = getattr(self._protocol_panel, '_sample_grid', None)
        if auto_grid is not None:
            self._sample_grid.config_loaded.connect(auto_grid.apply_config)
            auto_grid.config_loaded.connect(self._sample_grid.apply_config)

        # Current-FoV acquisition: the panel requests, the worker (owned here)
        # runs over the points generated on the captured image.
        self._protocol_panel.acquisition_requested.connect(
            self._on_start_current_fov)
        self._protocol_panel.robotic_requested.connect(self._on_start_robotic)
        self._protocol_panel.pause_requested.connect(self._on_acq_pause)
        self._protocol_panel.resume_requested.connect(self._on_acq_resume)
        self._protocol_panel.stop_requested.connect(self._on_acq_stop)

        # Calibration tab: lock the rest of the UI while a test runs.
        self._calibration_panel.running_changed.connect(
            self._on_calibration_running)
        self._calibration_panel.status_message.connect(
            self._status_widget.show_message)
        self._calibration_panel.progress.connect(self._status_widget.set_progress)

        # The illumination LED is independent of the measurement hardware: it
        # connects at launch and stays available (controls are always live and
        # the thread reconnects on demand), so it is NOT tied to the global
        # Connect/Disconnect. The port closes only on app exit.
        self._led_thread.start()

    def _on_widefield_connected(self):
        sim = self._widefield_camera.is_simulated
        self._status_widget.set_device_connected('Wide Cam', True, simulated=sim)

    def _on_objective_connected(self):
        sim = self._objective_camera.is_simulated
        self._status_widget.set_device_connected('Obj Cam', True, simulated=sim)

    def _on_spectrometer_connected(self):
        sim = self._spectrometer.is_simulated
        self._status_widget.set_device_connected('Spectrometer', True, simulated=sim)
        label = self._spectrometer.backend_label
        self._status_widget.set_device_detail('Spectrometer', label)
        self._status_widget.show_message(f"Spectrometer connected: {label}", 5000)
        logger.info(f"Spectrometer backend in use: {self._spectrometer.backend_name}, "
                    f"focus metric: {self._spectrometer.focus_metric}")

    def _on_spectrometer_status(self, status):
        """Live detector temperature (Kymera/Newton) in the status bar."""
        label = self._spectrometer.backend_label
        temp = getattr(status, 'detector_temperature_c', None)
        if temp is None:
            self._status_widget.set_device_detail('Spectrometer', label)
            return
        stable = getattr(status, 'temperature_stable', None)
        mark = "" if stable else " (cooling)"
        self._status_widget.set_device_detail(
            'Spectrometer', f"{label} {temp:.0f}°C{mark}")

    def _on_laser_available_changed(self, available: bool):
        if not available and self._spectrometer.is_connected:
            self._status_widget.show_message(
                "Excitation laser (Wasatch unit) not available", 6000)

    _TOOLBAR_BTN_CONNECT = '''
        QPushButton {
            background-color: transparent;
            border: 2px solid #00b4b4;
            border-radius: 6px;
            font-size: 13px;
            font-weight: bold;
            color: #00b4b4;
            padding: 4px 16px;
        }
        QPushButton:hover { background-color: #2a3a3a; border-color: #00d4d4; color: #00d4d4; }
        QPushButton:pressed { background-color: #1a3030; }
    '''

    # Flat, borderless element — matches the old toolbar action look.
    _TOOLBAR_BTN_FLAT = f'''
        QPushButton {{
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 4px;
            padding: 6px 12px;
            color: {COLORS['text_primary']};
        }}
        QPushButton:hover {{ background-color: {COLORS['bg_elevated']}; }}
        QPushButton:pressed {{ background-color: {COLORS['bg_light']}; }}
    '''

    # Main view tabs (in the content area): the selected tab is filled cyan.
    _TAB_WIDGET_STYLE = '''
        QTabWidget::pane {
            border: 1px solid #3d3d3d;
            border-radius: 6px;
            top: -1px;
        }
        QTabWidget::tab-bar { left: 8px; }
        QTabBar { background: transparent; }
        QTabBar::tab {
            background-color: transparent;
            border: 1px solid #3d3d3d;
            border-bottom: none;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            font-size: 13px;
            color: #b0b0b0;
            padding: 8px 18px;
            margin-right: 2px;
            min-height: 20px;
        }
        QTabBar::tab:hover { background-color: #2a3a3a; color: #e0e0e0; }
        QTabBar::tab:selected {
            background-color: #00b4b4;
            color: #1a1a1a;
            font-weight: bold;
        }
    '''

    _TOOLBAR_BTN_CYAN = f'''
        QPushButton {{
            background-color: {COLORS['accent_primary']};
            border: 2px solid {COLORS['accent_hover']};
            border-radius: 6px;
            font-size: 13px;
            font-weight: bold;
            color: {COLORS['bg_dark']};
            padding: 4px 16px;
        }}
        QPushButton:hover {{ background-color: {COLORS['accent_hover']}; }}
        QPushButton:pressed {{ background-color: {COLORS['accent_pressed']}; }}
    '''

    def _update_connect_button_state(self):
        """Update the Connect/Disconnect button text and style based on
        connection state."""
        if self._devices_connected:
            self._connect_btn.setText("Disconnect")
            self._connect_btn.setStyleSheet(self._TOOLBAR_BTN_CYAN)
            self._connect_btn.setToolTip(
                "Disconnect all hardware.\n"
                "Warning: you will lose the calibrated home position. The stage "
                "must be re-referenced (Reference All) after reconnecting, and the "
                "current captured image and sample points will be cleared.")
        else:
            self._connect_btn.setText("Connect")
            self._connect_btn.setStyleSheet(self._TOOLBAR_BTN_CONNECT)
            self._connect_btn.setToolTip("Connect to all hardware.")

    def _on_connect_toggle(self):
        if self._devices_connected:
            self._on_disconnect_all()
        else:
            self._on_connect_all()

    def _on_connect_all(self):
        logger.info("Connecting to hardware...")
        self._status_widget.show_message("Connecting...")

        if not self._motion_thread.isRunning():
            self._motion_thread.start()
        if not self._widefield_camera.isRunning():
            self._widefield_camera.start()
        if not self._objective_camera.isRunning():
            self._objective_camera.start()
        if not self._spectrometer.isRunning():
            self._spectrometer.start()
        # NOTE: the LED is independent of this Connect button — it runs from
        # app launch (see _init_threads) and reconnects on demand.

        # NanoDrive (not a QThread – connect directly)
        if self._nanodrive and not self._nanodrive.is_connected:
            self._nanodrive.connect()
            if self._nanodrive.is_connected:
                sim = self._nanodrive.is_simulated
                self._status_widget.set_device_connected('NanoDrive', True, simulated=sim)
                logger.info(f"NanoDrive connected ({'simulated' if sim else 'hardware'})")
            else:
                self._status_widget.show_message("NanoDrive connection failed", 5000)

        self._devices_connected = True
        self._update_connect_button_state()

    def _on_disconnect_all(self):
        logger.info("Disconnecting...")
        for thread in [self._motion_thread, self._widefield_camera,
                       self._objective_camera, self._spectrometer]:
            if thread and thread.isRunning():
                thread.stop()

        # The LED is independent of this Disconnect button (it stays available);
        # it is only turned off by the emergency stop and closed on app exit.

        if self._nanodrive and self._nanodrive.is_connected:
            self._nanodrive.disconnect()
            self._status_widget.set_device_connected('NanoDrive', False)

        # Re-initialise the capture/segmentation/points state: the stage must be
        # re-referenced after reconnecting, which invalidates the captured
        # image's home-anchored calibration and the points generated from it.
        # (The stage-referenced gating itself resets via stage_control's
        # referenced_changed on the motion thread's 'disconnected' signal.)
        self._live_view.reset_capture_state()

        self._devices_connected = False
        self._update_connect_button_state()

    def _on_stage_connected(self):
        self._status_widget.set_device_connected('Stage', True)
        self._status_widget.show_message("Stage connected")

    def _on_stage_disconnected(self):
        self._status_widget.set_device_connected('Stage', False)

    def _on_position_changed(self, x: float, y: float, z: float):
        # Cache the REAL physical stage position. It's shown in the stage
        # panel and used for autofocus and capture metadata. Sample points are
        # kept in the widefield frame and add the wide->objective offset for
        # the active camera (see _on_move_to_point / _on_camera_switch).
        self._current_position = (x, y, z)
        # Track the live "you are here" marker on the captured image.
        self._live_view.update_probe_marker()
        # Keep the live-tab droplet map in sync with where the stage actually is.
        self._sync_droplet_map_to_position(x, y)

    def _well_at_position(self, x: float, y: float):
        """The (row, col) of the droplet-map well the physical stage position is
        over, or None if it's more than half a well-pitch from any valid well."""
        grid = getattr(self, '_sample_grid', None)
        if grid is None:
            return None
        drop = self._config.sample_drop
        stage = self._config.stage
        pitch = drop.well_pitch_mm or 1.0
        cm = self._coordinate_manager
        # The well grid is in widefield (logical) coords; convert the physical
        # stage position back to logical (remove the objective offset in obj view).
        if cm.current_view == CameraView.OBJECTIVE:
            lx = x - cm.offset_x
            ly = y - cm.offset_y_at(lx)
        else:
            lx, ly = x, y
        row_f = (lx - stage.home_x - drop.offset_home_to_a1_x) / pitch
        col_f = (ly - stage.home_y - drop.offset_home_to_a1_y) / pitch
        row, col = round(row_f), round(col_f)
        dist = ((row_f - row) ** 2 + (col_f - col) ** 2) ** 0.5 * pitch
        if dist <= pitch * 0.5 and grid.is_valid_well(row, col):
            return (row, col)
        return None

    def _sync_droplet_map_to_position(self, x: float, y: float):
        """Highlight the droplet-map well the stage is currently over (live tab),
        or clear the highlight + label once it moves away, so the map reflects
        the real position."""
        grid = getattr(self, '_sample_grid', None)
        if grid is None:
            return
        well = self._well_at_position(x, y)
        if well is not None:
            grid.set_current_position(*well)
        else:
            grid.clear_current_position()

    def _droplet_info_at_position(self, x: float, y: float):
        """(well_name, sample_name, droplet_number) for the well at the given
        physical position. well_name is None if not over a valid well; sample
        name / number are None if no config is loaded for that well."""
        well = self._well_at_position(x, y)
        if well is None:
            return None, None, None
        from raman_control.gui.widgets.sample_grid import well_name
        name = well_name(*well)
        info = self._sample_grid.droplet_map().get(name) or {}
        return name, info.get('sample_name'), info.get('droplet_number')

    def _capture_metadata(self) -> dict:
        """Stage / nanodrive position metadata for a spectrum capture.

        Uses cached values only (no blocking hardware calls): the stage
        position from the ``position_changed`` signal and the nanodrive's
        cached Z. ``stage_z`` is the coarse (Prusa) focus; ``nanodrive_z`` is
        the fine piezo focus.
        """
        x, y, z = self._current_position
        nano_z = None
        if self._nanodrive is not None and self._nanodrive.is_connected:
            nano_z = self._nanodrive.position
        meta = {
            "stage_x_mm": round(x, 4),
            "stage_y_mm": round(y, 4),
            "stage_z_mm": round(z, 4),          # coarse / autofocus position
            "nanodrive_z_um": round(nano_z, 3) if nano_z is not None else None,
        }
        # Which spectrometer took the capture (Wasatch / Kymera + grating,
        # CCD temperature...). Cached values only, no hardware call.
        meta.update(self._spectrometer.metadata())
        return meta

    def _on_stage_error(self, message: str):
        if "not connected" not in message.lower():
            self._status_widget.show_message(f"⚠ Stage: {message}", 5000)

    def _on_referenced_changed(self, ref_status: dict):
        """Enable FOV switch buttons and sample grid once all axes are referenced."""
        all_ref = all(ref_status.values())
        self._live_view.set_stage_referenced(all_ref)
        self._sample_grid.set_grid_enabled(all_ref)

    def _on_camera_switch_requested(self, camera_name: str):
        """Handle camera view switch.

        The stage offset move keeps both cameras centered on the same sample
        point. ``self._current_position`` is the REAL physical stage position,
        so the target is simply the current position plus (to objective) or
        minus (to wide) the wide->objective offset.
        """
        target_view = CameraView.OBJECTIVE if camera_name == "objective" else CameraView.WIDE_FOV

        try:
            if self._coordinate_manager.current_view != target_view:
                cur = self._current_position  # real physical stage position

                if target_view == CameraView.OBJECTIVE:
                    is_safe, msg = self._coordinate_manager.can_switch_to_objective(
                        cur[0], cur[1], cur[2])
                    target = self._coordinate_manager.get_objective_position(
                        cur[0], cur[1], cur[2])
                else:
                    is_safe, msg = self._coordinate_manager.can_switch_to_wide(
                        cur[0], cur[1], cur[2])
                    target = self._coordinate_manager.get_wide_position(
                        cur[0], cur[1], cur[2])

                if not is_safe:
                    logger.warning(f"Camera switch blocked: {msg}")
                    self._status_widget.show_message(f"Switch blocked: {msg}", 5000)
                    old_cam = 'wide' if camera_name == 'objective' else 'objective'
                    self._live_view.switch_camera(old_cam)
                    return

                self._coordinate_manager.switch_view(target_view)

                self._obj_target_slide = None
                # Update editable Z boxes (WF vs Obj) and the spectrometer
                # focus buttons (objective view only).
                self._stage_control.on_camera_view_changed()
                self._update_af_buttons()
                if self._motion_thread:
                    self._status_widget.show_message("Moving for camera switch...")
                    self._motion_thread.request_move_to(
                        x=target[0], y=target[1], z=target[2])
        except Exception as e:
            logger.error(f"Error during camera switch: {e}")
            self._status_widget.show_message(f"Camera switch error: {e}", 5000)

        # View switch is always instant regardless of stage movement
        self._live_view.switch_camera(camera_name)
        logger.info(f"Camera switched to: {camera_name}")

    # --- Autofocus -----------------------------------------------------------

    def _update_af_buttons(self):
        """Enable the autofocus buttons once a range is registered and no scan
        is already running.

        Camera autofocus works in either view (each scans the range for the
        current view). The spectrometer focus buttons require objective view —
        in widefield the objective isn't over the sample, so focusing it makes
        no sense.
        """
        acq_running = (self._acq_worker is not None
                       and self._acq_worker.isRunning())
        base = (self._af_registered and self._af_worker is None
                and not acq_running and not self._calibration_running)
        objective = self._coordinate_manager.current_view == CameraView.OBJECTIVE
        self._live_view.set_autofocus_enabled(base)
        self._spectrum_viewer.set_autofocus_enabled(base and objective)

    def _active_prusa_z_limits(self) -> tuple:
        """Prusa Z limits (upper, lower) for the CURRENT camera view:
        widefield range in widefield view, objective range in objective view."""
        if self._coordinate_manager.current_view == CameraView.OBJECTIVE:
            return self._stage_control.get_objective_z_limits()
        return self._stage_control.get_widefield_z_limits()

    def _af_can_move_stage(self) -> bool:
        if not (self._motion_thread and self._motion_thread.is_connected):
            self._status_widget.show_message("Stage not connected", 4000)
            return False
        if not self._motion_thread.all_referenced:
            self._status_widget.show_message("Reference the stage first", 4000)
            return False
        return True

    def _reset_nanodrive_to_initial(self):
        """Park the NanoDrive at its configured initial position.

        Runs on the autofocus worker thread (never the GUI thread) as the
        prepare step of every COARSE Prusa Z scan. Without it the NanoDrive is
        still sitting wherever the previous point's fine autofocus left it, so
        the coarse scan is biased by that leftover fine offset and the fine
        scan that follows starts off-center in its range. Resetting first makes
        every coarse scan start from the same objective focus state.
        """
        nano = self._nanodrive
        if not (nano and getattr(nano, 'is_connected', False)):
            return
        z0 = float(self._config.nanodrive.initial_position)
        if nano.move_to(z=z0):
            logger.info(f"NanoDrive reset to initial position {z0:.1f} um "
                        f"before the Prusa autofocus scan")
        else:
            logger.warning(f"NanoDrive reset to {z0:.1f} um failed; "
                           f"the Prusa scan continues from its current Z")

    def _start_af(self, positions, move_fn, score_fn, settle, label,
                  plot_stage=None, on_finish=None, maximize=True,
                  focus_kind=None, prepare=None):
        """Launch an AutofocusWorker and wire its signals to the UI.

        ``on_finish`` runs on EVERY exit path, including the refusals below:
        the caller has already applied its autofocus spectrometer settings by
        this point, so returning without it would strand the device at the
        autofocus exposure.
        """
        if self._af_worker is not None and self._af_worker.isRunning():
            if on_finish is not None:
                on_finish()
            return  # one scan at a time
        if self._calibration_running or (
                self._acq_worker is not None and self._acq_worker.isRunning()):
            self._status_widget.show_message(
                "Autofocus unavailable during acquisition / calibration", 3000)
            if on_finish is not None:
                on_finish()
            return
        if plot_stage:
            self._spectrum_viewer.clear_focus_plot(plot_stage)
        self._status_widget.set_progress(0, label)

        worker = AutofocusWorker(positions, move_fn, score_fn,
                                 settle_s=settle, label=label, maximize=maximize,
                                 prepare_fn=prepare)
        worker.progress.connect(self._status_widget.set_progress)
        if plot_stage:
            worker.point.connect(
                lambda z, s, st=plot_stage:
                self._spectrum_viewer.add_focus_point(st, z, s))

        def _done(success: bool, best_z: float):
            self._status_widget.set_progress(0, "")
            self._status_widget.show_message(
                f"{label} complete. Best Z = {best_z:.3f}" if success
                else f"{label} stopped", 5000)
            if success and focus_kind:
                self._stage_control.set_focus_result(focus_kind, best_z)
            if on_finish is not None:
                on_finish()
            self._af_worker = None
            self._update_af_buttons()
        worker.finished_focus.connect(_done)

        self._af_worker = worker
        self._update_af_buttons()  # disable all AF buttons while scanning
        worker.start()

    def _on_camera_autofocus(self, num_steps: int):
        """Camera autofocus: step the Prusa Z and maximise image contrast."""
        if not self._af_can_move_stage():
            return
        # Camera autofocus uses the Prusa range for the current view.
        z_upper, z_lower = self._active_prusa_z_limits()
        if z_lower >= z_upper:
            self._status_widget.show_message(
                "Set a valid Z range in the Autofocus panel", 4000)
            return
        positions = np.linspace(z_lower, z_upper, max(2, num_steps))

        def move(z):
            return self._motion_thread.move_z_blocking(z, timeout=30.0)

        def score():
            frame = self._live_view.current_frame()
            return (AutofocusEngine.calculate_tenengrad(frame)
                    if frame is not None else 0.0)

        # Let stage vibration settle before grabbing the frame to score (the
        # Prusa Z step shakes the stage; too short and a blurry frame is scored).
        # The NanoDrive goes back to its initial position first so the coarse
        # scan always starts from the same objective focus state.
        self._start_af(positions, move, score,
                       settle=self._config.widefield_camera.af_settle_time,
                       label="Camera AF", maximize=True, focus_kind='cam_prusa',
                       prepare=self._reset_nanodrive_to_initial)

    def _on_prusa_autofocus(self, params: dict):
        """Spectrometer coarse autofocus: step the Prusa Z, maximise signal."""
        if not self._af_can_move_stage():
            return
        if not (self._spectrometer and self._spectrometer.is_connected):
            self._status_widget.show_message("Spectrometer not connected", 4000)
            return
        # Spectrometer Prusa autofocus uses the Prusa range for the current view.
        z_upper, z_lower = self._active_prusa_z_limits()
        if z_lower >= z_upper:
            self._status_widget.show_message(
                "Set a valid Z range in the Autofocus panel", 4000)
            return
        steps = max(2, int(params.get("coarse_steps", 20)))
        positions = np.linspace(z_lower, z_upper, steps)
        restore = self._apply_af_spectrometer_settings(params)
        reps = int(params.get("reps", 3))
        expo_ms = float(params.get("exposure_ms", 100))

        def move(z):
            return self._motion_thread.move_z_blocking(z, timeout=30.0)

        def score():
            return self._spectrum_focus_score(reps, expo_ms, stage='prusa')

        # Reset the NanoDrive to its initial position before the coarse scan,
        # otherwise it is still parked at the previous point's fine focus and
        # biases every score in this scan.
        self._start_af(positions, move, score, settle=0.05, label="Prusa AF",
                       plot_stage='prusa', on_finish=restore, maximize=False,
                       focus_kind='spec_prusa',
                       prepare=self._reset_nanodrive_to_initial)

    def _on_nanodrive_autofocus(self, params: dict):
        """Spectrometer fine autofocus: step the NanoDrive, maximise signal."""
        if not (self._nanodrive and self._nanodrive.is_connected):
            self._status_widget.show_message("NanoDrive not connected", 4000)
            return
        if not (self._spectrometer and self._spectrometer.is_connected):
            self._status_widget.show_message("Spectrometer not connected", 4000)
            return
        nano_upper, nano_lower = self._stage_control.get_nanodrive_limits()
        if nano_lower >= nano_upper:
            self._status_widget.show_message(
                "Set a valid NanoDrive range in the Autofocus panel", 4000)
            return
        steps = max(2, int(params.get("fine_steps", 10)))
        positions = np.linspace(nano_lower, nano_upper, steps)
        restore = self._apply_af_spectrometer_settings(params)
        reps = int(params.get("reps", 3))
        expo_ms = float(params.get("exposure_ms", 100))

        def move(z):
            self._nanodrive.move_to(z=z)
            return True

        def score():
            return self._spectrum_focus_score(reps, expo_ms, stage='nano')

        self._start_af(positions, move, score, settle=0.03, label="Nano AF",
                       plot_stage='nano', on_finish=restore, maximize=False,
                       focus_kind='spec_nano')

    def _apply_af_spectrometer_settings(self, params: dict):
        """Apply the autofocus exposure/power to the spectrometer; return a
        callable that puts the ACQUISITION exposure/power back.

        The restore reads the spectrometer panel's own Expo. Time / Power boxes
        AT RESTORE TIME instead of a value snapshotted here. Snapshotting read
        ``config.integration_time``, which the autofocus itself overwrites, so
        the device could be left sitting at the autofocus exposure; going back
        to what the panel shows is what the next manual capture expects.
        """
        sv = self._spectrum_viewer
        acq_power = sv.acquisition_power_mw()
        # Pause the free-running live view for the scan so only the scored
        # autofocus frames are taken (and plotted); _restore resumes it.
        self._spectrometer.hold_live()
        released = [False]
        self._spectrometer.set_integration_time(
            float(params.get("exposure_ms", 100)))
        self._spectrometer.set_laser_power_mw(
            float(params.get("power_mw", acq_power)))

        def _restore():
            expo = sv.acquisition_exposure_ms()
            power = sv.acquisition_power_mw()
            self._spectrometer.set_integration_time(expo)
            self._spectrometer.set_laser_power_mw(power)
            if not released[0]:  # release this scan's hold exactly once
                released[0] = True
                self._spectrometer.release_live()
            logger.info(f"Autofocus finished: spectrometer restored to the "
                        f"acquisition settings ({expo:.0f} ms, {power:.0f} mW)")
        return _restore

    def _spectrum_focus_score(self, reps: int, expo_ms: float,
                              stage: str = 'prusa') -> float:
        """Median of `reps` FRESH spectra (cosmic-ray rejection) → noodlepy
        ratio. Lower is better (the scan minimises).

        Uses the spectrometer thread's coordinated capture (acquire_frames) so
        every score is taken AFTER the stage/NanoDrive move has settled and AT
        the current integration time. Reading the cached free-running frame
        instead would score stale spectra (captured before the move / at the
        old exposure), which made repeated autofocus runs give alternating,
        unreliable best-Z results.
        """
        spec = self._spectrometer
        frames = spec.acquire_frames(max(1, reps))
        if frames is None or not len(frames):
            return float('inf')  # worst score for a minimised metric
        median = np.median(frames, axis=0)
        wl_nm = spec.wavelengths_nm
        try:
            spec_cfg = self._config.spectrometer
            # Metric follows the connected spectrometer: band ratio on the
            # Wasatch, spectral entropy on the Kymera (or as forced in Settings).
            return float(AutofocusEngine.spectrum_focus_score(
                spec.focus_metric, wl_nm, median,
                signal_band=spec_cfg.af_signal_band(stage),
                reference_band=spec_cfg.af_reference_band(stage),
                entropy_band=spec_cfg.af_entropy_band()))
        except Exception:  # noqa: BLE001
            return float('inf')

    def _on_ranges_registered(self):
        """Autofocus ranges were registered — unlock the autofocus buttons."""
        wf_up, wf_lo = self._stage_control.get_widefield_z_limits()
        obj_up, obj_lo = self._stage_control.get_objective_z_limits()
        self._af_registered = True
        self._update_af_buttons()
        # Unlock "Switch to Objective" now that a focus range is known.
        self._live_view.set_autofocus_registered(True)
        self._status_widget.show_message(
            f"Autofocus ranges registered. WF Z {wf_lo} to {wf_up} mm, "
            f"Obj Z {obj_lo} to {obj_up} mm", 4000)
        logger.info(
            f"Autofocus ranges registered: WF Z [{wf_lo}, {wf_up}], "
            f"Obj Z [{obj_lo}, {obj_up}]")

    def _on_registration_reset(self):
        """Autofocus ranges were invalidated (disconnect) — re-lock the buttons
        they gate so the user must register again. 'Switch to Objective' and the
        camera autofocus stay disabled until then."""
        self._af_registered = False
        self._update_af_buttons()
        self._live_view.set_autofocus_registered(False)
        logger.info("Autofocus ranges reset (must re-register after reconnect)")

    # --- Current-FoV acquisition --------------------------------------------
    def _disconnected_acquisition_devices(self) -> list:
        """Names of acquisition devices whose REAL connection is currently down.

        Checks the live hardware state of each device (not the cached Connect
        button flag), so a device that never connected or dropped out is caught
        before a run starts. Simulated devices report connected, so this does not
        block simulation runs."""
        devices = [
            ("Stage", self._motion_thread),
            ("Widefield camera", self._widefield_camera),
            ("Objective camera", self._objective_camera),
            ("Spectrometer", self._spectrometer),
            ("NanoDrive", self._nanodrive),
        ]
        return [name for name, dev in devices
                if dev is None or not getattr(dev, 'is_connected', False)]

    def _warn_if_devices_disconnected(self) -> bool:
        """Pop up a warning listing any disconnected acquisition device. Returns
        True if the caller should abort the start (something is disconnected),
        False when every device is connected."""
        missing = self._disconnected_acquisition_devices()
        if not missing:
            return False
        QMessageBox.warning(
            self,
            "Devices not connected",
            "Cannot start acquisition. The following device(s) are not "
            "connected:\n\n  •  " + "\n  •  ".join(missing) +
            "\n\nClick Connect and make sure every device is connected (and "
            "the stage referenced) before starting acquisition.")
        logger.warning(f"Acquisition blocked, disconnected devices: {missing}")
        return True

    def _on_start_current_fov(self, params: dict):
        """Build and start the Current-FoV acquisition worker over the points
        generated on the captured image."""
        def _bail(msg: str):
            # Surface the reason both ways: a transient status-bar message AND a
            # log line (so it isn't missed if the status bar has already faded).
            logger.warning(f"Current FoV not started: {msg}")
            self._status_widget.show_message(msg, 5000)

        logger.info("Current FoV start requested")
        if self._calibration_running:
            _bail("A calibration test is running")
            return
        if self._acq_worker is not None and self._acq_worker.isRunning():
            _bail("Acquisition already running")
            return
        if self._af_worker is not None and self._af_worker.isRunning():
            _bail("Finish the autofocus scan first")
            return

        points = list(getattr(self._live_view, '_generated_positions', []) or [])
        if not points:
            _bail("No sample points generated (capture → segment → generate "
                  "in the Live tab)")
            return
        # Verify every device is really connected (pops up a warning if not).
        if self._warn_if_devices_disconnected():
            return
        if not self._motion_thread.all_referenced:
            _bail("Stage not referenced (home/reference all axes first)")
            return

        cm = self._coordinate_manager
        # Stage limit getters return (upper, lower); the worker wants (lower, upper).
        wf_up, wf_lo = self._stage_control.get_widefield_z_limits()
        obj_up, obj_lo = self._stage_control.get_objective_z_limits()
        nano_up, nano_lo = self._stage_control.get_nanodrive_limits()

        ctx = AcquisitionContext(
            motion=self._motion_thread,
            spectrometer=self._spectrometer,
            nanodrive=self._nanodrive,
            frame_provider=self._live_view.current_frame,
            position_provider=lambda: self._current_position,
            wavelengths_nm_provider=lambda: self._spectrometer.wavelengths_nm,
            offset=(cm.offset_x, cm.offset_y, cm.offset_z),
            # Y offset varies linearly with widefield X (calibration curve).
            offset_y_provider=cm.offset_y_at,
            excitation_nm=self._config.spectrometer.excitation_wavelength_nm,
            # Focus-score bands (Settings -> Spectrometer), so a run scores
            # focus exactly like the interactive autofocus does.
            af_prusa_signal_band_nm=self._config.spectrometer.af_signal_band('prusa'),
            af_prusa_reference_band_nm=(
                self._config.spectrometer.af_reference_band('prusa')),
            af_nano_signal_band_nm=self._config.spectrometer.af_signal_band('nano'),
            af_nano_reference_band_nm=(
                self._config.spectrometer.af_reference_band('nano')),
            # Metric per connected spectrometer (ratio / entropy), its band,
            # and the spectrometer identity for the metadata files.
            af_metric=lambda: self._spectrometer.focus_metric,
            af_entropy_band_nm=self._config.spectrometer.af_entropy_band(),
            spectrometer_metadata_provider=self._spectrometer.metadata,
            wf_prusa_range=(wf_lo, wf_up),
            obj_prusa_range=(obj_lo, obj_up),
            nano_range=(nano_lo, nano_up),
            af_settle_s=self._config.widefield_camera.af_settle_time,
            af_capture_stabilization_s=(
                self._config.widefield_camera.af_capture_stabilization_time),
            # The Live tab's captured image + mask, saved once for this sample.
            # The points (self._points above) already include any manually added
            # ones, so the saved overlay shows them too.
            capture_image=getattr(self._live_view, '_captured_image', None),
            capture_mask=getattr(self._live_view, '_current_mask', None),
            pixel_size_um=(self._config.widefield_camera.pixel_size_x_um or 1.0),
            # Current FoV file names drop sample/droplet/position (start at the
            # pattern), so no droplet identity is passed here.
        )

        worker = AcquisitionWorker(ctx, points, params)
        worker.progress.connect(self._protocol_panel.on_acquisition_progress)
        worker.status.connect(self._protocol_panel.on_acquisition_status)
        worker.status.connect(lambda m: self._status_widget.show_message(m, 3000))
        worker.error.connect(self._on_acq_error)
        worker.view_changed.connect(self._on_acq_view_changed)
        worker.finished_acq.connect(self._on_acq_finished)
        # Highlight the point being measured (green) on the captured-image
        # preview, synced with the acquisition steps.
        worker.measuring_point.connect(self._on_measuring_point)
        # Stream the per-point autofocus scores to the Live-tab focus plot,
        # just like the interactive spectrometer autofocus does.
        worker.focus_started.connect(self._spectrum_viewer.clear_focus_plot)
        worker.focus_point.connect(self._spectrum_viewer.add_focus_point)
        # Update the Live-tab autofocus panel's focus-Z labels as each scan ends.
        worker.focus_result.connect(self._stage_control.set_focus_result)
        # Let the worker drive the illumination LED (on for widefield autofocus,
        # off afterwards) through the LED control widget so the button stays in
        # sync. Queued because the worker lives on its own thread.
        worker.led_set.connect(self._led_control.set_on)
        self._acq_worker = worker
        self._protocol_panel.on_acquisition_started()
        # Lock the UI (everything but Pause/Stop) for the duration of the run.
        self._set_ui_locked(True)
        # The Live-tab sample map can't move the stage mid-run.
        self._sample_grid.set_acquisition_active(True)
        self._status_widget.show_message(
            f"Current FoV acquisition: {len(points)} points", 4000)
        logger.info(f"Starting Current-FoV acquisition over {len(points)} points")
        worker.start()

    def _on_start_robotic(self, params: dict):
        """Build and start the Robotic Deposition worker over the selected
        droplets. Each droplet is driven to, imaged + segmented, and its pattern
        generated + measured at run time (see RoboticDepositionWorker)."""
        def _bail(msg: str):
            logger.warning(f"Robotic Deposition not started: {msg}")
            self._status_widget.show_message(msg, 5000)

        logger.info("Robotic Deposition start requested")
        if self._calibration_running:
            _bail("A calibration test is running")
            return
        if self._acq_worker is not None and self._acq_worker.isRunning():
            _bail("Acquisition already running")
            return
        if self._af_worker is not None and self._af_worker.isRunning():
            _bail("Finish the autofocus scan first")
            return

        droplets = params.get('droplets') or []
        if not droplets:
            _bail("No droplets selected")
            return
        # Verify every device is really connected (pops up a warning if not).
        if self._warn_if_devices_disconnected():
            return
        if not self._motion_thread.all_referenced:
            _bail("Stage not referenced (home/reference all axes first)")
            return

        # Each selected well -> its LOGICAL widefield stage XY (no objective
        # offset; the worker moves in widefield view and the per-point objective
        # offset is applied in _measure_point). Same formula as the well-move.
        from raman_control.gui.widgets.sample_grid import well_name
        drop_cfg = self._config.sample_drop
        stage_cfg = self._config.stage
        wells = []
        for (row, col) in droplets:
            lx = (stage_cfg.home_x + drop_cfg.offset_home_to_a1_x
                  + row * drop_cfg.well_pitch_mm)
            ly = (stage_cfg.home_y + drop_cfg.offset_home_to_a1_y
                  + col * drop_cfg.well_pitch_mm)
            wells.append((well_name(row, col), lx, ly))

        # Ensure the U-Net is loaded and snapshot the Live-tab pattern settings
        # on the GUI thread; the worker then segments + generates off-thread.
        self._status_widget.show_message("Preparing segmentation model…", 3000)
        try:
            segmenter = self._live_view.get_segmenter()
        except Exception as e:  # noqa: BLE001
            _bail(f"Could not load segmentation model: {e}")
            return
        pattern_settings = self._live_view.snapshot_pattern_settings()

        cm = self._coordinate_manager
        wf_up, wf_lo = self._stage_control.get_widefield_z_limits()
        obj_up, obj_lo = self._stage_control.get_objective_z_limits()
        nano_up, nano_lo = self._stage_control.get_nanodrive_limits()

        ctx = AcquisitionContext(
            motion=self._motion_thread,
            spectrometer=self._spectrometer,
            nanodrive=self._nanodrive,
            # Segmentation + widefield AF must always read the WIDEFIELD camera,
            # independent of which view the display happens to be showing (the
            # worker toggles views per droplet), so read that thread directly.
            frame_provider=self._widefield_camera.get_latest_frame,
            position_provider=lambda: self._current_position,
            wavelengths_nm_provider=lambda: self._spectrometer.wavelengths_nm,
            offset=(cm.offset_x, cm.offset_y, cm.offset_z),
            # Y offset varies linearly with widefield X (calibration curve).
            offset_y_provider=cm.offset_y_at,
            excitation_nm=self._config.spectrometer.excitation_wavelength_nm,
            # Focus-score bands (Settings -> Spectrometer), so a run scores
            # focus exactly like the interactive autofocus does.
            af_prusa_signal_band_nm=self._config.spectrometer.af_signal_band('prusa'),
            af_prusa_reference_band_nm=(
                self._config.spectrometer.af_reference_band('prusa')),
            af_nano_signal_band_nm=self._config.spectrometer.af_signal_band('nano'),
            af_nano_reference_band_nm=(
                self._config.spectrometer.af_reference_band('nano')),
            # Metric per connected spectrometer (ratio / entropy), its band,
            # and the spectrometer identity for the metadata files.
            af_metric=lambda: self._spectrometer.focus_metric,
            af_entropy_band_nm=self._config.spectrometer.af_entropy_band(),
            spectrometer_metadata_provider=self._spectrometer.metadata,
            wf_prusa_range=(wf_lo, wf_up),
            obj_prusa_range=(obj_lo, obj_up),
            nano_range=(nano_lo, nano_up),
            af_settle_s=self._config.widefield_camera.af_settle_time,
            af_capture_stabilization_s=(
                self._config.widefield_camera.af_capture_stabilization_time),
            pixel_size_um=(self._config.widefield_camera.pixel_size_x_um or 1.0),
            segmenter=segmenter,
            pattern_settings=pattern_settings,
            # Per-droplet sample name + number for the structured file names
            # (the worker also uses each droplet's well as its position).
            droplet_map=self._sample_grid.droplet_map(),
            slide_change_z_clearance=self._config.stage.slide_change_z_clearance,
            stage_z_max=self._config.stage.z_max,
        )

        worker = RoboticDepositionWorker(ctx, wells, params)
        worker.progress.connect(self._protocol_panel.on_acquisition_progress)
        worker.status.connect(self._protocol_panel.on_acquisition_status)
        worker.status.connect(lambda m: self._status_widget.show_message(m, 3000))
        worker.error.connect(self._on_acq_error)
        worker.view_changed.connect(self._on_acq_view_changed)
        worker.finished_acq.connect(self._on_acq_finished)
        worker.measuring_point.connect(self._on_measuring_point)
        worker.focus_started.connect(self._spectrum_viewer.clear_focus_plot)
        worker.focus_point.connect(self._spectrum_viewer.add_focus_point)
        worker.focus_result.connect(self._stage_control.set_focus_result)
        worker.led_set.connect(self._led_control.set_on)
        # Show each droplet's captured image + segmentation + points as the robot
        # processes it, so it's clear the run moves between distinct droplets.
        worker.droplet_segmented.connect(self._on_droplet_segmented)
        # Highlight the droplet currently being measured on the sample map.
        worker.droplet_active.connect(self._on_droplet_active)
        # During the upfront capture phase, highlight the droplet being captured
        # WITHOUT a check mark (it has not been measured yet).
        worker.droplet_capturing.connect(self._on_droplet_capturing)
        self._acq_worker = worker
        self._protocol_panel.on_acquisition_started()
        self._set_ui_locked(True)
        # Switch BOTH sample maps (Live + Auto tab) to the same acquisition
        # display over the acquired droplets (measured / current / to-measure /
        # unselected), and make them non-interactive.
        self._sample_grid.set_acquisition_active(True, selected=droplets)
        auto_grid = getattr(self._protocol_panel, '_sample_grid', None)
        if auto_grid is not None:
            auto_grid.set_acquisition_active(True, selected=droplets)
        self._status_widget.show_message(
            f"Robotic Deposition: {len(wells)} droplets", 4000)
        logger.info(f"Starting Robotic Deposition over {len(wells)} droplets")
        worker.start()

    def _on_droplet_segmented(self, payload: dict):
        """Show the droplet the Robotic Deposition worker just imaged + segmented
        on the captured-image preview (Live tab) and the Current-FoV preview
        (Auto Acquisition tab), so the per-droplet progress is visible."""
        self._live_view.show_external_capture(
            payload.get('image'),
            payload.get('mask'),
            payload.get('positions') or [],
            payload.get('capture_xyz') or (0.0, 0.0, 0.0),
        )

    def _on_measuring_point(self, idx: int):
        """The worker is measuring point ``idx`` (0-based): show it in green on
        the captured-image preview (and its Auto-tab mirror)."""
        self._live_view.set_active_point(idx + 1)

    def _on_droplet_active(self, name: str):
        """The Robotic worker moved to droplet ``name``: mark it current on BOTH
        sample maps (the previous one becomes 'measured')."""
        from raman_control.gui.widgets.sample_grid import well_rc
        rc = well_rc(name)
        if rc is None:
            return
        self._sample_grid.set_measuring_well(*rc)
        auto_grid = getattr(self._protocol_panel, '_sample_grid', None)
        if auto_grid is not None:
            auto_grid.set_measuring_well(*rc)

    def _on_droplet_capturing(self, name: str):
        """Upfront capture phase: highlight droplet ``name`` as being captured on
        BOTH sample maps WITHOUT a check mark (it is imaged, not yet measured)."""
        from raman_control.gui.widgets.sample_grid import well_rc
        rc = well_rc(name)
        if rc is None:
            return
        self._sample_grid.set_capturing_well(*rc)
        auto_grid = getattr(self._protocol_panel, '_sample_grid', None)
        if auto_grid is not None:
            auto_grid.set_capturing_well(*rc)

    def _acq_running(self) -> bool:
        """True while a Current-FoV acquisition worker is active."""
        return self._acq_worker is not None and self._acq_worker.isRunning()

    def _calibration_busy_reason(self):
        """Why a calibration test cannot start now (None if it can)."""
        if self._acq_running():
            return "an acquisition is running"
        if self._af_worker is not None and self._af_worker.isRunning():
            return "an autofocus scan is running"
        return None

    def _on_calibration_running(self, running: bool):
        """Lock everything except the calibration Abort / E-stop / event log
        while a test runs, so no other stage move can be issued."""
        if running == self._calibration_running:
            return
        self._calibration_running = running
        self._set_ui_locked(
            running, keep=self._calibration_panel.controls_enabled_while_running())
        self._update_af_buttons()

    def _set_ui_locked(self, locked: bool, keep=()):
        """Lock the UI during an acquisition: every button / entry box / slider /
        combo is disabled EXCEPT Pause and Stop. Display widgets (camera views,
        spectrum & focus plots, the droplet map) are NOT disabled, so they keep
        updating live. The previous enabled-state is snapshotted and restored on
        unlock, so controls that were already disabled (e.g. gated autofocus
        buttons) stay disabled afterwards."""
        from PyQt6.QtWidgets import (
            QAbstractButton, QAbstractSpinBox, QSlider, QLineEdit, QComboBox)
        input_types = (QAbstractButton, QAbstractSpinBox, QSlider,
                       QLineEdit, QComboBox)
        pnl = self._protocol_panel
        # The panel owns Start/Pause/Stop via on_acquisition_started/finished.
        # The laser on/off button stays usable during a run (safety: the user
        # must always be able to kill the laser).
        owned = {getattr(pnl, '_start_btn', None),
                 getattr(pnl, '_pause_btn', None),
                 getattr(pnl, '_stop_btn', None),
                 getattr(self._spectrum_viewer, '_laser_btn', None)}
        owned |= set(keep)

        from PyQt6.QtCore import Qt
        if locked:
            self._ui_lock_snapshot = {}
            for w in self.findChildren(QWidget):
                if w in owned or not isinstance(w, input_types):
                    continue
                # Snapshot the widget's OWN enabled intent, not isEnabled()
                # (which is False whenever an ancestor group is disabled). A
                # view-gated control inside a group the run disables (e.g. the
                # pattern entry boxes when the run switches to objective view)
                # must be restored to its own state so it re-enables once its
                # group is re-enabled — otherwise it stays stuck disabled.
                own_enabled = not w.testAttribute(
                    Qt.WidgetAttribute.WA_ForceDisabled)
                self._ui_lock_snapshot[w] = own_enabled
                w.setEnabled(False)
        else:
            for w, enabled in self._ui_lock_snapshot.items():
                try:
                    w.setEnabled(enabled)
                except RuntimeError:
                    pass  # widget was deleted
            self._ui_lock_snapshot = {}

    def _on_acq_pause(self):
        if self._acq_worker is not None and self._acq_worker.isRunning():
            self._acq_worker.pause()
            self._protocol_panel.on_acquisition_paused(True)
            self._status_widget.show_message("Acquisition paused", 3000)

    def _on_acq_resume(self):
        if self._acq_worker is not None and self._acq_worker.isRunning():
            self._acq_worker.resume()
            self._protocol_panel.on_acquisition_paused(False)
            self._status_widget.show_message("Acquisition resumed", 3000)

    def _on_acq_stop(self):
        if self._acq_worker is not None and self._acq_worker.isRunning():
            self._acq_worker.abort()
            self._status_widget.show_message("Stopping acquisition…", 3000)

    def _on_acq_error(self, message: str):
        logger.error(f"Acquisition: {message}")
        self._status_widget.show_message(f"⚠ {message}", 5000)

    def _on_acq_view_changed(self, view: str):
        """The worker switched the active optical path. Update the coordinate
        manager, stage panel and displayed camera on the GUI thread (the worker
        has already moved the stage)."""
        target = CameraView.OBJECTIVE if view == 'objective' else CameraView.WIDE_FOV
        if self._coordinate_manager.current_view != target:
            self._coordinate_manager.switch_view(target)
            self._obj_target_slide = None
            # The UI is locked during a run, so don't re-enable stage / AF
            # controls here; they're re-synced to the final view on finish.
            if not (self._acq_worker is not None and self._acq_worker.isRunning()):
                self._stage_control.on_camera_view_changed()
                self._update_af_buttons()
        self._live_view.switch_camera('objective' if view == 'objective' else 'wide')

    def _on_acq_finished(self, aborted: bool, n_saved: int):
        self._protocol_panel.on_acquisition_finished(aborted)
        # Clear the green current-acquisition point highlight.
        self._live_view.set_active_point(None)
        # Return the sample maps to their interactive selection display.
        self._sample_grid.set_acquisition_active(False)
        auto_grid = getattr(self._protocol_panel, '_sample_grid', None)
        if auto_grid is not None:
            auto_grid.set_acquisition_active(False)
        # Re-enable the UI that was locked for the run.
        self._set_ui_locked(False)
        # Let the worker's run() fully return before dropping our reference, so
        # the QThread object isn't garbage-collected while the thread is still
        # running (which would crash with "QThread: Destroyed while running").
        worker = self._acq_worker
        self._acq_worker = None
        if worker is not None:
            worker.wait(5000)
        # The run leaves the system in objective view, where the Live tab's
        # capture / segmentation / pattern controls (and their entry boxes) are
        # greyed out. Return to widefield so they're usable again.
        #
        # First try the proper offset move-back (same as the camera-switch
        # button). NOTE: that path REFUSES to switch if its safety check fails
        # (e.g. the stage is in a bad state after a stop / brown-out), which
        # would strand the Live controls disabled. So afterwards we GUARANTEE the
        # Live tab ends in a usable widefield state: if the display isn't already
        # wide, force the view bookkeeping + display to widefield (display-only,
        # no move) so the controls are re-enabled regardless.
        if self._coordinate_manager.current_view != CameraView.WIDE_FOV:
            self._on_camera_switch_requested('wide')
        if self._live_view.current_camera != 'wide':
            logger.warning("Return-to-widefield move was blocked; forcing the "
                           "Live view to widefield so its controls are usable.")
            if self._coordinate_manager.current_view != CameraView.WIDE_FOV:
                self._coordinate_manager.switch_view(CameraView.WIDE_FOV)
                self._obj_target_slide = None
            self._live_view.switch_camera('wide')
        # Re-sync stage-panel Z-box editability and AF buttons to the final view.
        self._stage_control.on_camera_view_changed()
        self._update_af_buttons()

        # Current FoV: return the stage to the captured-image center (where it
        # started) so it doesn't end parked on the last measured point. Done
        # AFTER the switch back to widefield (the capture position is a widefield
        # position). Skipped on abort/stop, where the user likely stopped to
        # inspect the current spot. Robotic Deposition has no single capture, so
        # it is skipped there too.
        if (not aborted and worker is not None
                and not isinstance(worker, RoboticDepositionWorker)
                and getattr(self._live_view, '_captured_image', None) is not None):
            cap_pos = getattr(self._live_view, '_capture_stage_position', None)
            if (cap_pos and self._coordinate_manager.current_view == CameraView.WIDE_FOV
                    and self._motion_thread and self._motion_thread.is_connected):
                self._motion_thread.request_move_to(
                    x=cap_pos[0], y=cap_pos[1], z=cap_pos[2])
                logger.info(
                    "Current FoV: returning stage to capture center "
                    f"({cap_pos[0]:.3f}, {cap_pos[1]:.3f}, {cap_pos[2]:.3f})")

        # Re-illuminate the sample for live viewing (the run turned the LED off
        # for the Raman measurements). Done last, after the Live controls +
        # widefield view are restored, so the Live view is usable and lit. This
        # runs on both a normal completion and a mid-run stop, so the LED is
        # always back on once auto acquisition ends.
        self._led_control.set_on(True)

        msg = (f"Acquisition stopped ({n_saved} spectra saved)" if aborted
               else f"Acquisition complete: {n_saved} spectra saved")
        self._status_widget.show_message(msg, 6000)
        logger.info(msg)

    def _on_positions_generated(self, positions: list):
        """Handle sample positions from live view segmentation.

        Applies the wide-to-objective offset so each SamplePosition has
        both widefield (stage_x/y/z) and objective (obj_stage_x/y/z) coordinates.
        """
        for pos in positions:
            obj_x, obj_y, obj_z = self._coordinate_manager.get_objective_position(
                pos.stage_x, pos.stage_y, pos.stage_z
            )
            pos.obj_stage_x = obj_x
            pos.obj_stage_y = obj_y
            pos.obj_stage_z = obj_z

        cm = self._coordinate_manager
        dy_note = (f"dy=curve @ each point's X"
                   if cm.offset_curve_active else f"dy={cm.offset_y}")
        logger.info(
            f"Received {len(positions)} sample positions "
            f"(objective offset applied: dx={cm.offset_x}, {dy_note}, "
            f"dz={cm.offset_z})"
        )

    def _on_move_to_point(self, logical_x: float, logical_y: float, logical_z: float):
        """Move the stage to a clicked pattern point.

        The point arrives in the widefield (logical) frame. ``logical_to_physical``
        converts it for the current view: in wide FOV it's used as-is; in
        objective view the wide->obj offset is added so the objective is
        placed over the same sample point. Z is left at the current value
        (XY-only move, matching well selection).
        """
        if self._acq_running() or self._calibration_running:
            msg = "Move-to-point is disabled during acquisition / calibration"
            logger.info(msg)
            self._status_widget.show_message(msg, 3000)
            return
        if not self._motion_thread or not self._motion_thread.is_connected:
            self._status_widget.show_message("Stage not connected", 3000)
            return
        if not self._motion_thread.all_referenced:
            self._status_widget.show_message("Stage not referenced", 3000)
            return

        cur_z = self._current_position[2]
        phys_x, phys_y, phys_z = self._coordinate_manager.logical_to_physical(
            logical_x, logical_y, cur_z
        )
        view = self._coordinate_manager.current_view.name
        logger.info(
            f"Move to pattern point: logical=({logical_x:.3f}, {logical_y:.3f}), "
            f"physical=({phys_x:.3f}, {phys_y:.3f}), view={view}"
        )
        self._status_widget.show_message(
            f"Moving to point ({logical_x:.2f}, {logical_y:.2f}) [{view}]..."
        )
        self._manual_move(phys_x, phys_y)

    def _on_well_clicked(self, row: int, col: int):
        """Move stage to the clicked well position.

        Computes the logical (wide-FOV frame) position of the well from
        the plate config, then converts to physical stage coordinates
        using ``logical_to_physical`` which accounts for the current
        camera view (wide FOV vs objective offset).
        """
        if self._acq_running() or self._calibration_running:
            msg = "Well move is disabled during acquisition / calibration"
            logger.info(msg)
            self._status_widget.show_message(msg, 3000)
            return
        if not self._motion_thread or not self._motion_thread.is_connected:
            self._status_widget.show_message("Stage not connected", 3000)
            return
        if not self._motion_thread.all_referenced:
            self._status_widget.show_message("Stage not referenced", 3000)
            return

        drop_cfg = self._config.sample_drop
        stage_cfg = self._config.stage

        # Well position relative to the home position on the uniform 384
        # well plate grid. row/col are 0-based indices (A=0, col1=0).
        # The row letter steps the X axis; the column number steps the Y
        # axis. Distance from home grows with both. Because positions are
        # anchored to home, recalibrating home moves the whole grid with it.
        logical_x = (stage_cfg.home_x + drop_cfg.offset_home_to_a1_x
                     + row * drop_cfg.well_pitch_mm)
        logical_y = (stage_cfg.home_y + drop_cfg.offset_home_to_a1_y
                     + col * drop_cfg.well_pitch_mm)
        logical_z = self._current_position[2]  # keep current Z

        # Convert to physical coords (adds objective offset if in objective view)
        phys_x, phys_y, phys_z = self._coordinate_manager.logical_to_physical(
            logical_x, logical_y, logical_z
        )

        from raman_control.gui.widgets.sample_grid import well_name
        name = well_name(row, col)
        logger.info(f"Moving to well {name}: logical=({logical_x:.2f}, {logical_y:.2f}), "
                    f"physical=({phys_x:.2f}, {phys_y:.2f}), "
                    f"view={self._coordinate_manager.current_view.name}")
        self._status_widget.show_message(f"Moving to well {name}...")
        self._manual_move(phys_x, phys_y)

    def _manual_move(self, phys_x: float, phys_y: float,
                     phys_z: Optional[float] = None) -> None:
        """Queue a manual stage move (well click, pattern-point click, Go).

        In objective view, if the target is on a different slide than the
        objective is over (or either is unknown), hop over the barrier between
        slides: lift Z to the objective "Highest" + clearance, travel XY, lower
        to "Highest", then (if ``phys_z`` is given) go to ``phys_z``. All moves
        are fire-and-forget; the motion thread executes them in order, so the
        GUI thread never waits on the stage. Widefield moves are unchanged."""
        cm = self._coordinate_manager
        if cm.current_view != CameraView.OBJECTIVE:
            self._motion_thread.request_move_to(x=phys_x, y=phys_y, z=phys_z)
            return

        stage_cfg, drop_cfg = self._config.stage, self._config.sample_drop
        cur = self._current_position
        # Slides are defined on the widefield (logical) grid; convert both the
        # objective's current and target positions back to it.
        cur_lx, cur_ly, _ = cm.physical_to_logical(cur[0], cur[1], cur[2])
        tgt_lx, tgt_ly, _ = cm.physical_to_logical(phys_x, phys_y, cur[2])
        cur_slide = slide_at_position(cur_lx, cur_ly, stage_cfg, drop_cfg)
        tgt_slide = slide_at_position(tgt_lx, tgt_ly, stage_cfg, drop_cfg)
        # The cached position lags queued moves, so also compare against the
        # last commanded target; hop unless both agree with the target.
        same_slide = (tgt_slide is not None and cur_slide == tgt_slide
                      and self._obj_target_slide == tgt_slide)
        if same_slide:
            self._motion_thread.request_move_to(x=phys_x, y=phys_y, z=phys_z)
            return

        obj_hi, obj_lo = self._stage_control.get_objective_z_limits()
        if not obj_hi > obj_lo:
            msg = ("Move blocked: set the objective Z limits (Highest > Lowest) "
                   "so the objective can lift over the slide barrier")
            logger.warning(msg)
            self._status_widget.show_message(msg, 5000)
            return

        moves = plan_slide_hop(phys_x, phys_y, cur[2], obj_hi,
                               stage_cfg.slide_change_z_clearance,
                               stage_cfg.z_max)
        src = "?" if cur_slide is None else str(cur_slide)
        dst = "?" if tgt_slide is None else str(tgt_slide)
        logger.info(
            f"Manual slide hop {src} -> {dst}: Z up to {moves[0][2]:.3f}, "
            f"XY to ({phys_x:.3f}, {phys_y:.3f}), Z down to {obj_hi:.3f}"
            + (f", then Z {phys_z:.3f}" if phys_z is not None else ""))
        self._status_widget.show_message(
            f"Slide {src} -> {dst}: lifting objective over barrier...")
        for x, y, z in moves:
            self._motion_thread.request_move_to(x=x, y=y, z=z)
        if phys_z is not None:
            self._motion_thread.request_move_to(z=phys_z)
        self._obj_target_slide = tgt_slide

    def _on_open_settings(self):
        """Open the hardware settings dialog."""
        dlg = SettingsDialog(self._config, parent=self)
        if dlg.exec():
            # Pixel-size (and other) settings may have changed — refresh the
            # live view's scale bar and proximity marker, which derive their
            # size from the pixel calibration.
            self._live_view.refresh_calibration()
            # Kymera bin rows apply live (queued to the spectrum thread); the
            # spectrum panel's row boxes / image lines follow the config.
            self._spectrometer.apply_readout_config()
            self._spectrum_viewer.refresh_bin_rows()

    def _show_about(self):
        QMessageBox.about(
            self,
            "About Raman Microscopy Control",
            "<h2>Raman Microscopy Control</h2>"
            "<p><b>Version 2.0</b></p>"
            "<p>Vision-guided Raman microscopy<br>for liquid biopsy.</p>"
        )

    def showEvent(self, event):
        """Set the initial Live View / Spectrometer split once, after the
        window is realized so child widths (and the live view's content
        height) are known.

        The top block is sized to the live view's full content height (with
        the Ring pattern selected), so everything fits without the inner
        panel scrolling; the spectrometer takes the remaining space.
        """
        super().showEvent(event)
        if self._default_split_applied:
            return
        self._default_split_applied = True

        live_h = self._live_view.preferred_content_height()
        # Default the split so the live-view block shows its full content, but
        # do NOT hard-pin its minimum: the spectrometer block (with the
        # autofocus panel) also needs room, so pinning would make the two
        # minimums exceed the window and the live view would overlap the
        # spectrum block. The live view's inner scroll area covers the rare
        # case where the window is too short for both.
        total = self._view_splitter.height()
        spectrum_h = max(260, total - live_h)  # keep the spectrum usable
        self._view_splitter.setSizes([live_h, spectrum_h])

    def closeEvent(self, event: QCloseEvent):
        reply = QMessageBox.question(
            self, "Confirm Exit",
            "Exit? All hardware will be disconnected.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._shutdown_hardware()
            event.accept()
        else:
            event.ignore()

    def _shutdown_hardware(self):
        """Disconnect all hardware cleanly on exit. Idempotent and
        error-isolated: each device is shut down in its own try/except so one
        failure can't skip the rest — in particular the LED serial port MUST be
        closed gracefully every time, or the CH340 comes back in the Windows
        'device not functioning' state and needs a physical re-plug.

        Called from closeEvent and (as a backstop) from QApplication.aboutToQuit.
        """
        if self._shutdown_done:
            return
        self._shutdown_done = True
        logger.info("Closing, shutting down hardware...")

        # Stop the acquisition / autofocus workers first.
        for worker, wait_ms in ((self._acq_worker, 5000),
                                (self._af_worker, 2000)):
            try:
                if worker is not None and worker.isRunning():
                    worker.abort()
                    worker.wait(wait_ms)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Worker shutdown error: {e}")

        try:
            self._calibration_panel.shutdown()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Calibration shutdown error: {e}")

        # Stop each hardware thread independently so one hang/error can't
        # prevent the others (notably the LED) from closing cleanly.
        for thread in (self._motion_thread, self._widefield_camera,
                       self._objective_camera, self._spectrometer,
                       self._led_thread):
            try:
                if thread is not None and thread.isRunning():
                    thread.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Thread shutdown error: {e}")

        try:
            if self._nanodrive and self._nanodrive.is_connected:
                self._nanodrive.disconnect()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"NanoDrive shutdown error: {e}")

        # Dispose the shared Thorlabs SDK exactly once, now that all camera
        # threads have stopped and the process is exiting.
        try:
            from raman_control.gui.threads.camera_thread import (
                ThorlabsCameraManager,
            )
            ThorlabsCameraManager.get_instance().dispose()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Camera SDK dispose error: {e}")

        logger.info("Hardware shutdown complete")


def main():
    setup_logging()
    logger.info("Starting application")

    app = QApplication(sys.argv)
    apply_dark_theme(app)

    # Tooltips (button / entry-box / label hints) should appear almost
    # immediately on hover instead of after Qt's ~700 ms default.
    install_fast_tooltips(app)

    # Block the mouse wheel from changing spin box / combo box values
    # anywhere in the app, values must be typed or set via the arrows.
    install_wheel_guard(app)

    config = HardwareConfig()
    window = MainWindow(config)
    window.show()

    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
