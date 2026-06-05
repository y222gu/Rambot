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
from raman_control.gui.threads.motion_thread import MotionThread
from raman_control.gui.threads.camera_thread import CameraThread
from raman_control.gui.threads.spectrum_thread import SpectrumThread
from raman_control.gui.threads.autofocus_thread import AutofocusWorker
from raman_control.gui.threads.led_thread import LEDThread
from raman_control.gui.widgets.stage_control import StageControlWidget
from raman_control.gui.widgets.led_control import LEDControlWidget
from raman_control.gui.widgets.live_view import LiveViewWidget
from raman_control.gui.widgets.spectrum_viewer import SpectrumViewerWidget
from raman_control.gui.widgets.protocol_panel import ProtocolPanelWidget
from raman_control.gui.widgets.sample_grid import SampleGridWidget
from raman_control.core.state_machine import AcquisitionStateMachine
from raman_control.core.autofocus import AutofocusEngine
from raman_control.core.coordinate_manager import CoordinateManager, CameraView
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
        self._set_status(connected)

        if connected and simulated:
            # Show "(SIM)" in orange next to device name
            self._label.setText(f'{self._name} <span style="color: {COLORS["warning"]};">(SIM)</span>')
            self._label.setStyleSheet(f'color: {COLORS["text_primary"]}; font-size: 11px;')
        else:
            self._label.setText(self._name)
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

        # Autofocus: buttons are gated until a Z range is registered, and only
        # one scan runs at a time.
        self._af_registered = False
        self._af_worker: Optional[AutofocusWorker] = None

        self._setup_window()
        self._setup_central_widget()
        self._setup_statusbar()
        self._init_threads()

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
        self._spectrometer = SpectrumThread(
            config=self._config.spectrometer,
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
        self._status_widget.add_right_widget(self._settings_btn, separator=True)

        # About is a flat, borderless element (matches the old toolbar action
        # look) rather than a bordered button.
        self._about_btn = QPushButton("ℹ About")
        self._about_btn.setMinimumHeight(28)
        self._about_btn.setStyleSheet(self._TOOLBAR_BTN_FLAT)
        self._about_btn.clicked.connect(self._show_about)
        self._status_widget.add_right_widget(self._about_btn)

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

        # Live view camera switch
        self._live_view.camera_switch_requested.connect(self._on_camera_switch_requested)

        # Stage reference status → enable/disable FOV switch buttons
        self._stage_control.referenced_changed.connect(self._on_referenced_changed)

        # Autofocus scan ranges registered from the stage panel
        self._stage_control.ranges_registered.connect(self._on_ranges_registered)

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
        else:
            self._connect_btn.setText("Connect")
            self._connect_btn.setStyleSheet(self._TOOLBAR_BTN_CONNECT)

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
        return {
            "stage_x_mm": round(x, 4),
            "stage_y_mm": round(y, 4),
            "stage_z_mm": round(z, 4),          # coarse / autofocus position
            "nanodrive_z_um": round(nano_z, 3) if nano_z is not None else None,
        }

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
        base = self._af_registered and self._af_worker is None
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

    def _start_af(self, positions, move_fn, score_fn, settle, label,
                  plot_stage=None, on_finish=None, maximize=True,
                  focus_kind=None):
        """Launch an AutofocusWorker and wire its signals to the UI."""
        if self._af_worker is not None and self._af_worker.isRunning():
            return  # one scan at a time
        if plot_stage:
            self._spectrum_viewer.clear_focus_plot(plot_stage)
        self._status_widget.set_progress(0, label)

        worker = AutofocusWorker(positions, move_fn, score_fn,
                                 settle_s=settle, label=label, maximize=maximize)
        worker.progress.connect(self._status_widget.set_progress)
        if plot_stage:
            worker.point.connect(
                lambda z, s, st=plot_stage:
                self._spectrum_viewer.add_focus_point(st, z, s))

        def _done(success: bool, best_z: float):
            self._status_widget.set_progress(0, "")
            self._status_widget.show_message(
                f"{label} complete — best Z = {best_z:.3f}" if success
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

        # Let stage vibration settle before grabbing the frame to score.
        self._start_af(positions, move, score, settle=0.5, label="Camera AF",
                       maximize=True, focus_kind='cam_prusa')

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
            return self._spectrum_focus_score(reps, expo_ms)

        self._start_af(positions, move, score, settle=0.05, label="Prusa AF",
                       plot_stage='prusa', on_finish=restore, maximize=False,
                       focus_kind='spec_prusa')

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
            return self._spectrum_focus_score(reps, expo_ms)

        self._start_af(positions, move, score, settle=0.03, label="Nano AF",
                       plot_stage='nano', on_finish=restore, maximize=False,
                       focus_kind='spec_nano')

    def _apply_af_spectrometer_settings(self, params: dict):
        """Apply the autofocus exposure/power to the spectrometer; return a
        callable that restores the acquisition settings afterward."""
        prev_expo = self._config.spectrometer.integration_time
        prev_power = self._config.spectrometer.laser_power_mw
        self._spectrometer.set_integration_time(float(params.get("exposure_ms", 100)))
        self._spectrometer.set_laser_power_mw(float(params.get("power_mw", prev_power)))

        def _restore():
            self._spectrometer.set_integration_time(prev_expo)
            self._spectrometer.set_laser_power_mw(prev_power)
        return _restore

    def _spectrum_focus_score(self, reps: int, expo_ms: float) -> float:
        """Median of `reps` spectra (cosmic-ray rejection) → noodlepy ratio.

        Runs inside the autofocus worker thread; uses only thread-safe
        accessors on the spectrum thread. Lower is better (the scan minimises).
        """
        dt = max(0.03, expo_ms / 1000.0)
        time.sleep(dt)  # let a fresh spectrum acquire after the move
        specs = []
        for _ in range(max(1, reps)):
            s = self._spectrometer.get_latest_spectrum()
            if s is not None:
                specs.append(np.asarray(s, dtype=float))
            time.sleep(dt)
        if not specs:
            return float('inf')  # worst score for a minimised metric
        median = np.median(np.array(specs), axis=0)
        wl_nm = self._spectrometer.wavelengths_nm
        return AutofocusEngine.calculate_ratio_score(wl_nm, median)

    def _on_ranges_registered(self):
        """Autofocus ranges were registered — unlock the autofocus buttons."""
        wf_up, wf_lo = self._stage_control.get_widefield_z_limits()
        obj_up, obj_lo = self._stage_control.get_objective_z_limits()
        self._af_registered = True
        self._update_af_buttons()
        # Unlock "Switch to Objective" now that a focus range is known.
        self._live_view.set_autofocus_registered(True)
        self._status_widget.show_message(
            f"Autofocus ranges registered — WF Z {wf_lo}–{wf_up} mm, "
            f"Obj Z {obj_lo}–{obj_up} mm", 4000)
        logger.info(
            f"Autofocus ranges registered: WF Z [{wf_lo}, {wf_up}], "
            f"Obj Z [{obj_lo}, {obj_up}]")

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

        logger.info(
            f"Received {len(positions)} sample positions "
            f"(objective offset applied: "
            f"dx={self._coordinate_manager.offset_x}, "
            f"dy={self._coordinate_manager.offset_y}, "
            f"dz={self._coordinate_manager.offset_z})"
        )

    def _on_move_to_point(self, logical_x: float, logical_y: float, logical_z: float):
        """Move the stage to a clicked pattern point.

        The point arrives in the widefield (logical) frame. ``logical_to_physical``
        converts it for the current view: in wide FOV it's used as-is; in
        objective view the wide->obj offset is added so the objective is
        placed over the same sample point. Z is left at the current value
        (XY-only move, matching well selection).
        """
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
        self._motion_thread.request_move_to(x=phys_x, y=phys_y)

    def _on_well_clicked(self, row: int, col: int):
        """Move stage to the clicked well position.

        Computes the logical (wide-FOV frame) position of the well from
        the plate config, then converts to physical stage coordinates
        using ``logical_to_physical`` which accounts for the current
        camera view (wide FOV vs objective offset).
        """
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
        self._motion_thread.request_move_to(x=phys_x, y=phys_y)

    def _on_open_settings(self):
        """Open the hardware settings dialog."""
        dlg = SettingsDialog(self._config, parent=self)
        if dlg.exec():
            # Pixel-size (and other) settings may have changed — refresh the
            # live view's scale bar and proximity marker, which derive their
            # size from the pixel calibration.
            self._live_view.refresh_calibration()

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
            logger.info("Closing...")
            if self._af_worker is not None and self._af_worker.isRunning():
                self._af_worker.abort()
                self._af_worker.wait(2000)
            for thread in [self._motion_thread, self._widefield_camera,
                           self._objective_camera, self._spectrometer,
                           self._led_thread]:
                if thread and thread.isRunning():
                    thread.stop()
            if self._nanodrive and self._nanodrive.is_connected:
                self._nanodrive.disconnect()
            # Dispose the shared Thorlabs SDK exactly once, now that all
            # camera threads have stopped and the process is exiting. The
            # SDK is kept alive across reconnects, so this is the only
            # place it gets torn down.
            from raman_control.gui.threads.camera_thread import (
                ThorlabsCameraManager,
            )
            ThorlabsCameraManager.get_instance().dispose()
            event.accept()
        else:
            event.ignore()


def main():
    setup_logging()
    logger.info("Starting application")

    app = QApplication(sys.argv)
    apply_dark_theme(app)

    # Block the mouse wheel from changing spin box / combo box values
    # anywhere in the app — values must be typed or set via the arrows.
    install_wheel_guard(app)

    config = HardwareConfig()
    window = MainWindow(config)
    window.show()

    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
