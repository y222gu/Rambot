"""
Main window for the Raman microscopy control application.

Modern dark theme PyQt6 application with:
- Stage control panel (includes autofocus)
- Live view with integrated segmentation
- Spectrum display
- Protocol control
"""

import sys
from typing import Optional
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QMessageBox, QSplitter, QFrame, QLabel,
    QToolBar, QSizePolicy, QProgressBar
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QCloseEvent

from raman_control.config.logging_config import setup_logging, get_logger
from raman_control.config.hardware_config import HardwareConfig
from raman_control.gui.styles import DARK_STYLESHEET, COLORS, apply_dark_theme
from raman_control.gui.threads.motion_thread import MotionThread
from raman_control.gui.threads.camera_thread import CameraThread
from raman_control.gui.threads.spectrum_thread import SpectrumThread
from raman_control.gui.widgets.stage_control import StageControlWidget
from raman_control.gui.widgets.live_view import LiveViewWidget
from raman_control.gui.widgets.spectrum_viewer import SpectrumViewerWidget
from raman_control.gui.widgets.protocol_panel import ProtocolPanelWidget
from raman_control.gui.widgets.sample_grid import SampleGridWidget
from raman_control.core.state_machine import AcquisitionStateMachine
from raman_control.core.coordinate_manager import CoordinateManager, CameraView


logger = get_logger('main_window')


class DeviceIndicator(QWidget):
    """Single device connection indicator with LED-style light."""

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._led = QLabel()
        self._led.setFixedSize(10, 10)
        self._set_status(False)

        self._label = QLabel(name)
        self._label.setStyleSheet(f'color: {COLORS["text_secondary"]}; font-size: 11px;')

        layout.addWidget(self._led)
        layout.addWidget(self._label)
        self._connected = False

    def _set_status(self, connected: bool):
        color = COLORS['success'] if connected else '#5c5c5c'
        border = '#2e7d32' if connected else '#3d3d3d'
        self._led.setStyleSheet(f'''
            background-color: {color};
            border: 1px solid {border};
            border-radius: 5px;
        ''')

    def set_connected(self, connected: bool):
        self._connected = connected
        self._set_status(connected)
        color = COLORS['text_primary'] if connected else COLORS['text_secondary']
        self._label.setStyleSheet(f'color: {color}; font-size: 11px;')


class StatusBarWidget(QWidget):
    """Custom status bar with device indicators and single progress bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(12)

        # Device indicators
        devices_layout = QHBoxLayout()
        devices_layout.setSpacing(16)

        self._device_indicators = {}
        for device in ['Stage', 'Wide Cam', 'Obj Cam', 'Spectrometer']:
            indicator = DeviceIndicator(device)
            self._device_indicators[device] = indicator
            devices_layout.addWidget(indicator)

        layout.addLayout(devices_layout)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f'background-color: {COLORS["border_dark"]};')
        layout.addWidget(sep)

        # Position display
        self._position_label = QLabel("X: —  Y: —  Z: —")
        self._position_label.setStyleSheet(f'''
            font-family: "SF Mono", "Consolas", monospace;
            color: {COLORS['text_primary']};
            font-size: 12px;
        ''')
        layout.addWidget(self._position_label)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet(f'background-color: {COLORS["border_dark"]};')
        layout.addWidget(sep2)

        # Global progress bar
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

        layout.addStretch()

        # Status message
        self._message_label = QLabel("")
        self._message_label.setStyleSheet(f'color: {COLORS["text_secondary"]};')
        layout.addWidget(self._message_label)

    def set_device_connected(self, device: str, connected: bool):
        if device in self._device_indicators:
            self._device_indicators[device].set_connected(connected)

    def set_position(self, x: float, y: float, z: float):
        self._position_label.setText(f"X: {x:>8.2f}  Y: {y:>8.2f}  Z: {z:>8.2f}")

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
        self._coordinate_manager = CoordinateManager()

        self._setup_window()
        self._setup_toolbar()
        self._setup_central_widget()
        self._setup_statusbar()
        self._init_threads()

        logger.info("Main window initialized")

    def _setup_window(self):
        self.setWindowTitle("Raman Microscopy Control")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(DARK_STYLESHEET)

    def _setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setStyleSheet(f'''
            QToolBar {{
                background-color: {COLORS['bg_medium']};
                border-bottom: 1px solid {COLORS['border_dark']};
                padding: 4px;
                spacing: 8px;
            }}
            QToolButton {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 6px 12px;
                color: {COLORS['text_primary']};
            }}
            QToolButton:hover {{ background-color: {COLORS['bg_elevated']}; }}
            QToolButton:pressed {{ background-color: {COLORS['bg_light']}; }}
        ''')
        self.addToolBar(toolbar)

        connect_action = QAction("🔌 Connect", self)
        connect_action.triggered.connect(self._on_connect_all)
        toolbar.addAction(connect_action)

        disconnect_action = QAction("⏏ Disconnect", self)
        disconnect_action.triggered.connect(self._on_disconnect_all)
        toolbar.addAction(disconnect_action)

        toolbar.addSeparator()

        stop_action = QAction("⛔ STOP", self)
        stop_action.setShortcut("Escape")
        stop_action.triggered.connect(self._on_emergency_stop)
        toolbar.addAction(stop_action)

        toolbar.addSeparator()

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        about_action = QAction("ℹ About", self)
        about_action.triggered.connect(self._show_about)
        toolbar.addAction(about_action)

    def _setup_central_widget(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)

        # === Left Panel: Stage Control ===
        left_panel = QFrame()
        left_panel.setStyleSheet(f'''
            QFrame {{
                background-color: {COLORS['bg_dark']};
                border-right: 1px solid {COLORS['border_dark']};
            }}
        ''')
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._motion_thread = MotionThread(self._config.stage, self)

        self._stage_control = StageControlWidget(
            self._motion_thread,
            self._config.stage
        )
        left_layout.addWidget(self._stage_control)

        left_panel.setMinimumWidth(380)
        left_panel.setMaximumWidth(500)
        splitter.addWidget(left_panel)

        # === Right Panel: Tabs ===
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)

        self._tab_widget = QTabWidget()
        self._tab_widget.setDocumentMode(True)

        # Hardware threads
        self._widefield_camera = CameraThread(camera_id='widefield', config=self._config.widefield_camera)
        self._objective_camera = CameraThread(camera_id='objective', config=self._config.objective_camera)
        self._spectrometer = SpectrumThread(config=self._config.spectrometer)

        # State machine
        self._state_machine = AcquisitionStateMachine()

        # Live View tab (includes segmentation and pattern)
        self._live_view = LiveViewWidget(
            widefield_thread=self._widefield_camera,
            objective_thread=self._objective_camera
        )
        self._tab_widget.addTab(self._live_view, "Live View")

        # Spectrometer tab
        self._spectrum_viewer = SpectrumViewerWidget(
            spectrum_thread=self._spectrometer,
            config=self._config.spectrometer
        )
        self._tab_widget.addTab(self._spectrum_viewer, "Spectrometer")

        # Sample Grid tab
        self._sample_grid = SampleGridWidget()
        self._tab_widget.addTab(self._sample_grid, "Sample Grid")

        # Protocol tab
        self._protocol_panel = ProtocolPanelWidget(
            state_machine=self._state_machine
        )
        self._tab_widget.addTab(self._protocol_panel, "Protocol")

        right_layout.addWidget(self._tab_widget)
        splitter.addWidget(right_panel)

        splitter.setSizes([380, 1020])
        main_layout.addWidget(splitter)

    def _setup_statusbar(self):
        self._status_widget = StatusBarWidget()
        self.statusBar().addPermanentWidget(self._status_widget, 1)
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
        self._widefield_camera.connected.connect(
            lambda: self._status_widget.set_device_connected('Wide Cam', True))
        self._widefield_camera.disconnected.connect(
            lambda: self._status_widget.set_device_connected('Wide Cam', False))

        # Objective camera
        self._objective_camera.connected.connect(
            lambda: self._status_widget.set_device_connected('Obj Cam', True))
        self._objective_camera.disconnected.connect(
            lambda: self._status_widget.set_device_connected('Obj Cam', False))

        # Spectrometer
        self._spectrometer.connected.connect(
            lambda: self._status_widget.set_device_connected('Spectrometer', True))
        self._spectrometer.disconnected.connect(
            lambda: self._status_widget.set_device_connected('Spectrometer', False))

        # Live view camera switch
        self._live_view.camera_switch_requested.connect(self._on_camera_switch_requested)

        # Stage autofocus signal
        self._stage_control.autofocus_requested.connect(self._on_autofocus_requested)

        # Sample points from live view to grid
        self._live_view.positions_generated.connect(self._on_positions_generated)

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

    def _on_disconnect_all(self):
        logger.info("Disconnecting...")
        for thread in [self._motion_thread, self._widefield_camera,
                       self._objective_camera, self._spectrometer]:
            if thread and thread.isRunning():
                thread.stop()

    def _on_emergency_stop(self):
        logger.warning("EMERGENCY STOP")
        self._status_widget.show_message("⚠ EMERGENCY STOP", 5000)
        if self._motion_thread:
            self._motion_thread.request_stop()

    def _on_stage_connected(self):
        self._status_widget.set_device_connected('Stage', True)
        self._status_widget.show_message("Stage connected")

    def _on_stage_disconnected(self):
        self._status_widget.set_device_connected('Stage', False)

    def _on_position_changed(self, x: float, y: float, z: float):
        self._status_widget.set_position(x, y, z)

    def _on_stage_error(self, message: str):
        if "not connected" not in message.lower():
            self._status_widget.show_message(f"⚠ Stage: {message}", 5000)

    def _on_camera_switch_requested(self, camera_name: str):
        """Handle camera switch with safe stage movement."""
        target_view = CameraView.OBJECTIVE if camera_name == "objective" else CameraView.WIDE_FOV

        if self._coordinate_manager.current_view != target_view:
            current_pos = self._motion_thread.get_position() if self._motion_thread else None

            if current_pos:
                if target_view == CameraView.OBJECTIVE:
                    is_safe, msg = self._coordinate_manager.validate_objective_z(current_pos[2])
                    if not is_safe:
                        QMessageBox.warning(self, "Safety Warning", msg)
                        return

                offset = self._coordinate_manager.get_camera_switch_offset(target_view)
                if offset and self._motion_thread:
                    new_x = current_pos[0] + offset.x
                    new_y = current_pos[1] + offset.y
                    new_z = current_pos[2] + offset.z
                    self._status_widget.show_message(f"Moving for camera switch...")
                    self._motion_thread.move_to(new_x, new_y, new_z)

            self._coordinate_manager.current_view = target_view

        self._live_view.switch_camera(camera_name)
        logger.info(f"Camera switched to: {camera_name}")

    def _on_autofocus_requested(self, af_type: str, z_upper: float, z_lower: float):
        """Handle autofocus request from stage control."""
        self._status_widget.show_message(f"Running {af_type} autofocus...")
        self._status_widget.set_progress(0, f"{af_type} AF")
        # TODO: Connect to actual autofocus engine
        logger.info(f"Autofocus requested: {af_type}, Z range: [{z_lower}, {z_upper}]")

    def _on_positions_generated(self, positions: list):
        """Handle sample positions from live view segmentation."""
        logger.info(f"Received {len(positions)} sample positions")
        # Could update sample grid or protocol panel here

    def _show_about(self):
        QMessageBox.about(
            self,
            "About Raman Microscopy Control",
            "<h2>Raman Microscopy Control</h2>"
            "<p><b>Version 2.0</b></p>"
            "<p>Vision-guided Raman microscopy<br>for liquid biopsy.</p>"
        )

    def closeEvent(self, event: QCloseEvent):
        reply = QMessageBox.question(
            self, "Confirm Exit",
            "Exit? All hardware will be disconnected.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            logger.info("Closing...")
            for thread in [self._motion_thread, self._widefield_camera,
                           self._objective_camera, self._spectrometer]:
                if thread and thread.isRunning():
                    thread.stop()
            event.accept()
        else:
            event.ignore()


def main():
    setup_logging()
    logger.info("Starting application")

    app = QApplication(sys.argv)
    apply_dark_theme(app)

    config = HardwareConfig()
    window = MainWindow(config)
    window.show()

    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
