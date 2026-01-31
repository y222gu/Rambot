"""
Live camera view widget with integrated segmentation and sample pattern generation.

Features:
- Single camera display (wide FOV or objective)
- Safe camera switching with stage movement
- Image segmentation (SAM point prompt)
- Sample pattern generation (grid, ring, line, random)
- Overlay of detected area and sample points
"""

from typing import Optional, List, Tuple
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QFrame,
    QSizePolicy, QMessageBox, QButtonGroup, QRadioButton, QStackedWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush

from raman_control.gui.threads.camera_thread import CameraThread
from raman_control.gui.styles import COLORS
from raman_control.config.logging_config import get_logger

logger = get_logger('live_view')


class ImageDisplayWidget(QLabel):
    """Widget for displaying camera images with overlays (mask + sample points)."""

    clicked = pyqtSignal(int, int)  # x, y in image coordinates

    def __init__(self, parent=None):
        super().__init__(parent)
        self._show_crosshair = True
        self._current_image: Optional[np.ndarray] = None
        self._mask_overlay: Optional[np.ndarray] = None
        self._sample_points: List[Tuple[float, float]] = []

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f'''
            background-color: {COLORS['bg_light']};
            border: 1px solid {COLORS['border_dark']};
            border-radius: 4px;
        ''')
        self.setText("📷 No image")

    def update_image(self, image: np.ndarray):
        """Update the displayed image."""
        self._current_image = image
        self._redraw()

    def set_mask_overlay(self, mask: Optional[np.ndarray]):
        """Set mask overlay (binary mask)."""
        self._mask_overlay = mask
        if self._current_image is not None:
            self._redraw()

    def set_sample_points(self, points: List[Tuple[float, float]]):
        """Set sample points to display."""
        self._sample_points = points
        if self._current_image is not None:
            self._redraw()

    def clear_overlays(self):
        """Clear mask and sample points."""
        self._mask_overlay = None
        self._sample_points = []
        if self._current_image is not None:
            self._redraw()

    def _redraw(self):
        """Redraw image with all overlays."""
        if self._current_image is None:
            return

        image = self._current_image

        # Convert to QImage
        if image.ndim == 2:
            h, w = image.shape
            qimage = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, c = image.shape
            qimage = QImage(image.data, w, h, w * c, QImage.Format.Format_RGB888)

        # Scale to fit
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # Calculate scale factors
        scale_x = scaled.width() / w
        scale_y = scaled.height() / h

        # Draw overlays
        painter = QPainter(scaled)

        # Draw mask overlay (semi-transparent blue)
        if self._mask_overlay is not None:
            mask_color = QColor(30, 144, 255, 100)  # Dodger blue with alpha
            for y in range(min(self._mask_overlay.shape[0], h)):
                for x in range(min(self._mask_overlay.shape[1], w)):
                    if self._mask_overlay[y, x]:
                        sx, sy = int(x * scale_x), int(y * scale_y)
                        painter.fillRect(sx, sy, max(1, int(scale_x)), max(1, int(scale_y)), mask_color)

        # Draw sample points
        if self._sample_points:
            pen = QPen(QColor('#f39c12'))  # Orange
            pen.setWidth(2)
            painter.setPen(pen)
            brush = QBrush(QColor('#f39c12'))
            painter.setBrush(brush)
            for px, py in self._sample_points:
                sx, sy = int(px * scale_x), int(py * scale_y)
                painter.drawEllipse(sx - 4, sy - 4, 8, 8)

        # Draw crosshair
        if self._show_crosshair:
            pen = QPen(QColor(COLORS['accent_primary']))
            pen.setWidth(1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            cx, cy = scaled.width() // 2, scaled.height() // 2
            painter.drawLine(cx, 0, cx, scaled.height())
            painter.drawLine(0, cy, scaled.width(), cy)
            painter.drawEllipse(cx - 10, cy - 10, 20, 20)

        painter.end()
        self.setPixmap(scaled)

    def set_crosshair_visible(self, visible: bool):
        self._show_crosshair = visible
        if self._current_image is not None:
            self._redraw()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._current_image is not None:
            pixmap = self.pixmap()
            if pixmap:
                x_off = (self.width() - pixmap.width()) // 2
                y_off = (self.height() - pixmap.height()) // 2
                px = event.pos().x() - x_off
                py = event.pos().y() - y_off
                if 0 <= px < pixmap.width() and 0 <= py < pixmap.height():
                    img_h, img_w = self._current_image.shape[:2]
                    img_x = int(px * img_w / pixmap.width())
                    img_y = int(py * img_h / pixmap.height())
                    self.clicked.emit(img_x, img_y)
        super().mousePressEvent(event)


class LiveViewWidget(QWidget):
    """
    Live view with integrated segmentation and sample pattern generation.

    Signals:
        position_clicked: (camera_id, x, y)
        camera_switch_requested: (target_camera)
        positions_generated: list of (x, y) pixel coordinates
    """

    position_clicked = pyqtSignal(str, int, int)
    camera_switch_requested = pyqtSignal(str)
    positions_generated = pyqtSignal(list)

    def __init__(
        self,
        widefield_thread: Optional[CameraThread] = None,
        objective_thread: Optional[CameraThread] = None,
        parent=None
    ):
        super().__init__(parent)

        self._widefield_thread = widefield_thread
        self._objective_thread = objective_thread
        self._current_camera = 'wide'
        self._acquiring = False

        # Segmentation state
        self._captured_image: Optional[np.ndarray] = None
        self._current_mask: Optional[np.ndarray] = None
        self._sample_points: List[Tuple[float, float]] = []

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # === Left: Image Display ===
        left = QVBoxLayout()

        # Camera selector row
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Camera:"))
        self._camera_group = QButtonGroup(self)
        self._wide_radio = QRadioButton("Wide FOV")
        self._wide_radio.setChecked(True)
        self._wide_radio.toggled.connect(self._on_camera_changed)
        self._camera_group.addButton(self._wide_radio)
        cam_row.addWidget(self._wide_radio)
        self._obj_radio = QRadioButton("Objective")
        self._obj_radio.toggled.connect(self._on_camera_changed)
        self._camera_group.addButton(self._obj_radio)
        cam_row.addWidget(self._obj_radio)
        cam_row.addStretch()
        self._status_label = QLabel("Disconnected")
        self._status_label.setStyleSheet(f'color: {COLORS["text_secondary"]};')
        cam_row.addWidget(self._status_label)
        left.addLayout(cam_row)

        # Image display
        self._display = ImageDisplayWidget()
        self._display.clicked.connect(self._on_image_clicked)
        left.addWidget(self._display, 1)

        # Controls row
        ctrl_row = QHBoxLayout()
        self._acq_btn = QPushButton("▶ Start")
        self._acq_btn.setFixedWidth(70)
        self._acq_btn.setCheckable(True)
        self._acq_btn.clicked.connect(self._on_acquisition_toggled)
        ctrl_row.addWidget(self._acq_btn)

        self._fps_label = QLabel("— fps")
        self._fps_label.setFixedWidth(50)
        ctrl_row.addWidget(self._fps_label)

        ctrl_row.addWidget(QLabel("Exp:"))
        self._exposure_spin = QSpinBox()
        self._exposure_spin.setRange(1, 5000)
        self._exposure_spin.setValue(100)
        self._exposure_spin.setSuffix(" ms")
        self._exposure_spin.setFixedWidth(80)
        self._exposure_spin.valueChanged.connect(self._on_exposure_changed)
        ctrl_row.addWidget(self._exposure_spin)

        self._crosshair_btn = QPushButton("✛")
        self._crosshair_btn.setCheckable(True)
        self._crosshair_btn.setChecked(True)
        self._crosshair_btn.setFixedWidth(30)
        self._crosshair_btn.clicked.connect(lambda c: self._display.set_crosshair_visible(c))
        ctrl_row.addWidget(self._crosshair_btn)

        ctrl_row.addStretch()
        left.addLayout(ctrl_row)

        layout.addLayout(left, 3)

        # === Right: Segmentation & Pattern ===
        right = QVBoxLayout()
        right.setSpacing(8)

        # Capture section
        cap_group = QGroupBox("Capture")
        cap_lay = QVBoxLayout(cap_group)
        cap_lay.setContentsMargins(6, 6, 6, 6)
        self._capture_btn = QPushButton("📷 Capture for Segmentation")
        self._capture_btn.clicked.connect(self._on_capture)
        cap_lay.addWidget(self._capture_btn)
        right.addWidget(cap_group)

        # Segmentation section
        seg_group = QGroupBox("Segmentation")
        seg_lay = QVBoxLayout(seg_group)
        seg_lay.setContentsMargins(6, 6, 6, 6)
        self._run_seg_btn = QPushButton("▶ Run Segmentation")
        self._run_seg_btn.setEnabled(False)
        self._run_seg_btn.clicked.connect(self._on_run_segmentation)
        seg_lay.addWidget(self._run_seg_btn)
        self._seg_status = QLabel("No image captured")
        self._seg_status.setStyleSheet(f'color: {COLORS["text_secondary"]}; font-size: 11px;')
        seg_lay.addWidget(self._seg_status)
        right.addWidget(seg_group)

        # Pattern section
        pat_group = QGroupBox("Sample Pattern")
        pat_lay = QVBoxLayout(pat_group)
        pat_lay.setContentsMargins(6, 6, 6, 6)
        pat_lay.setSpacing(4)

        # Pattern type
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self._pattern_combo = QComboBox()
        self._pattern_combo.addItems(["Grid", "Ring", "Line", "Random"])
        self._pattern_combo.currentIndexChanged.connect(self._on_pattern_changed)
        type_row.addWidget(self._pattern_combo, 1)
        pat_lay.addLayout(type_row)

        # Stacked widget for pattern params
        self._pattern_stack = QStackedWidget()

        # Grid params
        grid_w = QWidget()
        grid_l = QHBoxLayout(grid_w)
        grid_l.setContentsMargins(0, 0, 0, 0)
        grid_l.addWidget(QLabel("Rows:"))
        self._grid_rows = QSpinBox()
        self._grid_rows.setRange(1, 20)
        self._grid_rows.setValue(3)
        grid_l.addWidget(self._grid_rows)
        grid_l.addWidget(QLabel("Cols:"))
        self._grid_cols = QSpinBox()
        self._grid_cols.setRange(1, 20)
        self._grid_cols.setValue(3)
        grid_l.addWidget(self._grid_cols)
        self._pattern_stack.addWidget(grid_w)

        # Ring params
        ring_w = QWidget()
        ring_l = QVBoxLayout(ring_w)
        ring_l.setContentsMargins(0, 0, 0, 0)
        ring_l.setSpacing(2)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Rays:"))
        self._ring_rays = QSpinBox()
        self._ring_rays.setRange(3, 36)
        self._ring_rays.setValue(8)
        r1.addWidget(self._ring_rays)
        r1.addWidget(QLabel("Rings:"))
        self._ring_rings = QSpinBox()
        self._ring_rings.setRange(1, 10)
        self._ring_rings.setValue(3)
        r1.addWidget(self._ring_rings)
        ring_l.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Interval:"))
        self._ring_interval = QSpinBox()
        self._ring_interval.setRange(5, 100)
        self._ring_interval.setValue(20)
        self._ring_interval.setSuffix(" px")
        r2.addWidget(self._ring_interval)
        r2.addWidget(QLabel("Offset:"))
        self._ring_offset = QSpinBox()
        self._ring_offset.setRange(0, 50)
        self._ring_offset.setValue(10)
        self._ring_offset.setSuffix(" px")
        r2.addWidget(self._ring_offset)
        ring_l.addLayout(r2)
        self._pattern_stack.addWidget(ring_w)

        # Line params
        line_w = QWidget()
        line_l = QHBoxLayout(line_w)
        line_l.setContentsMargins(0, 0, 0, 0)
        line_l.addWidget(QLabel("Points:"))
        self._line_points = QSpinBox()
        self._line_points.setRange(2, 50)
        self._line_points.setValue(5)
        line_l.addWidget(self._line_points)
        line_l.addStretch()
        self._pattern_stack.addWidget(line_w)

        # Random params
        rand_w = QWidget()
        rand_l = QHBoxLayout(rand_w)
        rand_l.setContentsMargins(0, 0, 0, 0)
        rand_l.addWidget(QLabel("Points:"))
        self._rand_points = QSpinBox()
        self._rand_points.setRange(1, 100)
        self._rand_points.setValue(10)
        rand_l.addWidget(self._rand_points)
        rand_l.addStretch()
        self._pattern_stack.addWidget(rand_w)

        pat_lay.addWidget(self._pattern_stack)

        self._gen_pattern_btn = QPushButton("Generate Pattern")
        self._gen_pattern_btn.setEnabled(False)
        self._gen_pattern_btn.clicked.connect(self._on_generate_pattern)
        pat_lay.addWidget(self._gen_pattern_btn)

        self._pattern_status = QLabel("Run segmentation first")
        self._pattern_status.setStyleSheet(f'color: {COLORS["text_secondary"]}; font-size: 11px;')
        pat_lay.addWidget(self._pattern_status)

        right.addWidget(pat_group)

        # Clear button
        self._clear_btn = QPushButton("Clear Overlays")
        self._clear_btn.clicked.connect(self._on_clear)
        right.addWidget(self._clear_btn)

        right.addStretch()
        layout.addLayout(right, 1)

    def _connect_signals(self):
        if self._widefield_thread:
            self._widefield_thread.frame_ready.connect(self._on_wide_frame)
            self._widefield_thread.fps_updated.connect(self._on_fps_updated)
            self._widefield_thread.connected.connect(self._on_wide_connected)

        if self._objective_thread:
            self._objective_thread.frame_ready.connect(self._on_obj_frame)
            self._objective_thread.fps_updated.connect(self._on_fps_updated)
            self._objective_thread.connected.connect(self._on_obj_connected)

    def _on_wide_frame(self, frame: np.ndarray, ts: float):
        if self._current_camera == 'wide':
            self._display.update_image(frame)

    def _on_obj_frame(self, frame: np.ndarray, ts: float):
        if self._current_camera == 'objective':
            self._display.update_image(frame)

    def _on_fps_updated(self, fps: float):
        self._fps_label.setText(f"{fps:.1f} fps")

    def _on_wide_connected(self):
        if self._current_camera == 'wide':
            self._status_label.setText("Wide FOV")
            self._status_label.setStyleSheet(f'color: {COLORS["success"]};')

    def _on_obj_connected(self):
        if self._current_camera == 'objective':
            self._status_label.setText("Objective")
            self._status_label.setStyleSheet(f'color: {COLORS["success"]};')

    def _on_camera_changed(self, checked: bool):
        if not checked:
            return
        new_cam = 'wide' if self._wide_radio.isChecked() else 'objective'
        if new_cam == self._current_camera:
            return

        if self._acquiring:
            self._stop_acquisition()

        if new_cam == 'objective':
            reply = QMessageBox.warning(
                self, "Camera Switch",
                "Switching to Objective requires stage movement.\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._wide_radio.setChecked(True)
                return

        self.camera_switch_requested.emit(new_cam)

    def switch_camera(self, camera: str):
        """Complete camera switch after stage movement."""
        self._current_camera = camera
        if camera == 'wide':
            self._wide_radio.blockSignals(True)
            self._wide_radio.setChecked(True)
            self._wide_radio.blockSignals(False)
        else:
            self._obj_radio.blockSignals(True)
            self._obj_radio.setChecked(True)
            self._obj_radio.blockSignals(False)

    def _on_acquisition_toggled(self, checked: bool):
        if checked:
            self._start_acquisition()
        else:
            self._stop_acquisition()

    def _start_acquisition(self):
        thread = self._widefield_thread if self._current_camera == 'wide' else self._objective_thread
        if thread and thread.is_connected:
            thread.start_acquisition()
            self._acq_btn.setText("⏹ Stop")
            self._acquiring = True

    def _stop_acquisition(self):
        thread = self._widefield_thread if self._current_camera == 'wide' else self._objective_thread
        if thread:
            thread.stop_acquisition()
        self._acq_btn.setText("▶ Start")
        self._acq_btn.setChecked(False)
        self._acquiring = False

    def _on_exposure_changed(self, val: int):
        thread = self._widefield_thread if self._current_camera == 'wide' else self._objective_thread
        if thread:
            thread.set_exposure(val / 1000.0)

    def _on_image_clicked(self, x: int, y: int):
        self.position_clicked.emit(self._current_camera, x, y)

    def _on_capture(self):
        """Capture current frame for segmentation."""
        thread = self._widefield_thread if self._current_camera == 'wide' else self._objective_thread
        if thread:
            frame = thread.get_latest_frame()
            if frame is not None:
                self._captured_image = frame.copy()
                self._display.update_image(frame)
                self._run_seg_btn.setEnabled(True)
                self._seg_status.setText(f"Captured: {frame.shape[1]}x{frame.shape[0]}")
                logger.info("Image captured for segmentation")
                return
        self._seg_status.setText("No camera image available")

    def _on_run_segmentation(self):
        """Run SAM segmentation on captured image."""
        if self._captured_image is None:
            return

        self._seg_status.setText("Running segmentation...")

        # Import here to avoid load time
        try:
            from raman_control.segmentation import SAMSegmenter
            segmenter = SAMSegmenter()
            if not segmenter.load_model():
                self._seg_status.setText("Failed to load SAM model")
                return

            # Use center point prompt
            h, w = self._captured_image.shape[:2]
            mask = segmenter.segment_point(self._captured_image, w // 2, h // 2)

            if mask is not None:
                self._current_mask = mask
                self._display.set_mask_overlay(mask)
                self._gen_pattern_btn.setEnabled(True)
                area = np.sum(mask)
                self._seg_status.setText(f"Detected area: {area} px²")
                self._pattern_status.setText("Ready to generate pattern")
            else:
                self._seg_status.setText("No object detected")
        except Exception as e:
            self._seg_status.setText(f"Error: {str(e)[:30]}")
            logger.error(f"Segmentation error: {e}")

    def _on_pattern_changed(self, idx: int):
        self._pattern_stack.setCurrentIndex(idx)

    def _on_generate_pattern(self):
        """Generate sampling points based on pattern and mask."""
        if self._current_mask is None:
            return

        pattern = self._pattern_combo.currentText().lower()
        mask = self._current_mask
        h, w = mask.shape

        # Get mask pixels
        ys, xs = np.where(mask == 1)
        if len(xs) == 0:
            self._pattern_status.setText("No mask area")
            return

        points = []

        if pattern == 'grid':
            rows, cols = self._grid_rows.value(), self._grid_cols.value()
            x_lin = np.linspace(xs.min(), xs.max(), cols)
            y_lin = np.linspace(ys.min(), ys.max(), rows)
            xx, yy = np.meshgrid(x_lin, y_lin)
            for px, py in zip(xx.flatten(), yy.flatten()):
                ix, iy = int(px), int(py)
                if 0 <= iy < h and 0 <= ix < w and mask[iy, ix]:
                    points.append((px, py))

        elif pattern == 'ring':
            num_rays = self._ring_rays.value()
            num_rings = self._ring_rings.value()
            interval = self._ring_interval.value()
            offset = self._ring_offset.value()

            cx, cy = xs.mean(), ys.mean()
            angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

            for angle in angles:
                dx, dy = np.cos(angle), np.sin(angle)
                # Find edge distance
                edge_dist = 0
                for d in range(max(w, h)):
                    x, y = int(cx + d * dx), int(cy + d * dy)
                    if x < 0 or x >= w or y < 0 or y >= h or not mask[y, x]:
                        edge_dist = d - 1
                        break
                # Place points from edge inward
                for r in range(num_rings):
                    d = edge_dist - offset - r * interval
                    if d > 10:
                        px, py = cx + d * dx, cy + d * dy
                        if 0 <= int(py) < h and 0 <= int(px) < w and mask[int(py), int(px)]:
                            points.append((px, py))

        elif pattern == 'line':
            num = self._line_points.value()
            cx, cy = int(xs.mean()), int(ys.mean())
            # Find row with mask pixels through center
            row_xs = np.where(mask[cy] == 1)[0]
            if len(row_xs) > 0:
                x_pts = np.linspace(row_xs.min(), row_xs.max(), num)
                points = [(x, cy) for x in x_pts]

        elif pattern == 'random':
            num = self._rand_points.value()
            indices = np.random.choice(len(xs), min(num, len(xs)), replace=False)
            points = [(xs[i], ys[i]) for i in indices]

        self._sample_points = points
        self._display.set_sample_points(points)
        self._pattern_status.setText(f"Generated {len(points)} points")
        self.positions_generated.emit(points)
        logger.info(f"Generated {len(points)} sample points ({pattern})")

    def _on_clear(self):
        """Clear all overlays."""
        self._current_mask = None
        self._sample_points = []
        self._display.clear_overlays()
        self._gen_pattern_btn.setEnabled(False)
        self._pattern_status.setText("Cleared")

    @property
    def current_camera(self) -> str:
        return self._current_camera

    def get_sample_points(self) -> List[Tuple[float, float]]:
        """Get generated sample points."""
        return self._sample_points.copy()
