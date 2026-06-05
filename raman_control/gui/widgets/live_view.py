"""
Live camera view widget with integrated segmentation and sample pattern generation.

Features:
- Single camera display (wide FOV or objective)
- Safe camera switching with stage movement
- Image segmentation (SAM point prompt)
- Sample pattern generation (grid, ring, line, random)
- Overlay of detected area and sample points
- Display freeze on capture with coordinate conversion
"""

from typing import Optional, List, Tuple, Callable
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QSlider, QLineEdit, QCheckBox,
    QComboBox, QSizePolicy, QButtonGroup, QRadioButton, QStackedWidget,
    QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QPointF, QRectF, QSize, QThread, QTimer, QEvent
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QIcon, QPolygon, QFont

from raman_control.gui.threads.camera_thread import CameraThread
from raman_control.gui.styles import COLORS
from raman_control.config.logging_config import get_logger
from raman_control.config.hardware_config import CameraConfig
from raman_control.segmentation.pattern_generator import (
    PatternGenerator, CalibrationParams, SamplePosition,
)

logger = get_logger('live_view')

# Fixed marker diameter (um) = the Prusa XY step limit. Two markers overlap
# exactly when their points are closer than this, flagging spacings the stage
# can't resolve. The on-screen pixel size is derived from the widefield
# pixel-size calibration, so recalibrating updates the marker automatically.
MARKER_DIAMETER_UM = 60.0

# Filled-cyan button style (with grey-outline disabled state), matching the
# "Go to Target" button. Used for primary actions: Capture, Segment, Generate.
CYAN_BTN_STYLE = '''
    QPushButton {
        background-color: #00b4b4;
        border: 1px solid #00d4d4;
        border-radius: 5px;
        color: #1a1a1a;
        font-weight: 600;
        padding: 5px 12px;
        min-height: 18px;
    }
    QPushButton:hover { background-color: #00d4d4; }
    QPushButton:pressed { background-color: #008080; }
    QPushButton:disabled {
        background-color: transparent;
        border: 1px solid #4d4d4d;
        color: #505050;
    }
'''

# Cyan outline (transparent fill) — secondary actions. Greys out when disabled.
CYAN_OUTLINE_BTN_STYLE = '''
    QPushButton {
        background-color: transparent;
        border: 1px solid #00b4b4;
        border-radius: 5px;
        color: #00b4b4;
        font-weight: 600;
        padding: 5px 12px;
        min-height: 18px;
    }
    QPushButton:hover {
        background-color: rgba(0, 180, 180, 30);
        border-color: #00d4d4;
        color: #00d4d4;
    }
    QPushButton:pressed { background-color: rgba(0, 180, 180, 60); }
    QPushButton:disabled {
        background-color: transparent;
        border: 1px solid #4d4d4d;
        color: #505050;
    }
'''


class ImageDisplayWidget(QLabel):
    """Widget for displaying camera images with overlays (mask + sample points)."""

    clicked = pyqtSignal(int, int)  # x, y in image coordinates

    def __init__(self, parent=None):
        super().__init__(parent)
        self._show_crosshair = True
        self._current_image: Optional[np.ndarray] = None
        self._mask_overlay: Optional[np.ndarray] = None
        self._overlay_rgba: Optional[np.ndarray] = None  # kept alive for QImage
        self._sample_points: List[Tuple[float, float]] = []
        # Scale bar (bottom-right): physical length in um and image um/pixel.
        self._scalebar_um: float = 0.0
        self._scalebar_um_per_px: float = 0.0

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f'''
            background-color: {COLORS['bg_light']};
            border: 1px solid {COLORS['border_dark']};
            border-radius: 4px;
        ''')
        self.setText("No image")

        # Zoom / pan state. zoom == 1 fits the frame to the widget; the wheel
        # zooms in toward the cursor and left-drag pans once zoomed in. A
        # minimap in the bottom-left shows the visible region while zoomed.
        self._zoom = 1.0
        self._min_zoom = 1.0
        self._max_zoom = 20.0
        self._view_center: Optional[Tuple[float, float]] = None  # image px at widget center
        self._image_size = (0, 0)          # (w, h) of the current frame
        self._image_pixmap: Optional[QPixmap] = None
        self._overlay_pixmap: Optional[QPixmap] = None
        # Left-drag panning vs. click discrimination.
        self._press_pos: Optional[QPoint] = None
        self._pan_last: Optional[QPoint] = None
        self._dragged = False

    def update_image(self, image: np.ndarray):
        """Update the displayed image."""
        self._current_image = image

        if image.ndim == 2:
            h, w = image.shape
            qimage = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, c = image.shape
            qimage = QImage(image.data, w, h, w * c, QImage.Format.Format_RGB888)
        # fromImage deep-copies, so the numpy buffer need not be kept alive.
        self._image_pixmap = QPixmap.fromImage(qimage)

        # Reset zoom/pan if the frame dimensions change (e.g. camera switch),
        # otherwise the cached view center would refer to the wrong frame.
        if (w, h) != self._image_size:
            self._image_size = (w, h)
            self._reset_view()

        if self.text():
            self.setText("")  # hide the "No image" placeholder
        self.update()

    def set_mask_overlay(self, mask: Optional[np.ndarray]):
        """Set mask overlay (binary mask). Uses numpy RGBA overlay for speed."""
        self._mask_overlay = mask
        # Pre-compute the overlay as a native-resolution pixmap so paintEvent
        # can draw it under the same zoom/pan transform as the image.
        if mask is not None:
            h, w = mask.shape[:2]
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[mask > 0] = [30, 144, 255, 100]  # Dodger blue with alpha
            rgba = np.ascontiguousarray(rgba)
            self._overlay_rgba = rgba  # keep alive until fromImage copies it
            qimage = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
            self._overlay_pixmap = QPixmap.fromImage(qimage)
        else:
            self._overlay_rgba = None
            self._overlay_pixmap = None
        self.update()

    def set_sample_points(self, points: List[Tuple[float, float]]):
        """Set sample points to display."""
        self._sample_points = points
        self.update()

    def clear_overlays(self):
        """Clear mask and sample points."""
        self._mask_overlay = None
        self._overlay_rgba = None
        self._sample_points = []
        self.update()

    # --- Zoom / pan geometry -------------------------------------------------

    def _reset_view(self):
        """Return to fit-to-widget (zoom 1, centered)."""
        self._zoom = 1.0
        self._view_center = None

    def _fit_scale(self) -> float:
        """Display pixels per image pixel at zoom == 1 (whole frame fits)."""
        w, h = self._image_size
        if w == 0 or h == 0:
            return 1.0
        return min(self.width() / w, self.height() / h)

    def _scale(self) -> float:
        """Effective display pixels per image pixel at the current zoom."""
        return self._fit_scale() * self._zoom

    def _center(self) -> Tuple[float, float]:
        """Image coordinate currently shown at the widget center."""
        if self._view_center is None:
            w, h = self._image_size
            return (w / 2.0, h / 2.0)
        return self._view_center

    def _clamp_view(self):
        """Keep the visible region within the image (no empty margins)."""
        w, h = self._image_size
        if w == 0:
            return
        scale = self._scale()
        half_w = (self.width() / 2.0) / scale
        half_h = (self.height() / 2.0) / scale
        cx, cy = self._center()
        cx = w / 2.0 if 2 * half_w >= w else max(half_w, min(w - half_w, cx))
        cy = h / 2.0 if 2 * half_h >= h else max(half_h, min(h - half_h, cy))
        self._view_center = (cx, cy)

    def _widget_to_image(self, wx: float, wy: float) -> Tuple[float, float]:
        scale = self._scale()
        cx, cy = self._center()
        return (cx + (wx - self.width() / 2.0) / scale,
                cy + (wy - self.height() / 2.0) / scale)

    def _image_to_widget(self, ix: float, iy: float) -> Tuple[float, float]:
        scale = self._scale()
        cx, cy = self._center()
        return (self.width() / 2.0 + (ix - cx) * scale,
                self.height() / 2.0 + (iy - cy) * scale)

    # --- Rendering -----------------------------------------------------------

    def paintEvent(self, event):
        # QLabel paints the styled background/border and the "No image" text.
        super().paintEvent(event)
        if self._current_image is None or self._image_pixmap is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        W, H = self.width(), self.height()
        scale = self._scale()
        cx, cy = self._center()

        # Image (+ mask overlay) under the zoom/pan transform.
        painter.save()
        painter.translate(W / 2.0, H / 2.0)
        painter.scale(scale, scale)
        painter.translate(-cx, -cy)
        painter.drawPixmap(0, 0, self._image_pixmap)
        if self._overlay_pixmap is not None:
            painter.drawPixmap(0, 0, self._overlay_pixmap)
        painter.restore()

        # Sample points — drawn in widget space so markers stay a constant
        # on-screen size regardless of zoom.
        if self._sample_points:
            pen = QPen(QColor('#f39c12'))  # Orange
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor('#f39c12')))
            for px, py in self._sample_points:
                wx, wy = self._image_to_widget(px, py)
                painter.drawEllipse(QPointF(wx, wy), 4, 4)

        # Crosshair at the widget center.
        if self._show_crosshair:
            pen = QPen(QColor(COLORS['accent_primary']))
            pen.setWidth(1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            mx, my = W // 2, H // 2
            painter.drawLine(mx, 0, mx, H)
            painter.drawLine(0, my, W, my)
            painter.drawEllipse(mx - 10, my - 10, 20, 20)

        self._draw_scale_bar(painter, scale)

        # Minimap only matters when zoomed in (otherwise the whole frame shows).
        if self._zoom > 1.0:
            self._draw_minimap(painter)

        painter.end()

    def _draw_scale_bar(self, painter: QPainter, scale: float):
        """Bottom-right scale bar, sized to the current (zoomed) scale."""
        if not (self._scalebar_um > 0 and self._scalebar_um_per_px > 0):
            return
        bar_disp = (self._scalebar_um / self._scalebar_um_per_px) * scale
        sw, sh = self.width(), self.height()
        if not (4 < bar_disp < sw):
            return
        margin = 12
        thickness = 4
        pad = 4
        label = f"{int(self._scalebar_um)} um"
        font = QFont("Segoe UI", 9)
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()
        text_h = fm.height()
        text_w = fm.horizontalAdvance(label)

        bar_x2 = sw - margin
        bar_x1 = bar_x2 - bar_disp
        bar_y = sh - margin - thickness  # top of the bar
        block_w = max(bar_disp, text_w)

        # Translucent dark backing so the bar shows on any image.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 150)))
        painter.drawRoundedRect(
            int(bar_x2 - block_w - pad), int(bar_y - text_h - pad),
            int(block_w + 2 * pad), int(text_h + thickness + 2 * pad),
            4, 4,
        )

        # White bar
        painter.setBrush(QBrush(QColor('white')))
        painter.drawRect(int(bar_x1), int(bar_y), int(round(bar_disp)), thickness)

        # White label centered above the bar
        painter.setPen(QPen(QColor('white')))
        tx = int(bar_x2 - bar_disp / 2 - text_w / 2)
        ty = int(bar_y - pad)  # baseline just above the bar
        painter.drawText(tx, ty, label)

    def _draw_minimap(self, painter: QPainter):
        """Bottom-left overview thumbnail with a box marking the visible region."""
        w, h = self._image_size
        if w == 0 or self._image_pixmap is None:
            return

        MM_MAX = 150.0  # max thumbnail extent in px
        mm_scale = min(MM_MAX / w, MM_MAX / h)
        mw, mh = w * mm_scale, h * mm_scale
        margin = 12
        x0 = margin
        y0 = self.height() - margin - mh

        # Dark backing
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 160)))
        painter.drawRoundedRect(QRectF(x0 - 4, y0 - 4, mw + 8, mh + 8), 4, 4)

        # Thumbnail of the whole frame
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(QRectF(x0, y0, mw, mh), self._image_pixmap,
                           QRectF(0, 0, w, h))

        # Box = currently visible image region (clamped to the frame).
        ix0, iy0 = self._widget_to_image(0, 0)
        ix1, iy1 = self._widget_to_image(self.width(), self.height())
        ix0 = max(0.0, min(w, ix0)); ix1 = max(0.0, min(w, ix1))
        iy0 = max(0.0, min(h, iy0)); iy1 = max(0.0, min(h, iy1))
        rect = QRectF(x0 + ix0 * mm_scale, y0 + iy0 * mm_scale,
                      (ix1 - ix0) * mm_scale, (iy1 - iy0) * mm_scale)
        pen = QPen(QColor(COLORS['accent_primary']))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        # Thin outline around the whole minimap
        pen = QPen(QColor(255, 255, 255, 120))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRect(QRectF(x0, y0, mw, mh))

    def set_crosshair_visible(self, visible: bool):
        self._show_crosshair = visible
        self.update()

    def set_scale_bar(self, length_um: float, um_per_pixel: float):
        """Configure the bottom-right scale bar.

        length_um: physical length the bar represents (e.g. 500).
        um_per_pixel: object-plane micrometers per image pixel for the
        currently displayed camera. Pass 0 to hide the bar.
        """
        self._scalebar_um = float(length_um)
        self._scalebar_um_per_px = float(um_per_pixel)
        self.update()

    # --- Interaction ---------------------------------------------------------

    def wheelEvent(self, event):
        """Scroll to zoom in/out, keeping the point under the cursor fixed."""
        if self._current_image is None or self._image_size[0] == 0:
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        factor = 1.25 if delta > 0 else 1 / 1.25
        old_zoom = self._zoom
        new_zoom = max(self._min_zoom, min(self._max_zoom, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            event.accept()
            return

        pos = event.position()  # widget coords (QPointF)
        ix, iy = self._widget_to_image(pos.x(), pos.y())
        self._zoom = new_zoom
        # Solve for the view center that keeps (ix, iy) under the cursor.
        scale = self._scale()
        self._view_center = (ix - (pos.x() - self.width() / 2.0) / scale,
                             iy - (pos.y() - self.height() / 2.0) / scale)
        self._clamp_view()
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._current_image is not None:
            self._press_pos = event.pos()
            self._pan_last = event.pos()
            self._dragged = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press_pos is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            # Pan only when zoomed in (there's nothing off-screen otherwise).
            if self._zoom > 1.0 and self._pan_last is not None:
                dx = event.pos().x() - self._pan_last.x()
                dy = event.pos().y() - self._pan_last.y()
                if dx or dy:
                    scale = self._scale()
                    cx, cy = self._center()
                    self._view_center = (cx - dx / scale, cy - dy / scale)
                    self._clamp_view()
                    self.update()
                self._pan_last = event.pos()
            if (event.pos() - self._press_pos).manhattanLength() > 4:
                self._dragged = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._press_pos is not None:
            was_drag = self._dragged
            press = self._press_pos
            self._press_pos = None
            self._pan_last = None
            self._dragged = False
            # A press-release without panning is a click → emit image coords.
            if not was_drag and self._current_image is not None:
                ix, iy = self._widget_to_image(press.x(), press.y())
                iw, ih = self._image_size
                if 0 <= ix < iw and 0 <= iy < ih:
                    self.clicked.emit(int(ix), int(iy))
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Double-click resets the view to fit-to-widget."""
        if event.button() == Qt.MouseButton.LeftButton and self._current_image is not None:
            self._reset_view()
            self.update()
        super().mouseDoubleClickEvent(event)


class ClickableLabel(QLabel):
    """Captured-image preview with scroll-wheel zoom and drag-pan.

    Displays a composited "base pixmap" (image + overlays). The wheel zooms
    toward the cursor and left-drag pans when zoomed — the same interaction as
    the live camera view — with a minimap while zoomed. ``clicked`` is emitted
    in BASE-PIXMAP pixel coordinates (a press-release without panning), so the
    host can map it to image coordinates for point editing / stage moves.
    """

    clicked = pyqtSignal(int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_pixmap: Optional[QPixmap] = None
        self._zoom = 1.0
        self._min_zoom = 1.0
        self._max_zoom = 10.0
        self._view_center: Optional[Tuple[float, float]] = None  # base-pixmap px
        self._press_pos: Optional[QPoint] = None
        self._pan_last: Optional[QPoint] = None
        self._dragged = False
        # Live "you are here" marker: (base_x, base_y, radius_base_px) or None.
        self._probe_marker: Optional[Tuple[float, float, float]] = None

    def set_probe_marker(self, marker: Optional[Tuple[float, float, float]]):
        """Set the green probe marker as (base_x, base_y, radius_base_px), or
        None to hide. The radius is in base-pixmap pixels (a physical size)."""
        self._probe_marker = marker
        self.update()

    # --- pixmap / view state ---

    def base_pixmap(self) -> Optional[QPixmap]:
        return self._base_pixmap

    def set_base_pixmap(self, pixmap: QPixmap):
        old = self._base_pixmap
        self._base_pixmap = pixmap
        # Only reset the view when the image size actually changes, so routine
        # redraws (marker highlight, label toggle) keep the current zoom.
        if old is None or pixmap is None or old.size() != pixmap.size():
            self._reset_view()
        self.update()

    def reset_view(self):
        self._reset_view()
        self.update()

    def _reset_view(self):
        self._zoom = 1.0
        self._view_center = None

    def _bw_bh(self) -> Tuple[int, int]:
        if self._base_pixmap is None:
            return (0, 0)
        return (self._base_pixmap.width(), self._base_pixmap.height())

    def _fit_scale(self) -> float:
        bw, bh = self._bw_bh()
        if bw == 0 or bh == 0:
            return 1.0
        return min(self.width() / bw, self.height() / bh)

    def _scale(self) -> float:
        return self._fit_scale() * self._zoom

    def _center(self) -> Tuple[float, float]:
        if self._view_center is None:
            bw, bh = self._bw_bh()
            return (bw / 2.0, bh / 2.0)
        return self._view_center

    def _clamp(self):
        bw, bh = self._bw_bh()
        if bw == 0:
            return
        scale = self._scale()
        half_w = (self.width() / 2.0) / scale
        half_h = (self.height() / 2.0) / scale
        cx, cy = self._center()
        cx = bw / 2.0 if 2 * half_w >= bw else max(half_w, min(bw - half_w, cx))
        cy = bh / 2.0 if 2 * half_h >= bh else max(half_h, min(bh - half_h, cy))
        self._view_center = (cx, cy)

    def _widget_to_base(self, wx: float, wy: float) -> Tuple[float, float]:
        scale = self._scale()
        cx, cy = self._center()
        return (cx + (wx - self.width() / 2.0) / scale,
                cy + (wy - self.height() / 2.0) / scale)

    def _base_to_widget(self, bx: float, by: float) -> Tuple[float, float]:
        scale = self._scale()
        cx, cy = self._center()
        return (self.width() / 2.0 + (bx - cx) * scale,
                self.height() / 2.0 + (by - cy) * scale)

    # --- rendering ---

    def paintEvent(self, event):
        if self._base_pixmap is None:
            super().paintEvent(event)  # placeholder text + styled background
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        W, H = self.width(), self.height()
        scale = self._scale()
        cx, cy = self._center()
        painter.save()
        painter.translate(W / 2.0, H / 2.0)
        painter.scale(scale, scale)
        painter.translate(-cx, -cy)
        painter.drawPixmap(0, 0, self._base_pixmap)
        painter.restore()

        # Live probe marker — green dot at the current "looking at" spot. Its
        # radius is a physical size (base-pixmap px), so it scales with zoom
        # like the sample-point markers.
        if self._probe_marker is not None:
            bx, by, r_base = self._probe_marker
            wx, wy = self._base_to_widget(bx, by)
            wr = max(3.0, r_base * self._scale())
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            pen = QPen(QColor('#2ecc71'))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(46, 204, 113, 110)))
            painter.drawEllipse(QPointF(wx, wy), wr, wr)

        if self._zoom > 1.0:
            self._draw_minimap(painter)
        painter.end()

    def _draw_minimap(self, painter: QPainter):
        bw, bh = self._bw_bh()
        if bw == 0:
            return
        MM_MAX = 90.0
        mm_scale = min(MM_MAX / bw, MM_MAX / bh)
        mw, mh = bw * mm_scale, bh * mm_scale
        margin = 6
        x0, y0 = margin, self.height() - margin - mh
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 160)))
        painter.drawRoundedRect(QRectF(x0 - 3, y0 - 3, mw + 6, mh + 6), 3, 3)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(QRectF(x0, y0, mw, mh), self._base_pixmap,
                           QRectF(0, 0, bw, bh))
        bx0, by0 = self._widget_to_base(0, 0)
        bx1, by1 = self._widget_to_base(self.width(), self.height())
        bx0 = max(0.0, min(bw, bx0)); bx1 = max(0.0, min(bw, bx1))
        by0 = max(0.0, min(bh, by0)); by1 = max(0.0, min(bh, by1))
        pen = QPen(QColor(COLORS['accent_primary']))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(x0 + bx0 * mm_scale, y0 + by0 * mm_scale,
                                (bx1 - bx0) * mm_scale, (by1 - by0) * mm_scale))

    # --- interaction ---

    def wheelEvent(self, event):
        if self._base_pixmap is None:
            event.ignore()
            return
        d = event.angleDelta().y()
        if d == 0:
            event.ignore()
            return
        factor = 1.25 if d > 0 else 1 / 1.25
        old = self._zoom
        new = max(self._min_zoom, min(self._max_zoom, old * factor))
        if abs(new - old) < 1e-6:
            event.accept()
            return
        pos = event.position()
        bx, by = self._widget_to_base(pos.x(), pos.y())
        self._zoom = new
        scale = self._scale()
        self._view_center = (bx - (pos.x() - self.width() / 2.0) / scale,
                             by - (pos.y() - self.height() / 2.0) / scale)
        self._clamp()
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._base_pixmap is not None:
            self._press_pos = event.pos()
            self._pan_last = event.pos()
            self._dragged = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press_pos is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            if self._zoom > 1.0 and self._pan_last is not None:
                dx = event.pos().x() - self._pan_last.x()
                dy = event.pos().y() - self._pan_last.y()
                if dx or dy:
                    scale = self._scale()
                    cx, cy = self._center()
                    self._view_center = (cx - dx / scale, cy - dy / scale)
                    self._clamp()
                    self.update()
                self._pan_last = event.pos()
            if (event.pos() - self._press_pos).manhattanLength() > 4:
                self._dragged = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._press_pos is not None:
            was_drag = self._dragged
            press = self._press_pos
            self._press_pos = None
            self._pan_last = None
            self._dragged = False
            if not was_drag and self._base_pixmap is not None:
                bx, by = self._widget_to_base(press.x(), press.y())
                bw, bh = self._bw_bh()
                if 0 <= bx < bw and 0 <= by < bh:
                    self.clicked.emit(int(bx), int(by))
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._base_pixmap is not None:
            self._reset_view()
            self.update()
        super().mouseDoubleClickEvent(event)


class _AutoSizeStack(QStackedWidget):
    """QStackedWidget that takes the height of the CURRENT page only.

    The default sizes to the tallest page, leaving a blank gap under
    shorter pages. Reporting the current page's size hint instead lets
    the panel grow/shrink as the pattern type changes — no dead space.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.currentChanged.connect(self.updateGeometry)

    def sizeHint(self):
        w = self.currentWidget()
        return w.sizeHint() if w is not None else super().sizeHint()

    def minimumSizeHint(self):
        w = self.currentWidget()
        return w.minimumSizeHint() if w is not None else super().minimumSizeHint()


class SegmentationWorker(QThread):
    """Background thread: load the U-Net (if needed) and segment an image.

    Loading torch + the checkpoint takes several seconds and would freeze
    the GUI if run on the main thread. This runs it off-thread and emits
    progress so the status bar can show the app is alive.
    """

    progress = pyqtSignal(int, str)       # percent, message
    finished_mask = pyqtSignal(object)    # np.ndarray mask, or None
    failed = pyqtSignal(str)              # error message

    def __init__(self, segmenter, image, parent=None):
        super().__init__(parent)
        self._segmenter = segmenter
        self._image = image

    def run(self):
        try:
            self.progress.emit(5, "Starting segmentation…")
            if not self._segmenter.is_loaded:
                ok = self._segmenter.load_model(
                    progress_cb=lambda p, m: self.progress.emit(p, m)
                )
                if not ok:
                    self.failed.emit("Failed to load U-Net model")
                    return

            self.progress.emit(80, "Segmenting…")
            mask = self._segmenter.segment(self._image)
            self.progress.emit(100, "Segmentation done")
            self.finished_mask.emit(mask)
        except Exception as e:  # noqa: BLE001 - report any failure to the GUI
            self.failed.emit(str(e))


class LiveViewWidget(QWidget):
    """
    Live view with integrated segmentation and sample pattern generation.

    Signals:
        position_clicked: (camera_id, x, y)
        camera_switch_requested: (target_camera)
        positions_generated: list of SamplePosition objects with stage coordinates
    """

    position_clicked = pyqtSignal(str, int, int)
    camera_switch_requested = pyqtSignal(str)
    positions_generated = pyqtSignal(list)
    # Widefield-frame (logical) stage coordinate of a clicked pattern point.
    # The main window converts it for the current view and moves the stage.
    move_to_point_requested = pyqtSignal(float, float, float)
    # Segmentation progress (percent, message) for the status-bar progress bar.
    segmentation_progress = pyqtSignal(int, str)
    # Camera autofocus requested with the number of Prusa Z steps to scan.
    camera_autofocus_requested = pyqtSignal(int)

    def __init__(
        self,
        widefield_thread: Optional[CameraThread] = None,
        objective_thread: Optional[CameraThread] = None,
        position_provider: Optional[Callable] = None,
        looking_at_provider: Optional[Callable] = None,
        widefield_config: Optional[CameraConfig] = None,
        objective_config: Optional[CameraConfig] = None,
        parent=None
    ):
        super().__init__(parent)

        self._widefield_thread = widefield_thread
        self._objective_thread = objective_thread
        self._position_provider = position_provider or (lambda: (0.0, 0.0, 0.0))
        # Widefield-frame position of where the active camera is looking now
        # (the green "you are here" marker). Defaults to the raw position.
        self._looking_at_provider = looking_at_provider or self._position_provider
        self._widefield_config = widefield_config or CameraConfig()
        self._objective_config = objective_config or CameraConfig()
        self._current_camera = 'wide'
        self._acquiring = False

        # Segmentation state
        self._captured_image: Optional[np.ndarray] = None
        self._current_mask: Optional[np.ndarray] = None
        self._sample_points: List[Tuple[float, float]] = []
        # True only while the probe (green tracker) is inside the captured
        # area — clicking points to move is blocked otherwise (safety: the
        # probe may be far away / at an unsafe Z).
        self._probe_in_area = False
        # Full generated positions (with stage coords), parallel to
        # _sample_points, kept so a clicked point can drive the stage.
        self._generated_positions: List[SamplePosition] = []
        self._selected_point_id: Optional[int] = None

        # Point-editing mode ('add' / 'delete' / None) and the snapshot used
        # to revert on Cancel.
        self._edit_mode: Optional[str] = None
        self._edit_snapshot = None

        # Segmentation: cached U-Net segmenter (loaded once) + worker thread.
        self._segmenter = None
        self._seg_worker: Optional[SegmentationWorker] = None

        # Auto-center loop state (segment -> move -> re-segment -> check edge
        # -> repeat). _center_phase is 'initial_segment' (the first segmentation
        # done on click) or 'loop' (segmentations following a centering move).
        self._centering_active = False
        self._center_phase = None
        self._center_repeats = 0
        self._center_max_repeats = 3
        self._center_awaiting_move = False
        # Center needs a referenced stage (it moves the stage).
        self._stage_referenced = False
        # Switching to the objective is only allowed once an autofocus Z range
        # is registered (objective on the sample without a known focus range is
        # unsafe).
        self._af_registered = False
        self._center_watchdog = QTimer(self)
        self._center_watchdog.setSingleShot(True)
        self._center_watchdog.timeout.connect(
            lambda: self._finish_centering("move timed out")
        )

        self._captured_camera: str = 'wide'
        self._capture_stage_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        self._setup_ui()
        self._connect_signals()
        self._update_scale_bar()  # initial (wide FOV: 500 um)

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        # Extra bottom margin leaves a gap between this block and the
        # splitter divider below it.
        layout.setContentsMargins(4, 4, 4, 12)
        layout.setSpacing(8)

        # === Left: Image Display (stretches) ===
        left = QVBoxLayout()

        # Camera selector row — prominent toggle buttons because switching
        # triggers stage movement that could collide with the sample.
        cam_row = QHBoxLayout()
        cam_row.setSpacing(4)

        self._switch_cam_btn = QPushButton("Switch to Objective")
        self._switch_cam_btn.setMinimumHeight(34)
        self._switch_cam_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # Disabled until the stage is referenced AND an autofocus range is
        # registered. When enabled it's a gold (warning-color) outline button,
        # matching the gold "STAGE MOVES ON SWITCH" label below it.
        self._switch_cam_btn.setEnabled(False)
        self._switch_cam_btn.setStyleSheet(f'''
            QPushButton {{
                background-color: transparent;
                border: 2px solid {COLORS['warning']};
                border-radius: 6px;
                color: {COLORS['warning']};
                font-size: 13px;
                padding: 6px 20px;
            }}
            QPushButton:hover {{
                background-color: rgba(255, 152, 0, 30);
                border-color: {COLORS['warning']};
                color: {COLORS['warning']};
            }}
            QPushButton:disabled {{
                background-color: transparent;
                border: 2px solid #4d4d4d;
                color: #505050;
            }}
        ''')
        self._switch_cam_btn.clicked.connect(self._on_switch_cam_clicked)
        cam_row.addWidget(self._switch_cam_btn)

        # Stage-moves warning label
        self._cam_warning = QLabel("REFERENCE THE STAGE TO SWITCH")
        self._cam_warning.setStyleSheet('''
            color: #505050;
            font-size: 10px;
            font-weight: bold;
            padding: 0 8px;
        ''')
        cam_row.addWidget(self._cam_warning)

        cam_row.addStretch()
        self._status_label = QLabel("Disconnected")
        self._status_label.setStyleSheet(f'color: {COLORS["text_secondary"]};')
        cam_row.addWidget(self._status_label)
        left.addLayout(cam_row)

        # Image display
        self._display = ImageDisplayWidget()
        self._display.clicked.connect(self._on_image_clicked)
        left.addWidget(self._display, 1)

        # Contrast / Smooth / Crosshair float in the image's top-right corner.
        self._create_image_overlay()

        # Controls row: play button | exposure | capture
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        _ACQ_BTN_DISABLED = '''
            QPushButton:disabled {
                background-color: transparent;
                border: 2px solid #4d4d4d;
                color: #505050;
            }
        '''
        _ACQ_BTN_OUTLINE = f'''
            QPushButton {{
                background-color: transparent;
                border: 2px solid {COLORS['accent_primary']};
                border-radius: 6px;
                color: {COLORS['accent_primary']};
            }}
            QPushButton:hover {{
                background-color: rgba(0, 180, 180, 30);
            }}
            {_ACQ_BTN_DISABLED}
        '''
        _ACQ_BTN_FILLED = f'''
            QPushButton {{
                background-color: {COLORS['accent_primary']};
                border: 2px solid {COLORS['accent_primary']};
                border-radius: 6px;
                color: {COLORS['bg_dark']};
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
                border-color: {COLORS['accent_hover']};
            }}
            {_ACQ_BTN_DISABLED}
        '''
        self._acq_btn_outline = _ACQ_BTN_OUTLINE
        self._acq_btn_filled = _ACQ_BTN_FILLED

        self._acq_btn = QPushButton()
        self._acq_btn.setFixedSize(48, 36)
        self._acq_btn.setIconSize(QSize(24, 24))
        self._acq_btn.setCheckable(True)
        self._acq_btn.setEnabled(False)  # Disabled until a camera connects
        self._acq_btn.clicked.connect(self._on_acquisition_toggled)
        self._update_acq_btn(False)
        ctrl_row.addWidget(self._acq_btn)

        ctrl_row.addWidget(QLabel("Exp:"))
        self._exposure_entry = QLineEdit("3")
        self._exposure_entry.setFixedWidth(40)
        self._exposure_entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._exposure_entry.returnPressed.connect(self._on_exposure_changed)
        ctrl_row.addWidget(self._exposure_entry)
        ctrl_row.addWidget(QLabel("ms"))

        # Autofocus: scan the Prusa Z range (set in the left Autofocus panel)
        # in this many steps.
        ctrl_row.addSpacing(12)
        ctrl_row.addWidget(QLabel("Steps:"))
        self._af_steps_spin = QSpinBox()
        self._af_steps_spin.setRange(2, 200)
        self._af_steps_spin.setValue(5)
        self._af_steps_spin.setFixedWidth(56)
        self._af_steps_spin.setToolTip("Number of Prusa Z steps across the autofocus range")
        ctrl_row.addWidget(self._af_steps_spin)
        self._autofocus_btn = QPushButton("Autofocus Camera with Prusa")
        self._autofocus_btn.setStyleSheet(CYAN_BTN_STYLE)
        self._autofocus_btn.setEnabled(False)  # until an autofocus range is registered
        self._autofocus_btn.setToolTip(
            "Camera autofocus over the Prusa Z range for the current camera view")
        self._autofocus_btn.clicked.connect(self._on_autofocus_clicked)
        ctrl_row.addWidget(self._autofocus_btn)

        ctrl_row.addStretch()

        self._capture_btn = QPushButton("Capture")
        self._capture_btn.setStyleSheet(CYAN_BTN_STYLE)
        self._capture_btn.clicked.connect(self._on_capture)
        ctrl_row.addWidget(self._capture_btn)

        left.addLayout(ctrl_row)

        layout.addLayout(left, 1)

        # === Right: Segmentation & Pattern (fixed width) ===
        right_container = QWidget()
        right_container.setFixedWidth(450)
        right = QVBoxLayout(right_container)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(8)

        # Captured image preview
        cap_group = QGroupBox("Captured Image")
        cap_lay = QVBoxLayout(cap_group)
        cap_lay.setContentsMargins(6, 6, 6, 6)

        img_w = self._widefield_config.crop_width or 500
        img_h = self._widefield_config.crop_height or 375
        self._capture_aspect = img_h / img_w

        self._capture_preview = ClickableLabel("No capture")
        self._capture_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._capture_preview.clicked.connect(self._on_capture_preview_clicked)
        self._capture_placeholder_style = f'''
            background-color: {COLORS['bg_light']};
            border: 1px solid {COLORS['border_dark']};
            border-radius: 4px;
            color: {COLORS['text_disabled']};
            font-size: 11px;
        '''
        self._capture_preview.setStyleSheet(self._capture_placeholder_style)
        cap_lay.addWidget(self._capture_preview)
        right.addWidget(cap_group)

        # Segmentation section (always uses the trained U-Net model).
        seg_group = QGroupBox("Segmentation")
        self._seg_group = seg_group
        seg_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        seg_lay = QVBoxLayout(seg_group)
        seg_lay.setContentsMargins(6, 6, 6, 6)

        model_row = QHBoxLayout()
        self._run_seg_btn = QPushButton("Segment")
        self._run_seg_btn.setEnabled(False)
        self._run_seg_btn.setStyleSheet(CYAN_BTN_STYLE)
        self._run_seg_btn.clicked.connect(self._on_run_segmentation)
        model_row.addWidget(self._run_seg_btn, 1)
        self._center_btn = QPushButton("Center")
        self._center_btn.setEnabled(False)
        self._center_btn.setStyleSheet(CYAN_OUTLINE_BTN_STYLE)
        self._center_btn.setToolTip(
            "Move the center of the segmented sample to the center of the live view"
        )
        self._center_btn.clicked.connect(self._on_center_clicked)
        model_row.addWidget(self._center_btn, 1)
        seg_lay.addLayout(model_row)
        self._seg_status = QLabel("No image captured")
        self._seg_status.setStyleSheet(f'color: {COLORS["text_secondary"]}; font-size: 11px;')
        seg_lay.addWidget(self._seg_status)
        right.addWidget(seg_group)

        # Pattern section
        pat_group = QGroupBox("Sample Pattern")
        self._pat_group = pat_group
        pat_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        pat_lay = QVBoxLayout(pat_group)
        pat_lay.setContentsMargins(6, 6, 6, 6)
        pat_lay.setSpacing(8)

        # Normal controls live in their own container so Add/Clear Points can
        # swap the whole panel for a Confirm/Cancel editing view.
        self._pattern_normal = QWidget()
        normal_lay = QVBoxLayout(self._pattern_normal)
        normal_lay.setContentsMargins(0, 0, 0, 0)
        normal_lay.setSpacing(8)

        # Consistent spinbox width and no scroll wheel (wide enough for
        # values like "-1000 um").
        _SPIN_W = 88

        def _make_spin(low, high, val, suffix=""):
            sb = QSpinBox()
            sb.setRange(low, high)
            sb.setValue(val)
            sb.setFixedWidth(_SPIN_W)
            sb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if suffix:
                sb.setSuffix(suffix)
            # Disable scroll-wheel changes
            sb.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            sb.wheelEvent = lambda e: e.ignore()
            return sb

        _LABEL_W = 52

        def _make_label(text):
            lbl = QLabel(text)
            lbl.setFixedWidth(_LABEL_W)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return lbl

        # Pattern type radio buttons
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self._pattern_btn_group = QButtonGroup(self)
        self._pattern_radios = []
        for idx, name in enumerate(["Grid", "Ring", "Line", "Random"]):
            radio = QRadioButton(name)
            if idx == 1:  # Ring is the default pattern
                radio.setChecked(True)
            self._pattern_btn_group.addButton(radio, idx)
            self._pattern_radios.append(radio)
            type_row.addWidget(radio)
        self._pattern_btn_group.idClicked.connect(self._on_pattern_changed)
        normal_lay.addLayout(type_row)

        # Spinboxes for each pattern type's parameters.
        self._grid_rows = _make_spin(1, 200, 3)
        self._grid_cols = _make_spin(1, 200, 3)
        self._grid_buffer = _make_spin(-1000, 1000, 30, " um")
        self._ring_rays = _make_spin(3, 360, 8)
        self._ring_rings = _make_spin(1, 100, 3)
        self._ring_interval = _make_spin(1, 2000, 100, " um")
        self._ring_buffer = _make_spin(-1000, 1000, 30, " um")
        self._line_points = _make_spin(2, 500, 5)
        self._line_buffer = _make_spin(-1000, 1000, 30, " um")
        self._rand_points = _make_spin(1, 2000, 10)

        # Buffer = margin from the mask edge, in MICROMETERS (converted to
        # pixels at generation time using the widefield pixel calibration).
        # Positive keeps points that far INSIDE the mask; negative places them
        # OUTSIDE (beyond the edge).
        for _sb in (self._grid_buffer, self._line_buffer, self._ring_buffer):
            _sb.setToolTip(
                "Margin from the mask edge (um). Negative places points "
                "OUTSIDE the segmented mask."
            )
        self._ring_interval.setToolTip("Spacing between concentric rings (um).")

        def _make_param_page(rows):
            """Build a params page from rows of (label, spinbox) pairs.

            Each row is left-aligned with a trailing stretch and uniform
            spacing so every pattern type lines up the same way.
            """
            page = QWidget()
            vbox = QVBoxLayout(page)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(6)
            for row in rows:
                hbox = QHBoxLayout()
                hbox.setSpacing(6)
                for label, spin in row:
                    hbox.addWidget(_make_label(label))
                    hbox.addWidget(spin)
                hbox.addStretch()
                vbox.addLayout(hbox)
            return page

        # Stacked widget that adopts the current page's height (no gap).
        self._pattern_stack = _AutoSizeStack()
        self._pattern_stack.addWidget(_make_param_page([
            [("Rows:", self._grid_rows), ("Cols:", self._grid_cols)],
            [("Buffer:", self._grid_buffer)],
        ]))
        self._pattern_stack.addWidget(_make_param_page([
            [("Rays:", self._ring_rays), ("Rings:", self._ring_rings)],
            [("Interval:", self._ring_interval), ("Buffer:", self._ring_buffer)],
        ]))
        self._pattern_stack.addWidget(_make_param_page([
            [("Points:", self._line_points), ("Buffer:", self._line_buffer)],
        ]))
        self._pattern_stack.addWidget(_make_param_page([
            [("Points:", self._rand_points)],
        ]))
        self._pattern_stack.setCurrentIndex(1)  # Ring (matches default radio)
        # Marker controls, with the "Show #" toggle stacked directly under
        # the marker-size box:
        #       Marker: [size] um
        #               [Show #]
        # Marker diameter is fixed at MARKER_DIAMETER_UM (the XY step limit);
        # only its on-screen pixel size is derived from the pixel calibration.
        marker_value = QLabel(f"{int(MARKER_DIAMETER_UM)} um")
        marker_value.setToolTip(
            "Marker diameter is fixed at the XY step limit. Markers overlap "
            "when two points are closer than this (i.e. too close to resolve)."
        )
        self._show_labels_chk = QCheckBox("Show #")
        self._show_labels_chk.setChecked(False)
        self._show_labels_chk.setToolTip("Show the point number next to each marker")
        self._show_labels_chk.toggled.connect(lambda _: self._update_capture_preview())

        marker_grid = QGridLayout()
        marker_grid.setContentsMargins(0, 0, 0, 0)
        marker_grid.setHorizontalSpacing(6)
        marker_grid.setVerticalSpacing(4)
        marker_grid.addWidget(QLabel("Marker Size:"), 0, 0)
        marker_grid.addWidget(marker_value, 0, 1)

        # Keep the marker block pinned to the top of the right column.
        marker_holder = QVBoxLayout()
        marker_holder.addLayout(marker_grid)
        marker_holder.addStretch()

        # Pattern params (left) and marker controls (right) share one row,
        # so the marker block fills the otherwise-empty right side instead
        # of taking its own row below.
        params_row = QHBoxLayout()
        params_row.setSpacing(8)
        params_row.addWidget(self._pattern_stack, 1, Qt.AlignmentFlag.AlignTop)
        params_row.addLayout(marker_holder)
        normal_lay.addLayout(params_row)

        self._gen_pattern_btn = QPushButton("Generate Points")
        self._gen_pattern_btn.setEnabled(False)
        self._gen_pattern_btn.setStyleSheet(CYAN_BTN_STYLE)
        self._gen_pattern_btn.clicked.connect(self._on_generate_pattern)
        normal_lay.addWidget(self._gen_pattern_btn)

        # Manual point editing: each enters a Confirm/Cancel editing mode.
        point_row = QHBoxLayout()
        self._add_point_btn = QPushButton("Add Point")
        self._add_point_btn.setStyleSheet(CYAN_OUTLINE_BTN_STYLE)
        self._add_point_btn.setToolTip(
            "Add custom points by clicking the captured image"
        )
        self._add_point_btn.clicked.connect(lambda: self._enter_edit_mode('add'))
        point_row.addWidget(self._add_point_btn)
        self._clear_points_btn = QPushButton("Clear Points")
        self._clear_points_btn.setStyleSheet(CYAN_OUTLINE_BTN_STYLE)
        self._clear_points_btn.setToolTip(
            "Remove points by clicking them on the captured image"
        )
        self._clear_points_btn.clicked.connect(lambda: self._enter_edit_mode('delete'))
        point_row.addWidget(self._clear_points_btn)
        normal_lay.addLayout(point_row)

        # Status label (left) and the "Show #" toggle (right edge), on the
        # row below the Add/Clear buttons.
        status_row = QHBoxLayout()
        self._pattern_status = QLabel("Run segmentation first")
        self._pattern_status.setStyleSheet(f'color: {COLORS["text_secondary"]}; font-size: 11px;')
        status_row.addWidget(self._pattern_status)
        status_row.addStretch()
        status_row.addWidget(self._show_labels_chk)
        normal_lay.addLayout(status_row)

        pat_lay.addWidget(self._pattern_normal)

        # Editing view: shown while adding/deleting points — just a hint and
        # Confirm/Cancel. Cancel reverts the points changed in this session.
        self._pattern_edit = QWidget()
        edit_lay = QVBoxLayout(self._pattern_edit)
        edit_lay.setContentsMargins(0, 0, 0, 0)
        edit_lay.setSpacing(8)
        self._edit_hint = QLabel("")
        self._edit_hint.setWordWrap(True)
        self._edit_hint.setStyleSheet(
            f'color: {COLORS["text_secondary"]}; font-size: 11px;'
        )
        edit_lay.addWidget(self._edit_hint)
        edit_btn_row = QHBoxLayout()
        self._edit_confirm_btn = QPushButton("Confirm")
        self._edit_confirm_btn.clicked.connect(self._on_edit_confirm)
        self._edit_cancel_btn = QPushButton("Cancel")
        self._edit_cancel_btn.clicked.connect(self._on_edit_cancel)
        edit_btn_row.addWidget(self._edit_confirm_btn, 1)
        edit_btn_row.addWidget(self._edit_cancel_btn, 1)
        edit_lay.addLayout(edit_btn_row)
        self._pattern_edit.setVisible(False)
        pat_lay.addWidget(self._pattern_edit)

        right.addWidget(pat_group)

        right.addStretch()

        # Wrap right panel in a scroll area so it scrolls if window is very short
        self._right_container = right_container
        scroll = QScrollArea()
        scroll.setWidget(right_container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFixedWidth(450)
        layout.addWidget(scroll)

    def preferred_content_height(self) -> int:
        """Height needed to show the whole live-view block without scrolling.

        The right column (captured-image preview + segmentation + pattern
        controls) is the tallest part and lives in a scroll area, so its own
        size hint doesn't propagate. This reports that content height — with
        the Ring pattern selected (the tallest pattern page) — so the main
        window can size the splitter to fit everything by default.
        """
        # Make sure the preview placeholder has its aspect-correct height
        # before measuring (it's set lazily on first show).
        self._update_placeholder_size()
        self._right_container.layout().activate()
        content = self._right_container.sizeHint().height()
        margins = self.layout().contentsMargins()
        # Small buffer so rounding / the splitter handle can never leave the
        # block one pixel short (which would re-trigger the scrollbar).
        return content + margins.top() + margins.bottom() + 8

    def _create_image_overlay(self):
        """Contrast / Smooth / Crosshair controls floating in the live image's
        top-right corner, keeping the bottom control row compact."""
        self._image_overlay = QWidget(self._display)
        self._image_overlay.setObjectName('imageOverlay')
        self._image_overlay.setStyleSheet(f'''
            QWidget#imageOverlay {{
                background-color: rgba(20, 20, 20, 175);
                border: 1px solid {COLORS['border_dark']};
                border-radius: 6px;
            }}
        ''')
        row = QHBoxLayout(self._image_overlay)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(6)

        row.addWidget(QLabel("Contrast:"))
        self._contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self._contrast_slider.setRange(0, 10)
        self._contrast_slider.setValue(0)
        self._contrast_slider.setFixedWidth(80)
        self._contrast_slider.setToolTip("CLAHE adaptive contrast (0 = off)")
        row.addWidget(self._contrast_slider)
        self._contrast_label = QLabel("off")
        self._contrast_label.setFixedWidth(24)
        row.addWidget(self._contrast_label)
        self._contrast_slider.valueChanged.connect(
            lambda v: self._contrast_label.setText("off" if v == 0 else str(v))
        )

        row.addWidget(QLabel("Smooth:"))
        self._smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self._smooth_slider.setRange(0, 10)
        self._smooth_slider.setValue(0)
        self._smooth_slider.setFixedWidth(80)
        self._smooth_slider.setToolTip("Gaussian smoothing (0 = off)")
        row.addWidget(self._smooth_slider)
        self._smooth_label = QLabel("off")
        self._smooth_label.setFixedWidth(24)
        row.addWidget(self._smooth_label)
        self._smooth_slider.valueChanged.connect(
            lambda v: self._smooth_label.setText("off" if v == 0 else str(v * 2 + 1))
        )

        self._crosshair_check = QCheckBox("Crosshair")
        self._crosshair_check.setChecked(True)
        self._crosshair_check.toggled.connect(self._display.set_crosshair_visible)
        row.addWidget(self._crosshair_check)

        # Reposition the overlay whenever the image display resizes/shows.
        self._display.installEventFilter(self)
        self._reposition_image_overlay()

    def _reposition_image_overlay(self):
        """Anchor the controls overlay to the image's top-right corner."""
        if not hasattr(self, '_image_overlay'):
            return
        self._image_overlay.adjustSize()
        margin = 8
        x = max(margin, self._display.width() - self._image_overlay.width() - margin)
        self._image_overlay.move(x, margin)
        self._image_overlay.raise_()

    def eventFilter(self, obj, event):
        if (obj is self._display
                and event.type() in (QEvent.Type.Resize, QEvent.Type.Show)):
            self._reposition_image_overlay()
        return super().eventFilter(obj, event)

    def _update_placeholder_size(self):
        """Set placeholder height to match the camera image aspect ratio."""
        if self._captured_image is None:
            w = self._capture_preview.width()
            if w > 0:
                self._capture_preview.setFixedHeight(int(w * self._capture_aspect))

    def showEvent(self, event):
        super().showEvent(event)
        self._update_placeholder_size()

    def _connect_signals(self):
        if self._widefield_thread:
            self._widefield_thread.frame_ready.connect(self._on_wide_frame)
            self._widefield_thread.connected.connect(self._on_wide_connected)

        if self._objective_thread:
            self._objective_thread.frame_ready.connect(self._on_obj_frame)
            self._objective_thread.connected.connect(self._on_obj_connected)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast and smoothing (display-only, does not modify source)."""
        contrast = self._contrast_slider.value()
        smooth = self._smooth_slider.value()

        if contrast == 0 and smooth == 0:
            return frame

        img = frame
        if contrast > 0:
            # CLAHE: adaptive histogram equalization for local contrast
            # clip_limit controls enhancement strength (higher = more contrast)
            clip_limit = contrast / 2.0  # range 0.5 to 5.0
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            if img.ndim == 2:
                img = clahe.apply(img)
            else:
                # Apply to luminance channel only
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        if smooth > 0:
            ksize = smooth * 2 + 1
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        return img

    def _on_wide_frame(self, frame: np.ndarray, ts: float):
        if self._current_camera == 'wide':
            self._display.update_image(self._process_frame(frame))

    def _on_obj_frame(self, frame: np.ndarray, ts: float):
        if self._current_camera == 'objective':
            self._display.update_image(self._process_frame(frame))

    def _on_wide_connected(self):
        if self._current_camera == 'wide':
            self._status_label.setText("Wide FOV")
            self._status_label.setStyleSheet(f'color: {COLORS["success"]};')
        # Camera auto-starts acquisition on connect
        self._acquiring = True
        self._acq_btn.setEnabled(True)
        self._update_acq_btn(True)
        self._acq_btn.setChecked(True)

    def _on_obj_connected(self):
        if self._current_camera == 'objective':
            self._status_label.setText("Objective")
            self._status_label.setStyleSheet(f'color: {COLORS["success"]};')
        # Enable play button if not already enabled
        self._acq_btn.setEnabled(True)

    def _on_switch_cam_clicked(self):
        target = 'objective' if self._current_camera == 'wide' else 'wide'
        self.camera_switch_requested.emit(target)

    def switch_camera(self, camera: str):
        """Switch displayed camera feed. Both cameras keep acquiring."""
        self._current_camera = camera
        if camera == 'wide':
            self._switch_cam_btn.setText("Switch to Objective")
            self._status_label.setText("Wide FOV")
        else:
            self._switch_cam_btn.setText("Switch to Widefield")
            self._status_label.setText("Objective")
        self._status_label.setStyleSheet(f'color: {COLORS["success"]};')

        # Segmentation and pattern generation are widefield-only operations.
        is_wide = (camera == 'wide')
        # Leaving widefield mid-edit would strand the Confirm/Cancel view in a
        # disabled panel — cancel (revert) the edit first.
        if not is_wide and self._edit_mode is not None:
            self._on_edit_cancel()
        self._seg_group.setEnabled(is_wide)
        self._pat_group.setEnabled(is_wide)
        # Capture grabs the widefield frame, so it's disabled (greyed) in
        # objective view.
        self._capture_btn.setEnabled(is_wide)

        self._update_scale_bar()

    def _update_scale_bar(self):
        """Set the live-view scale bar for the displayed camera.

        500 um for the wide FOV, 100 um for the objective; length in pixels
        is derived from that camera's object-plane pixel size.
        """
        if self._current_camera == 'wide':
            self._display.set_scale_bar(500.0, self._widefield_config.pixel_size_x_um)
        else:
            self._display.set_scale_bar(100.0, self._objective_config.pixel_size_x_um)

    def refresh_calibration(self):
        """Re-read the pixel-size calibration and update overlays.

        Call after the user edits camera pixel sizes in Settings so the scale
        bar length and the proximity marker (both derived from um/pixel) update
        without needing to re-capture or switch cameras.
        """
        self._update_scale_bar()
        self._update_capture_preview()

    def _on_acquisition_toggled(self, checked: bool):
        if checked:
            self._start_acquisition()
        else:
            self._stop_acquisition()

    @staticmethod
    def _make_play_icon(color: str, size: int = 20) -> QIcon:
        """Draw a play triangle with the given color."""
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
        """Update acquisition button icon and style."""
        if running:
            icon = self._make_pause_icon(COLORS['bg_dark'])
            self._acq_btn.setIcon(icon)
            self._acq_btn.setStyleSheet(self._acq_btn_filled)
        else:
            icon = self._make_play_icon(COLORS['accent_primary'])
            self._acq_btn.setIcon(icon)
            self._acq_btn.setStyleSheet(self._acq_btn_outline)

    def _start_acquisition(self):
        for thread in [self._widefield_thread, self._objective_thread]:
            if thread and thread.is_connected:
                thread.start_acquisition()
        self._update_acq_btn(True)
        self._acquiring = True

    def _stop_acquisition(self):
        for thread in [self._widefield_thread, self._objective_thread]:
            if thread:
                thread.stop_acquisition()
        self._update_acq_btn(False)
        self._acq_btn.setChecked(False)
        self._acquiring = False

    def _on_exposure_changed(self):
        try:
            val = float(self._exposure_entry.text())
        except ValueError:
            return
        thread = self._widefield_thread if self._current_camera == 'wide' else self._objective_thread
        if thread:
            thread.set_exposure(val / 1000.0)

    def _on_autofocus_clicked(self):
        """Request camera autofocus; the main window supplies the Prusa Z
        range (from the Autofocus panel) and drives the scan."""
        self.camera_autofocus_requested.emit(self._af_steps_spin.value())

    def set_autofocus_enabled(self, enabled: bool):
        """Enable/disable the camera autofocus button."""
        self._autofocus_btn.setEnabled(enabled)

    def current_frame(self):
        """Latest raw frame from the active camera (for autofocus scoring)."""
        thread = (self._widefield_thread if self._current_camera == 'wide'
                  else self._objective_thread)
        if thread is None:
            return None
        return thread.get_latest_frame()

    def _on_image_clicked(self, x: int, y: int):
        self.position_clicked.emit(self._current_camera, x, y)

    def _on_capture_preview_clicked(self, x: int, y: int):
        """Handle a click on the captured image.

        Three modes:
          * edit 'add'    -> add a custom sample point at the click.
          * edit 'delete' -> remove the point nearest the click.
          * otherwise     -> move the stage to the nearest point.
        """
        coords = self._label_to_image_coords(x, y)
        if coords is None:
            return
        img_x, img_y = coords

        if self._edit_mode == 'add':
            self._add_custom_point(img_x, img_y)
        elif self._edit_mode == 'delete':
            self._delete_nearest_point(img_x, img_y)
        else:
            self._move_to_nearest_point(img_x, img_y)

    def _label_to_image_coords(self, bx: int, by: int):
        """Map a base-pixmap click (from ClickableLabel) to image pixel coords.

        The preview emits coordinates in its composited base pixmap (which
        already accounts for zoom/pan), so this just rescales to the captured
        image. Returns (img_x, img_y) floats, or None if out of range.
        """
        if self._captured_image is None:
            return None
        base = self._capture_preview.base_pixmap()
        if base is None or base.isNull():
            return None
        if not (0 <= bx < base.width() and 0 <= by < base.height()):
            return None

        img_h, img_w = self._captured_image.shape[:2]
        return (bx * img_w / base.width(), by * img_h / base.height())

    def _move_to_nearest_point(self, img_x: float, img_y: float):
        """Drive the stage to the generated point nearest the click."""
        # Clicking a point moves the stage. Before the stage is referenced
        # that can't happen, so ignore the click entirely (no move, and no
        # selection highlight) to avoid misleading the user.
        if not self._stage_referenced:
            self._pattern_status.setText("Reference the stage to move to points")
            return
        # Safety: only allow moves while the probe (green tracker) is inside the
        # captured area. If it's off, the probe could be far away / at an unsafe
        # Z, and an XY move from there could crash it.
        if not self._probe_in_area:
            self._pattern_status.setText(
                "Probe is outside the captured area — move back in to click points")
            return
        if not self._generated_positions:
            return

        nearest = min(
            self._generated_positions,
            key=lambda p: (p.pixel_x - img_x) ** 2 + (p.pixel_y - img_y) ** 2,
        )
        dist = ((nearest.pixel_x - img_x) ** 2
                + (nearest.pixel_y - img_y) ** 2) ** 0.5

        # Only act on clicks reasonably close to a point (image pixels).
        img_w = self._captured_image.shape[1]
        tol = max(20.0, img_w * 0.03)
        if dist > tol:
            return

        # Don't change the point's marker — the live green probe marker shows
        # where we are; clicking just drives the stage to the point.
        self._pattern_status.setText(
            f"Move to point {nearest.id}: "
            f"({nearest.stage_x:.2f}, {nearest.stage_y:.2f}) mm"
        )
        logger.info(
            f"Move-to-point {nearest.id}: pixel=({nearest.pixel_x:.1f}, "
            f"{nearest.pixel_y:.1f}) -> widefield stage "
            f"({nearest.stage_x:.3f}, {nearest.stage_y:.3f}, {nearest.stage_z:.3f})"
        )
        self.move_to_point_requested.emit(
            nearest.stage_x, nearest.stage_y, nearest.stage_z
        )

    def _add_custom_point(self, img_x: float, img_y: float):
        """Add a custom sample point at the clicked image pixel."""
        if self._captured_image is None:
            return

        cal = self._make_calibration()
        if cal is None:
            return

        gen = PatternGenerator(cal)
        sx, sy, sz = gen.pixel_to_stage(img_x, img_y)

        new_id = len(self._generated_positions) + 1
        pos = SamplePosition(
            id=new_id,
            pixel_x=img_x,
            pixel_y=img_y,
            stage_x=sx,
            stage_y=sy,
            stage_z=sz,
            label=f"Custom_{new_id}",
        )
        # Keep _sample_points (drawn) and _generated_positions in lockstep.
        # Do NOT select it (stays yellow) and do NOT move the stage — adding
        # a point only registers it.
        self._generated_positions.append(pos)
        self._sample_points.append((img_x, img_y))
        self._update_capture_preview()

        self._pattern_status.setText(
            f"{len(self._generated_positions)} points "
            f"(added point {new_id})"
        )
        logger.info(
            f"Added custom point {new_id}: pixel=({img_x:.1f}, {img_y:.1f}) "
            f"-> widefield stage ({sx:.3f}, {sy:.3f}, {sz:.3f})"
        )
        # Republish so the objective coords are recomputed downstream.
        self.positions_generated.emit(self._generated_positions)

    def _delete_nearest_point(self, img_x: float, img_y: float):
        """Remove the sample point nearest the click (within tolerance)."""
        if not self._generated_positions:
            return

        # Nearest point index in image-pixel space.
        best_i, best_d2 = None, None
        for i, (px, py) in enumerate(self._sample_points):
            d2 = (px - img_x) ** 2 + (py - img_y) ** 2
            if best_d2 is None or d2 < best_d2:
                best_i, best_d2 = i, d2

        img_w = self._captured_image.shape[1]
        tol = max(20.0, img_w * 0.03)
        if best_i is None or best_d2 ** 0.5 > tol:
            return

        self._generated_positions.pop(best_i)
        self._sample_points.pop(best_i)
        # Indices shift after removal, so clear the (index-based) selection.
        self._selected_point_id = None
        self._renumber_positions()
        self._update_capture_preview()

        self._pattern_status.setText(
            f"Removed a point ({len(self._generated_positions)} left)"
        )
        logger.info(
            f"Removed point near pixel ({img_x:.1f}, {img_y:.1f}); "
            f"{len(self._generated_positions)} remain"
        )
        # Republish the updated list so downstream consumers stay in sync.
        self.positions_generated.emit(self._generated_positions)

    def _renumber_positions(self):
        """Reassign point ids to match list order (1-based)."""
        for i, p in enumerate(self._generated_positions):
            p.id = i + 1

    def _enter_edit_mode(self, mode: str):
        """Enter point add/delete mode: hide other panels, show Confirm/Cancel.

        Snapshots the current points so Cancel can revert this session.
        """
        if self._captured_image is None or self._edit_mode is not None:
            return
        self._edit_mode = mode
        # Snapshot (shallow copies of the parallel lists) for revert on Cancel.
        self._edit_snapshot = (
            list(self._generated_positions),
            list(self._sample_points),
        )
        if mode == 'add':
            self._edit_hint.setText("Add points: click the image, then Confirm.")
        else:
            self._edit_hint.setText("Delete points: click points, then Confirm.")

        # Collapse the panel to just Confirm/Cancel and hide segmentation.
        self._seg_group.setVisible(False)
        self._pattern_normal.setVisible(False)
        self._pattern_edit.setVisible(True)

    def _exit_edit_mode(self):
        """Leave edit mode and restore the normal panels."""
        self._edit_mode = None
        self._edit_snapshot = None
        self._pattern_edit.setVisible(False)
        self._pattern_normal.setVisible(True)
        self._seg_group.setVisible(True)

    def _on_edit_confirm(self):
        """Keep the points edited in this session and return to normal."""
        n = len(self._generated_positions)
        self._exit_edit_mode()
        self._pattern_status.setText(f"{n} points")
        self.positions_generated.emit(self._generated_positions)

    def _on_edit_cancel(self):
        """Revert the points changed in this session and return to normal."""
        if self._edit_snapshot is not None:
            gp, sp = self._edit_snapshot
            self._generated_positions = list(gp)
            self._sample_points = list(sp)
            self._renumber_positions()
            self._selected_point_id = None
            self._update_capture_preview()
            self.positions_generated.emit(self._generated_positions)
        self._exit_edit_mode()
        self._pattern_status.setText("Edit cancelled")

    # ---- Capture preview helpers ----

    def _update_capture_preview(self):
        """Render the captured image + overlays into the preview.

        The base pixmap is built at the captured image's FULL resolution (with
        overlays drawn at full res), so zooming into the preview stays sharp.
        The label is sized to a fit-to-width display height; the ClickableLabel
        scales the high-res pixmap down to fit and reveals the detail on zoom.
        Display-fixed sizes (marker outline, point labels) are multiplied by
        the base/display ratio so they look the same on screen.
        """
        if self._captured_image is None:
            return

        image = self._captured_image
        if image.ndim == 2:
            h, w = image.shape
            qimage = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, c = image.shape
            qimage = QImage(image.data, w, h, w * c, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage)  # full resolution (fromImage copies)

        available_w = self._capture_preview.width() or 260
        ss = pixmap.width() / float(available_w)  # base px per display px

        # Draw overlays at full resolution
        if self._current_mask is not None or self._sample_points:
            scale_x = pixmap.width() / w
            scale_y = pixmap.height() / h
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            if self._current_mask is not None:
                mh, mw = self._current_mask.shape[:2]
                rgba = np.zeros((mh, mw, 4), dtype=np.uint8)
                rgba[self._current_mask > 0] = [30, 144, 255, 100]
                rgba = np.ascontiguousarray(rgba)
                overlay_qi = QImage(rgba.data, mw, mh, mw * 4, QImage.Format.Format_RGBA8888)
                overlay_pm = QPixmap.fromImage(overlay_qi).scaled(
                    pixmap.size(), Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                painter.drawPixmap(0, 0, overlay_pm)

            if self._sample_points:
                # Marker radius from a physical DIAMETER (60 um); scales with
                # the image so two markers overlap exactly when closer than it.
                px_um = self._widefield_config.pixel_size_x_um or 1.0
                r = max(2.0, (MARKER_DIAMETER_UM / 2.0) / px_um * scale_x)
                show_labels = self._show_labels_chk.isChecked()
                label_font = QFont("Segoe UI", max(6, int(round(6 * ss))))
                label_font.setBold(True)
                for idx, (px, py) in enumerate(self._sample_points):
                    sx, sy = px * scale_x, py * scale_y
                    # (selection highlight is unused — clicking just moves)
                    selected = (self._selected_point_id == idx + 1)
                    color = QColor('#2ecc71') if selected else QColor('#f39c12')
                    rr = r + 3 * ss if selected else r
                    pen = QPen(color)
                    pen.setWidthF(max(1.0, 2.0 * ss))
                    painter.setPen(pen)
                    painter.setBrush(QBrush(color))
                    painter.drawEllipse(QPointF(sx, sy), rr, rr)

                    if show_labels:
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                        painter.setFont(label_font)
                        painter.drawText(QPointF(sx + rr + 2 * ss, sy - rr), str(idx + 1))

            painter.end()

        self._capture_preview.setStyleSheet('')
        self._capture_preview.setText('')
        # Label height is the fit-to-width display height; the base pixmap is
        # full-res and gets scaled to fit by the ClickableLabel.
        self._capture_preview.setFixedHeight(int(round(available_w * h / w)))
        self._capture_preview.set_base_pixmap(pixmap)
        self.update_probe_marker()

    # ---- Capture ----

    def _capture_current_frame(self) -> bool:
        """Grab the latest widefield frame as the captured image.

        Records the stage position at capture time and clears any previous
        mask/points. Returns True on success. Shared by the Capture button
        and the auto-center loop.
        """
        thread = self._widefield_thread
        if not thread:
            return False
        frame = thread.get_latest_frame()
        if frame is None:
            return False

        self._captured_image = self._process_frame(frame.copy())
        self._captured_camera = 'wide'
        self._capture_stage_position = self._position_provider()

        self._current_mask = None
        self._sample_points = []
        self._generated_positions = []
        self._selected_point_id = None
        self._capture_preview.reset_view()  # fresh image → start un-zoomed
        self._update_capture_preview()
        # Center is available with a captured image AND a referenced stage.
        self._update_center_enabled()
        self._gen_pattern_btn.setEnabled(False)  # needs a mask
        return True

    def _on_capture(self):
        """Capture current widefield frame for segmentation."""
        if self._capture_current_frame():
            self._run_seg_btn.setEnabled(True)
            pos = self._capture_stage_position
            img = self._captured_image
            self._seg_status.setText(
                f"Captured: {img.shape[1]}x{img.shape[0]} "
                f"@ ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            )
            logger.info(
                f"Image captured for segmentation, stage pos: "
                f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            )
        else:
            self._seg_status.setText("No camera image available")

    # ---- Segmentation ----

    def _on_run_segmentation(self):
        """Run U-Net segmentation in a background thread (keeps GUI responsive).

        The trained U-Net is loaded once (cached) and reused on subsequent
        runs. Progress is reported to the status-bar progress bar so the
        load — which takes several seconds — clearly isn't frozen.
        """
        if self._captured_image is None:
            return
        if self._seg_worker is not None and self._seg_worker.isRunning():
            return  # already running

        # Lazily create the cached segmenter (uses the default trained U-Net).
        if self._segmenter is None:
            from raman_control.segmentation import UNetSegmenter
            self._segmenter = UNetSegmenter()

        self._run_seg_btn.setEnabled(False)
        self._seg_status.setText("Loading U-Net…")
        self.segmentation_progress.emit(5, "Segmentation…")

        worker = SegmentationWorker(self._segmenter, self._captured_image.copy())
        worker.progress.connect(self._on_seg_progress)
        worker.finished_mask.connect(self._on_seg_finished)
        worker.failed.connect(self._on_seg_failed)
        self._seg_worker = worker
        worker.start()

    def _on_seg_progress(self, percent: int, message: str):
        """Forward worker progress to the status bar and seg status label."""
        self.segmentation_progress.emit(percent, message)
        self._seg_status.setText(message)

    def _on_seg_finished(self, mask):
        """Handle the segmentation result (runs on the GUI thread)."""
        # Keep Segment disabled mid auto-center; _finish_centering restores it.
        if not self._centering_active:
            self._run_seg_btn.setEnabled(True)
        self.segmentation_progress.emit(0, "")  # clear progress bar

        if mask is not None:
            self._current_mask = mask
            self._update_capture_preview()
            self._gen_pattern_btn.setEnabled(True)
            # Don't re-enable Center mid auto-center; _finish_centering does it.
            self._update_center_enabled()
            area_px = int(np.sum(mask))
            px_area_um2 = (self._widefield_config.pixel_size_x_um
                           * self._widefield_config.pixel_size_y_um)
            area_mm2 = area_px * px_area_um2 / 1e6
            self._seg_status.setText(f"Area: {area_px} px  ({area_mm2:.4f} mm²)")
            if not self._centering_active:
                self._pattern_status.setText("Ready to generate pattern")
        else:
            self._seg_status.setText("No object detected")

        self._seg_worker = None

        # Drive the auto-center loop with this result, if active.
        if self._centering_active:
            if self._center_phase == 'initial_segment':
                # First mask obtained on the Center click -> start the moves.
                if mask is None:
                    self._finish_centering("no object detected")
                else:
                    self._center_phase = 'loop'
                    self._do_center_move()
            else:
                # Segmentation that followed a centering move -> check edge.
                self._evaluate_centering(mask)

    def _on_seg_failed(self, message: str):
        """Handle a segmentation/load failure (runs on the GUI thread)."""
        self._run_seg_btn.setEnabled(True)
        self.segmentation_progress.emit(0, "")
        self._seg_status.setText(f"Error: {message[:40]}")
        logger.error(f"Segmentation error: {message}")
        self._seg_worker = None

    def _update_center_enabled(self):
        """Center needs a captured image, a referenced stage, AND the probe
        inside the captured area (it moves the stage; out-of-area means the
        probe may be far away / at an unsafe Z — same guard as point clicks)."""
        self._center_btn.setEnabled(
            self._captured_image is not None
            and self._stage_referenced
            and self._probe_in_area
            and not self._centering_active
        )

    def _on_center_clicked(self):
        """Auto-center the sample in the live view.

        Segments the captured image first (so the user need not click Segment),
        then loops: move so the mask centroid is at the FOV center -> wait for
        the move -> re-capture and re-segment -> if the new mask still touches
        an image edge, repeat. Capped at center_max_repeats (camera settings).
        """
        if self._captured_image is None or not self._stage_referenced:
            return
        if self._centering_active:
            return

        self._centering_active = True
        self._center_repeats = 0
        self._center_max_repeats = max(1, int(self._widefield_config.center_max_repeats))
        self._center_btn.setEnabled(False)
        self._run_seg_btn.setEnabled(False)

        # Segment the current image first; the result kicks off the move.
        self._center_phase = 'initial_segment'
        self._on_run_segmentation()

    def _do_center_move(self):
        """Compute the current mask centroid and request a centering move."""
        cal = self._make_calibration()
        if cal is None or self._current_mask is None:
            self._finish_centering("no mask")
            return
        ys, xs = np.where(self._current_mask == 1)
        if xs.size == 0:
            self._finish_centering("no mask")
            return

        cx = float(xs.mean())
        cy = float(ys.mean())
        sx, sy, sz = PatternGenerator(cal).pixel_to_stage(cx, cy)

        self._center_repeats += 1
        self._pattern_status.setText(
            f"Centering sample (attempt {self._center_repeats}/"
            f"{self._center_max_repeats})..."
        )
        logger.info(
            f"Auto-center attempt {self._center_repeats}: centroid "
            f"({cx:.1f}, {cy:.1f}) -> stage ({sx:.3f}, {sy:.3f})"
        )
        # Wait for this move's movement_complete (with a watchdog in case
        # the stage isn't connected/referenced and no move happens).
        self._center_awaiting_move = True
        self._center_watchdog.start(20000)
        self.move_to_point_requested.emit(sx, sy, sz)

    def on_stage_move_finished(self):
        """Called (by the main window) when a stage move completes.

        Drives the next step of the auto-center loop: let the camera settle,
        then re-capture and re-segment.
        """
        if not (self._centering_active and self._center_awaiting_move):
            return
        self._center_awaiting_move = False
        self._center_watchdog.stop()
        # Give the camera a moment to deliver a frame from the new position.
        QTimer.singleShot(300, self._center_recapture_and_segment)

    def _center_recapture_and_segment(self):
        if not self._centering_active:
            return
        if not self._capture_current_frame():
            self._finish_centering("no camera image")
            return
        # Re-segment; result handled in _on_seg_finished -> _evaluate_centering.
        self._on_run_segmentation()

    def _evaluate_centering(self, mask):
        """Decide whether the auto-center loop continues (after re-segment)."""
        if mask is None:
            self._finish_centering("no object detected")
            return
        if not self._mask_touches_edge(mask):
            self._finish_centering("sample centered")
            return
        if self._center_repeats >= self._center_max_repeats:
            self._finish_centering("max attempts reached, still at edge")
            return
        # Still touching an edge and under the cap -> center again.
        self._do_center_move()

    def _finish_centering(self, reason: str):
        """End the auto-center loop and restore button states."""
        self._centering_active = False
        self._center_phase = None
        self._center_awaiting_move = False
        self._center_watchdog.stop()
        # Center stays available with a captured image and a referenced stage.
        self._update_center_enabled()
        self._run_seg_btn.setEnabled(self._captured_image is not None)
        self._pattern_status.setText(f"Center: {reason}")
        logger.info(f"Auto-center finished: {reason}")

    @staticmethod
    def _mask_touches_edge(mask) -> bool:
        """True if any mask pixel lies on the image border."""
        return bool(
            mask[0, :].any() or mask[-1, :].any()
            or mask[:, 0].any() or mask[:, -1].any()
        )

    # ---- Pattern generation ----

    def _make_calibration(self) -> Optional[CalibrationParams]:
        """Build pixel->stage calibration from the captured image + capture pose.

        Center-relative model: the captured-image center corresponds to the
        stage position recorded at capture time. Shared by pattern
        generation and manual point adding so both use identical math.
        """
        if self._captured_image is None:
            return None
        captured_h, captured_w = self._captured_image.shape[:2]
        return CalibrationParams(
            pixel_size_x_um=self._widefield_config.pixel_size_x_um,
            pixel_size_y_um=self._widefield_config.pixel_size_y_um,
            image_width=captured_w,
            image_height=captured_h,
            capture_stage_x=self._capture_stage_position[0],
            capture_stage_y=self._capture_stage_position[1],
            capture_stage_z=self._capture_stage_position[2],
            invert_y=True,
        )

    def update_probe_marker(self):
        """Place the green "you are here" marker at the current probe position
        on the captured image, or hide it if we've moved outside the captured
        area. Call this on every stage-position update."""
        cp = self._capture_preview
        base = cp.base_pixmap()
        cal = self._make_calibration()
        if cal is None or base is None or base.isNull():
            self._probe_in_area = False
            cp.set_probe_marker(None)
        else:
            lx, ly, _ = self._looking_at_provider()
            px, py = PatternGenerator(cal).stage_to_pixel(lx, ly)
            img_h, img_w = self._captured_image.shape[:2]
            if 0 <= px < img_w and 0 <= py < img_h:
                scale_x = base.width() / img_w
                bx = px * scale_x
                by = py * base.height() / img_h
                # Fixed physical size (MARKER_DIAMETER_UM = 60 µm), base-pixmap px.
                px_um = self._widefield_config.pixel_size_x_um or 1.0
                r_base = (MARKER_DIAMETER_UM / 2.0) / px_um * scale_x
                self._probe_in_area = True
                cp.set_probe_marker((bx, by, r_base))
            else:
                self._probe_in_area = False
                cp.set_probe_marker(None)  # moved outside the captured area
        # The probe-in-area state gates the Center button too.
        self._update_center_enabled()

    def _um_to_px(self, um: float) -> float:
        """Convert a physical distance (um) to widefield image pixels.

        Uses the widefield pixel-size calibration, so the same um value maps
        to the right number of pixels even after the system is recalibrated.
        """
        px_um = self._widefield_config.pixel_size_x_um or 1.0
        return um / px_um

    def _on_pattern_changed(self, idx: int):
        self._pattern_stack.setCurrentIndex(idx)

    def _on_generate_pattern(self):
        """Generate sampling points based on pattern and mask, with coordinate conversion."""
        if self._current_mask is None:
            return

        pattern_idx = self._pattern_btn_group.checkedId()
        pattern_names = ['grid', 'ring', 'line', 'random']
        pattern = pattern_names[pattern_idx] if 0 <= pattern_idx < 4 else 'grid'

        mask = self._current_mask
        h, w = mask.shape

        # Get mask pixels
        ys, xs = np.where(mask == 1)
        if len(xs) == 0:
            self._pattern_status.setText("No mask area")
            return

        points = []
        coordinates_orders = []

        if pattern == 'grid':
            points, coordinates_orders = self._generate_grid(mask, xs, ys, h, w)

        elif pattern == 'ring':
            points, coordinates_orders = self._generate_ring(mask, xs, ys, h, w)

        elif pattern == 'line':
            points, coordinates_orders = self._generate_line(mask, xs, ys, h, w)

        elif pattern == 'random':
            points, coordinates_orders = self._generate_random(mask, xs, ys)

        if not points:
            self._pattern_status.setText("No points generated")
            return

        # Display pixel points on image
        self._sample_points = points
        self._update_capture_preview()

        # Create PatternGenerator and convert pixel -> stage
        gen = PatternGenerator(self._make_calibration())
        positions = gen.from_centroids(
            points,
            coordinates_orders=coordinates_orders if coordinates_orders else None,
        )

        # Keep positions (with stage coords) for click-to-move, parallel
        # to _sample_points. Clear any previous selection.
        self._generated_positions = positions
        self._selected_point_id = None
        self._update_capture_preview()

        self._pattern_status.setText(
            f"Generated {len(positions)} points. Click one to move there"
        )
        self.positions_generated.emit(positions)
        logger.info(f"Generated {len(positions)} sample points ({pattern})")

    def _apply_buffer(self, mask, buffer_px):
        """Shrink or expand the binary mask by ``buffer_px`` pixels.

        buffer_px > 0  -> erode: keep points at least that far INSIDE the edge
        buffer_px < 0  -> dilate: allow points that far OUTSIDE the mask
        buffer_px == 0 -> unchanged
        """
        b = int(buffer_px)
        if b == 0:
            return (mask > 0).astype(np.uint8)
        ksize = 2 * abs(b) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        m = (mask > 0).astype(np.uint8)
        return cv2.erode(m, kernel) if b > 0 else cv2.dilate(m, kernel)

    def _generate_grid(self, mask, xs, ys, h, w):
        """Grid pattern: meshgrid over the buffered mask bbox, keep points inside it."""
        rows_n = self._grid_rows.value()
        cols_n = self._grid_cols.value()

        # Apply the buffer (um -> px): positive shrinks the region, negative
        # expands it so grid points can fall outside the original mask.
        bmask = self._apply_buffer(mask, self._um_to_px(self._grid_buffer.value()))
        bys, bxs = np.where(bmask == 1)
        if bxs.size == 0:
            return [], []

        x_lin = np.linspace(bxs.min(), bxs.max(), cols_n)
        y_lin = np.linspace(bys.min(), bys.max(), rows_n)
        xx, yy = np.meshgrid(x_lin, y_lin)
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()

        # Check which points are inside the buffered mask
        ix = np.clip(np.round(xx_flat).astype(int), 0, w - 1)
        iy = np.clip(np.round(yy_flat).astype(int), 0, h - 1)
        inside = bmask[iy, ix] == 1

        # Compute row/col indices for all candidates, filter by inside
        col_indices = np.tile(np.arange(cols_n), rows_n)
        row_indices = np.repeat(np.arange(rows_n), cols_n)

        points = [(float(xx_flat[i]), float(yy_flat[i])) for i in range(len(xx_flat)) if inside[i]]
        orders = [
            {'row': int(row_indices[i]) + 1, 'col': int(col_indices[i]) + 1}
            for i in range(len(xx_flat)) if inside[i]
        ]
        return points, orders

    def _generate_ring(self, mask, xs, ys, h, w):
        """Ring/radial pattern matching old noodlepy edgedetectorSAM.py."""
        num_rays = self._ring_rays.value()
        num_rings = self._ring_rings.value()
        interval = self._um_to_px(self._ring_interval.value())
        buffer_px = self._um_to_px(self._ring_buffer.value())
        # Validity region: eroded (buffer>0) or dilated (buffer<0) mask, so a
        # negative buffer lets the outer ring sit outside the segmented edge.
        bmask = self._apply_buffer(mask, buffer_px)

        center_x = float(xs.mean())
        center_y = float(ys.mean())

        all_x, all_y = [], []
        ring_numbers, line_numbers = [], []

        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

        for ray_idx, angle in enumerate(angles):
            dx = np.cos(angle)
            dy = np.sin(angle)

            # Find edge distance along this ray
            max_dist = max(w, h)
            edge_dist = 0
            for dist in range(int(max_dist)):
                x_i = int(center_x + dist * dx)
                y_i = int(center_y + dist * dy)
                if x_i < 0 or x_i >= w or y_i < 0 or y_i >= h or mask[y_i, x_i] == 0:
                    edge_dist = dist - 1
                    break

            if edge_dist <= 0:
                continue

            # Place points from the (buffered) edge inward. A negative buffer
            # moves the start beyond the edge so points land outside the mask.
            start_dist = edge_dist - buffer_px
            min_dist_from_center = 10

            for ring_idx in range(num_rings):
                d = start_dist - ring_idx * interval
                if d > min_dist_from_center:
                    px = center_x + d * dx
                    py = center_y + d * dy
                    ipx, ipy = int(px), int(py)
                    if 0 <= ipx < w and 0 <= ipy < h and bmask[ipy, ipx] > 0:
                        all_x.append(px)
                        all_y.append(py)
                        ring_numbers.append(ring_idx + 1)
                        line_numbers.append(ray_idx + 1)

        if not all_x:
            return [], []

        all_x = np.array(all_x)
        all_y = np.array(all_y)
        ring_numbers = np.array(ring_numbers)
        line_numbers = np.array(line_numbers)

        # Sort by angle, rotate to start from min-Y point
        point_angles = np.arctan2(all_y - center_y, all_x - center_x)
        sorted_idx = np.argsort(point_angles)
        all_x = all_x[sorted_idx]
        all_y = all_y[sorted_idx]
        ring_numbers = ring_numbers[sorted_idx]
        line_numbers = line_numbers[sorted_idx]

        min_y_index = int(np.argmin(all_y))
        all_x = np.roll(all_x, -min_y_index)
        all_y = np.roll(all_y, -min_y_index)
        ring_numbers = np.roll(ring_numbers, -min_y_index)
        line_numbers = np.roll(line_numbers, -min_y_index)

        points = [(float(x), float(y)) for x, y in zip(all_x, all_y)]
        orders = [
            {'ring': int(r), 'line': int(l)}
            for r, l in zip(ring_numbers, line_numbers)
        ]
        return points, orders

    def _generate_line(self, mask, xs, ys, h, w):
        """Line pattern: find largest contiguous segment at centroid row (buffered)."""
        num = self._line_points.value()
        # Apply buffer (um -> px): positive shrinks the segment, negative
        # extends it past the original mask edge.
        bmask = self._apply_buffer(mask, self._um_to_px(self._line_buffer.value()))
        cx = int(np.round(xs.mean()))
        cy = int(np.round(ys.mean()))

        # Search outward from centroid for a row with mask pixels
        found_row = None
        sampling_x = None
        for off in range(h):
            for r in ((cy + off, cy - off) if off else (cy,)):
                if 0 <= r < h:
                    cols = np.where(bmask[r] == 1)[0]
                    if cols.size > 0:
                        # Pick largest contiguous segment
                        diffs = np.diff(cols)
                        splits = np.where(diffs > 1)[0]
                        segments = []
                        start = 0
                        for s in splits:
                            segments.append(cols[start:s + 1])
                            start = s + 1
                        segments.append(cols[start:])
                        largest = max(segments, key=len)
                        xmin, xmax = float(largest[0]), float(largest[-1])
                        sampling_x = np.linspace(xmin, xmax, num)
                        found_row = r
                        break
            if found_row is not None:
                break

        if sampling_x is None:
            # Fallback: use full mask horizontal span at centroid row
            xmin, xmax = float(xs.min()), float(xs.max())
            sampling_x = np.linspace(xmin, xmax, num)
            found_row = cy

        points = [(float(x), float(found_row)) for x in sampling_x]
        orders = [{'line_pos': i + 1} for i in range(len(points))]
        return points, orders

    def _generate_random(self, mask, xs, ys):
        """Random pattern: random sample from mask pixels."""
        num = self._rand_points.value()
        n = min(num, len(xs))
        indices = np.random.choice(len(xs), n, replace=False)
        points = [(float(xs[i]), float(ys[i])) for i in indices]
        orders = []  # No meaningful order for random
        return points, orders

    # ---- Clear ----


    def set_stage_referenced(self, all_referenced: bool):
        """Enable/disable FOV switch and Center based on stage reference status."""
        self._stage_referenced = all_referenced
        self._update_center_enabled()  # Center moves the stage
        self._update_switch_enabled()

    def set_autofocus_registered(self, registered: bool):
        """Gate the FOV switch on a registered autofocus range.

        The objective should not be driven onto the sample without a known
        focus range, so 'Switch to Objective' stays disabled until the range
        is registered (and the stage is referenced).
        """
        self._af_registered = registered
        self._update_switch_enabled()

    def _update_switch_enabled(self):
        """FOV switch needs both a referenced stage and a registered AF range."""
        enabled = self._stage_referenced and self._af_registered
        self._switch_cam_btn.setEnabled(enabled)
        if not self._stage_referenced:
            text, color = "REFERENCE THE STAGE TO SWITCH", '#505050'
        elif not self._af_registered:
            text, color = "REGISTER AF RANGE TO SWITCH", '#505050'
        else:
            text, color = "STAGE MOVES ON SWITCH", COLORS['warning']
        self._cam_warning.setText(text)
        self._cam_warning.setStyleSheet(f'''
            color: {color};
            font-size: 10px;
            font-weight: bold;
            padding: 0 8px;
        ''')

    @property
    def current_camera(self) -> str:
        return self._current_camera

    def get_sample_points(self) -> List[Tuple[float, float]]:
        """Get generated sample points (pixel coordinates)."""
        return self._sample_points.copy()
