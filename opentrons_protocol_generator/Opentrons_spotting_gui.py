"""
Opentrons Spotting Manager — PyQt6 edition.

Generates Opentrons OT-2 protocols (Mix / Spot / Combined) for depositing
sample droplets onto quartz slides, plus a JSON manifest and an optional PDF
report. The destination spot grid is aligned with the sample-drop grid used by
the Raman control software, and this UI mirrors that software's dark theme and
layout conventions.

Layout: a controls column and a sample-names column on the left, with the
three deck/plate maps (Source Deck, Destination, Tip Racks) all visible at once
in a grid on the right — no tabs.
"""

import sys
import math
import os
import re
import ast
import json
import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QLineEdit,
    QCheckBox, QScrollArea, QMessageBox,
    QFileDialog, QSizePolicy, QToolTip,
)
from PyQt6.QtCore import Qt, QSize, QRectF, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont

# Optional PDF support
try:
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.pagesizes import letter, landscape, portrait
    from reportlab.lib import colors as pdf_colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================
VALID_WELLS = [
    "B3", "C3", "D3", "G3", "H3", "I3", "J3", "M3", "N3", "O3",
    "A4", "A5", "A10", "A11", "A16", "A17", "A22", "A23",
    "B4", "B5", "B6", "B9", "B10", "B11", "B12", "B15", "B16", "B17", "B18", "B21", "B22", "B23", "B24",
    "C4", "C5", "C6", "C9", "C10", "C11", "C12", "C15", "C16", "C17", "C18", "C21", "C22", "C23", "C24",
    "D4", "D5", "D6", "D9", "D10", "D11", "D12", "D15", "D16", "D17", "D18", "D21", "D22", "D23", "D24",
    "E4", "E5", "E10", "E11", "E16", "E17", "E22", "E23",
    "G4", "G5", "G6", "G9", "G10", "G11", "G12", "G15", "G16", "G17", "G18", "G21", "G22", "G23", "G24",
    "H4", "H5", "H6", "H9", "H10", "H11", "H12", "H15", "H16", "H17", "H18", "H21", "H22", "H23", "H24",
    "I4", "I5", "I6", "I9", "I10", "I11", "I12", "I15", "I16", "I17", "I18", "I21", "I22", "I23", "I24",
    "J4", "J5", "J6", "J9", "J10", "J11", "J12", "J15", "J16", "J17", "J18", "J21", "J22", "J23", "J24",
    "L4", "L5", "L10", "L11", "L16", "L17", "L22", "L23",
    "M4", "M5", "M6", "M9", "M10", "M11", "M12", "M15", "M16", "M17", "M18", "M21", "M22", "M23", "M24",
    "N4", "N5", "N6", "N9", "N10", "N11", "N12", "N15", "N16", "N17", "N18", "N21", "N22", "N23", "N24",
    "O4", "O5", "O6", "O9", "O10", "O11", "O12", "O15", "O16", "O17", "O18", "O21", "O22", "O23", "O24",
    "P4", "P5", "P10", "P11", "P16", "P17", "P22", "P23"
]
VALID_WELLS_SET = set(VALID_WELLS)

# Capacity limits
MAX_SAMPLES = 96                     # 4 source tube racks x 24 tubes
PLATE_CAPACITY = len(VALID_WELLS)    # 192 spots on one destination plate

# Per-sample colors: the 20 discrete colors of matplotlib's tab20 qualitative
# colormap. Sample colors cycle through these every 20 samples, so the 21st
# sample reuses the 1st sample's color, the 22nd the 2nd, etc.
SAMPLE_COLORMAP = 'tab20'
N_BASE_COLORS = 20


def _rgba_to_hex(rgba):
    r, g, b = rgba[0], rgba[1], rgba[2]
    return "#{:02x}{:02x}{:02x}".format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def _build_base_palette():
    """The discrete colors of the tab20 qualitative colormap.

    Falls back to an HSV sweep if matplotlib is unavailable.
    """
    try:
        import matplotlib
        cmap = matplotlib.colormaps[SAMPLE_COLORMAP]
        colors = getattr(cmap, 'colors', None)
        if colors is not None:           # qualitative (ListedColormap)
            return [_rgba_to_hex(c) for c in colors[:N_BASE_COLORS]]
        return [_rgba_to_hex(cmap(i / (N_BASE_COLORS - 1)))
                for i in range(N_BASE_COLORS)]
    except Exception:
        import colorsys
        out = []
        for i in range(N_BASE_COLORS):
            r, g, b = colorsys.hsv_to_rgb((i / N_BASE_COLORS) * 0.85, 1.0, 1.0)
            out.append("#{:02x}{:02x}{:02x}".format(
                int(round(r * 255)), int(round(g * 255)), int(round(b * 255))))
        return out


BASE_PALETTE = _build_base_palette()


def sample_palette(n):
    """Return colors for n samples, cycling through the base colors."""
    n = max(1, int(n))
    return [BASE_PALETTE[i % len(BASE_PALETTE)] for i in range(n)]


# =============================================================================
# THEME — mirrors raman_control/gui/styles.py so the two apps look identical.
# =============================================================================
COLORS = {
    'bg_dark': '#1a1a1a',
    'bg_medium': '#242424',
    'bg_light': '#2d2d2d',
    'bg_elevated': '#363636',
    'border_dark': '#3d3d3d',
    'border_light': '#4d4d4d',
    'border_focus': '#00a0a0',
    'text_primary': '#e0e0e0',
    'text_secondary': '#a0a0a0',
    'text_disabled': '#606060',
    'accent_primary': '#00b4b4',
    'accent_hover': '#00d4d4',
    'accent_pressed': '#008080',
    'success': '#4caf50',
    'warning': '#ff9800',
    'error': '#f44336',
    'info': '#2196f3',
    'selection': '#264f78',
    'highlight': '#3d5a80',
}

DARK_STYLESHEET = f'''
QMainWindow {{ background-color: {COLORS['bg_dark']}; }}
QWidget#scrollContent {{ background-color: {COLORS['bg_dark']}; }}
QFrame {{ border: none; }}

QGroupBox {{
    background-color: {COLORS['bg_medium']};
    border: 1px solid {COLORS['border_dark']};
    border-radius: 6px;
    margin-top: 12px;
    padding: 8px 8px 6px 8px;
    font-weight: 600;
    font-size: 13px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: 2px;
    padding: 0 6px;
    background-color: {COLORS['bg_medium']};
    color: {COLORS['accent_primary']};
}}

QPushButton {{
    background-color: {COLORS['bg_elevated']};
    border: 1px solid {COLORS['border_dark']};
    border-radius: 5px;
    padding: 5px 12px;
    min-height: 18px;
    font-weight: 500;
    color: {COLORS['text_primary']};
}}
QPushButton:hover {{
    background-color: {COLORS['border_light']};
    border-color: {COLORS['accent_primary']};
}}
QPushButton:pressed {{
    background-color: {COLORS['bg_medium']};
    border-color: {COLORS['accent_pressed']};
}}
QPushButton:disabled {{
    background-color: transparent;
    border: 1px solid #4d4d4d;
    color: #505050;
}}
QPushButton[primary="true"] {{
    background-color: {COLORS['accent_primary']};
    border-color: {COLORS['accent_primary']};
    color: {COLORS['bg_dark']};
    font-weight: 600;
}}
QPushButton[primary="true"]:hover {{
    background-color: {COLORS['accent_hover']};
    border-color: {COLORS['accent_hover']};
}}
QPushButton[primary="true"]:pressed {{ background-color: {COLORS['accent_pressed']}; }}

QPushButton[secondaryAction="true"] {{
    background-color: transparent;
    border: 1px solid {COLORS['accent_primary']};
    color: {COLORS['accent_primary']};
    font-weight: 600;
}}
QPushButton[secondaryAction="true"]:hover {{ background-color: #1f3333; }}

QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS['bg_light']};
    border: 1px solid {COLORS['border_dark']};
    border-radius: 4px;
    padding: 6px 10px;
    color: {COLORS['text_primary']};
    selection-background-color: {COLORS['selection']};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COLORS['accent_primary']};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 0; height: 0; border: none;
}}

QLabel, QCheckBox {{
    background: transparent;
    border: none;
    color: {COLORS['text_primary']};
}}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 2px solid {COLORS['text_primary']};
    border-radius: 3px;
    background-color: transparent;
}}
QCheckBox::indicator:hover {{ border-color: white; }}
QCheckBox::indicator:checked {{ background-color: {COLORS['accent_primary']}; border-color: {COLORS['accent_primary']}; }}

QLabel[heading="true"] {{ font-size: 14px; font-weight: 600; color: {COLORS['accent_primary']}; }}
QLabel[secondary="true"] {{ color: {COLORS['text_secondary']}; font-size: 11px; }}
QLabel[warn="true"] {{ color: {COLORS['error']}; font-size: 11px; font-style: italic; }}
QLabel[info="true"] {{ color: {COLORS['info']}; font-size: 12px; }}

QScrollArea {{ border: none; background-color: {COLORS['bg_dark']}; }}

QScrollBar:vertical {{ background-color: {COLORS['bg_dark']}; width: 12px; border: none; }}
QScrollBar::handle:vertical {{ background-color: {COLORS['border_light']}; border-radius: 6px; min-height: 30px; margin: 2px; }}
QScrollBar::handle:vertical:hover {{ background-color: {COLORS['text_secondary']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background-color: {COLORS['bg_dark']}; height: 12px; border: none; }}
QScrollBar::handle:horizontal {{ background-color: {COLORS['border_light']}; border-radius: 6px; min-width: 30px; margin: 2px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

QMessageBox {{ background-color: {COLORS['bg_medium']}; }}
QMessageBox QLabel {{ color: {COLORS['text_primary']}; }}
QDialog {{ background-color: {COLORS['bg_dark']}; }}
QTextEdit {{
    background-color: {COLORS['bg_light']};
    border: 1px solid {COLORS['border_dark']};
    border-radius: 4px;
    color: {COLORS['text_primary']};
    selection-background-color: {COLORS['selection']};
}}
QToolTip {{
    background-color: {COLORS['bg_elevated']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border_light']};
    border-radius: 4px; padding: 6px 10px;
}}
'''


def _generate_checkmark_icon(size: int = 18) -> str:
    """Generate a white checkmark PNG (transparent background) and return its
    path, for use as the checked-checkbox indicator image."""
    import tempfile
    import os
    from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor as _QC
    from PyQt6.QtCore import Qt as _Qt, QPoint

    pixmap = QPixmap(size, size)
    pixmap.fill(_QC(0, 0, 0, 0))
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(_QC('white'))
    pen.setWidth(3)
    pen.setCapStyle(_Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(_Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.drawLine(QPoint(3, 9), QPoint(7, 13))
    p.drawLine(QPoint(7, 13), QPoint(14, 4))
    p.end()

    path = os.path.join(tempfile.gettempdir(), 'ot_spotting_checkmark.png')
    pixmap.save(path)
    return path.replace('\\', '/')


def apply_dark_theme(app):
    """Apply the shared dark palette + stylesheet (mirrors raman_control)."""
    from PyQt6.QtGui import QPalette, QColor as _QC, QFont as _QF
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, _QC(COLORS['bg_dark']))
    palette.setColor(QPalette.ColorRole.WindowText, _QC(COLORS['text_primary']))
    palette.setColor(QPalette.ColorRole.Base, _QC(COLORS['bg_light']))
    palette.setColor(QPalette.ColorRole.AlternateBase, _QC(COLORS['bg_medium']))
    palette.setColor(QPalette.ColorRole.Text, _QC(COLORS['text_primary']))
    palette.setColor(QPalette.ColorRole.Button, _QC(COLORS['bg_elevated']))
    palette.setColor(QPalette.ColorRole.ButtonText, _QC(COLORS['text_primary']))
    palette.setColor(QPalette.ColorRole.Highlight, _QC(COLORS['accent_primary']))
    palette.setColor(QPalette.ColorRole.HighlightedText, _QC(COLORS['bg_dark']))
    palette.setColor(QPalette.ColorGroup.Disabled,
                     QPalette.ColorRole.Text, _QC(COLORS['text_disabled']))
    palette.setColor(QPalette.ColorGroup.Disabled,
                     QPalette.ColorRole.ButtonText, _QC(COLORS['text_disabled']))
    app.setPalette(palette)
    app.setFont(_QF("Segoe UI", 10))
    checkmark = _generate_checkmark_icon()
    app.setStyleSheet(DARK_STYLESHEET + f'''
QCheckBox::indicator:checked {{ image: url({checkmark}); }}
''')


# =============================================================================
# WELL ORDERING — slide by slide (same slide numbering as the Raman sample
# position panel), each slide filled in a row-serpentine: the first row left
# to right, the next right to left, and so on.
# =============================================================================
# Slide clusters: 3 row-groups x 4 column-groups = 12 slides, 16 spots each.
ROW_GROUPS = [
    ['A', 'B', 'C', 'D', 'E'],
    ['G', 'H', 'I', 'J'],
    ['L', 'M', 'N', 'O', 'P'],
]
COL_GROUPS = [
    [3, 4, 5, 6],
    [9, 10, 11, 12],
    [15, 16, 17, 18],
    [21, 22, 23, 24],
]


def _slide_iter():
    """Yield (slide_num, row_letters, col_numbers) per slide, in the same
    snake order the Raman sample panel numbers them (column-group outer,
    row-group inner snake)."""
    slide_num = 0
    for cgi, cols in enumerate(COL_GROUPS):
        rgi_order = (range(len(ROW_GROUPS)) if cgi % 2 == 0
                     else range(len(ROW_GROUPS) - 1, -1, -1))
        for rgi in rgi_order:
            slide_num += 1
            yield slide_num, ROW_GROUPS[rgi], cols


def build_deposition_order():
    """Return (ordered_wells, well_to_slide).

    Wells are listed slide by slide; within a slide the rows are filled in a
    serpentine (first row L->R, next R->L, ...). Only valid wells are kept.
    """
    order = []
    well_slide = {}
    for slide_num, rows, cols in _slide_iter():
        for ri, row in enumerate(rows):
            row_cols = cols if ri % 2 == 0 else list(reversed(cols))
            for col in row_cols:
                well = f"{row}{col}"
                if well in VALID_WELLS_SET:
                    order.append(well)
                    well_slide[well] = slide_num
    return order, well_slide


def build_slide_centers():
    """Return {slide_num: (row_center, col_center)} in 0-based grid units, for
    drawing the slide number behind that slide's circles."""
    centers = {}
    for slide_num, rows, cols in _slide_iter():
        row_idxs = [ord(r) - ord('A') for r in rows]
        centers[slide_num] = (sum(row_idxs) / len(row_idxs),
                              sum(c - 1 for c in cols) / len(cols))
    return centers


DEPOSITION_ORDER, WELL_SLIDE = build_deposition_order()
SLIDE_CENTERS = build_slide_centers()


def _contrast_text(hex_color):
    """Return 'black' or 'white' for legible text on the given fill color."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 'black' if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else 'white'


def tip_index(well):
    """Tip-rack well name -> column-major index (A1=0, B1=1, ... H1=7, A2=8...)."""
    well = well.strip().upper()
    row = ord(well[0]) - ord('A')
    col = int(well[1:]) - 1
    return col * 8 + row


def tip_well(idx):
    """Inverse of tip_index, within a 96-tip rack (8 rows x 12 cols)."""
    idx = max(0, min(95, int(idx)))
    col, row = divmod(idx, 8)
    return f"{chr(ord('A') + row)}{col + 1}"


def grid_dims(n):
    """Arrange n items into a near-square grid; returns (cols, rows)."""
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols) if n else 1
    return cols, rows


# =============================================================================
# NO-WHEEL INPUTS — block accidental value changes from the scroll wheel,
# matching the raman_control convention.
# =============================================================================
class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


# =============================================================================
# MAP WIDGETS — each scales its content to fill its cell (no scrolling needed).
# =============================================================================
class ScalableMap(QWidget):
    """Base for the deck/plate maps: draws in logical coordinates and scales the
    painter so everything fits the widget, centered, preserving aspect ratio."""

    def __init__(self, empty_text="No data", parent=None):
        super().__init__(parent)
        self._empty_text = empty_text
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Painter transform (set each paint) + hover hot-spots in logical coords.
        self._scale = 1.0
        self._tx = 0.0
        self._ty = 0.0
        self._hot = []        # list of (cx, cy, r, tooltip_text)
        self._names = []      # sample names, for tooltips
        self.setMouseTracking(True)

    # --- subclass hooks ---
    def has_data(self) -> bool:
        return False

    def content_size(self):
        return (100.0, 100.0)

    def draw_content(self, painter):
        pass

    # --- tooltips ---
    def set_names(self, names):
        self._names = names

    def _name_for(self, sample_index):
        if 0 <= sample_index < len(self._names):
            return self._names[sample_index]
        return f"patient_{sample_index + 1}"

    def _register(self, cx, cy, r, sample_index, droplet=None, kind=None):
        # Store the sample index (+ optional droplet number / purpose); the name
        # is resolved at hover time so tooltips reflect edits to the sample-name
        # fields without a repaint.
        self._hot.append((cx, cy, r, sample_index, droplet, kind))

    def mouseMoveEvent(self, event):
        pos = event.position()
        text = ""
        if self._scale:
            lx = (pos.x() - self._tx) / self._scale
            ly = (pos.y() - self._ty) / self._scale
            for cx, cy, r, sample_index, droplet, kind in self._hot:
                if (lx - cx) ** 2 + (ly - cy) ** 2 <= r * r:
                    name = self._name_for(sample_index)
                    if kind == "mix":
                        text = f"Mixing {name}"
                    elif kind == "spot":
                        text = f"Depositing {name}_droplet_{droplet}"
                    else:
                        text = name
                        if droplet is not None:
                            text += f"_droplet_{droplet}"
                    break
        if text:
            QToolTip.showText(event.globalPosition().toPoint(), text, self)
        else:
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    # --- painting ---
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(COLORS['bg_light']))
        self._hot = []

        if not self.has_data():
            p.setPen(QColor(COLORS['text_secondary']))
            p.setFont(QFont("Segoe UI", 10))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._empty_text)
            p.end()
            return

        cw, ch = self.content_size()
        if cw <= 0 or ch <= 0:
            p.end()
            return
        margin = 10
        avail_w = max(1, self.width() - 2 * margin)
        avail_h = max(1, self.height() - 2 * margin)
        scale = min(avail_w / cw, avail_h / ch)
        self._scale = scale
        self._tx = (self.width() - cw * scale) / 2.0
        self._ty = (self.height() - ch * scale) / 2.0
        p.translate(self._tx, self._ty)
        p.scale(scale, scale)
        self.draw_content(p)
        p.end()

    # --- shared drawing helper ---
    def _circle(self, p, cx, cy, r, fill, outline, width=1.0,
                number=None, text_color=None, num_pt=7.0):
        if fill is None:
            p.setBrush(Qt.BrushStyle.NoBrush)
        else:
            p.setBrush(QBrush(QColor(fill)))
        p.setPen(QPen(QColor(outline), width))
        p.drawEllipse(QPointF(cx, cy), r, r)
        if number is not None:
            f = QFont("Segoe UI", int(num_pt))
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor(text_color or COLORS['text_primary']))
            p.drawText(QRectF(cx - r, cy - r, 2 * r, 2 * r),
                       Qt.AlignmentFlag.AlignCenter, str(number))


class DestinationMap(ScalableMap):
    """384-well destination plate(s); spots colored by sample with index labels."""

    CELL = 22.0
    PAD_L = 26.0   # room for row letters
    PAD_T = 32.0   # room for the "Slot N" label + the column numbers
    PLATE_GAP = 34.0
    FOOTER = 22.0  # room below the plate for the total-droplet count

    def __init__(self, parent=None):
        super().__init__("Enable Spot or Combined to preview destinations", parent)
        self._status = {}
        self._num_plates = 0

    def set_data(self, status, num_plates):
        self._status = status
        self._num_plates = num_plates
        self.update()

    def has_data(self):
        return self._num_plates > 0

    def content_size(self):
        w = self.PAD_L + 24 * self.CELL + 8
        plate_h = self.PAD_T + 16 * self.CELL
        plates = self._num_plates * plate_h + max(0, self._num_plates - 1) * self.PLATE_GAP
        h = plates + self.FOOTER + 4
        return (w, h)

    def draw_content(self, p):
        CELL = self.CELL
        rad = CELL * 0.40
        label_color = QColor(COLORS['text_secondary'])
        label_font = QFont("Segoe UI", 8); label_font.setBold(True)
        title_font = QFont("Segoe UI", 10); title_font.setBold(True)
        plate_h = self.PAD_T + 16 * CELL

        for pl in range(self._num_plates):
            oy = pl * (plate_h + self.PLATE_GAP)

            # Slot label above the plate (destination is always slot 3)
            p.setFont(title_font)
            p.setPen(QColor(COLORS['text_primary']))
            p.drawText(QRectF(0, oy, self.PAD_L + 24 * CELL, 15),
                       Qt.AlignmentFlag.AlignCenter, "Slot 3")

            grid_top = oy + self.PAD_T

            # Column numbers (1-24) and row letters (A-P)
            p.setFont(label_font)
            p.setPen(label_color)
            for c in range(24):
                cx = self.PAD_L + c * CELL
                p.drawText(QRectF(cx, grid_top - 14, CELL, 12),
                           Qt.AlignmentFlag.AlignCenter, str(c + 1))
            for r in range(16):
                cy = grid_top + r * CELL
                p.drawText(QRectF(0, cy, self.PAD_L - 5, CELL),
                           Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                           chr(65 + r))

            # Large slide numbers behind the circles (Raman-panel style).
            slide_font = QFont("Segoe UI", int(CELL * 1.8))
            slide_font.setBold(True)
            p.setFont(slide_font)
            p.setPen(QColor('#5c5c5c'))
            for slide_num, (rc, cc) in SLIDE_CENTERS.items():
                cx = self.PAD_L + cc * CELL + CELL / 2
                cy = grid_top + rc * CELL + CELL / 2
                p.drawText(QRectF(cx - 2 * CELL, cy - 2 * CELL, 4 * CELL, 4 * CELL),
                           Qt.AlignmentFlag.AlignCenter, str(slide_num))

            # Wells
            for r in range(16):
                for c in range(24):
                    name = f"{chr(65 + r)}{c + 1}"
                    cx = self.PAD_L + c * CELL + CELL / 2
                    cy = grid_top + r * CELL + CELL / 2
                    key = (pl, name)
                    if key in self._status:
                        s_idx, hexc, done, drop = self._status[key]
                        if s_idx is None:
                            # Already deposited before the resume point.
                            self._circle(p, cx, cy, rad, '#444', '#666', 1.0)
                        else:
                            self._circle(p, cx, cy, rad, hexc, 'white', 1.0,
                                         number=s_idx + 1,
                                         text_color=_contrast_text(hexc))
                            self._register(cx, cy, rad, s_idx, drop)
                    elif name in VALID_WELLS_SET:
                        # Available position with no sample — grey empty circle.
                        self._circle(p, cx, cy, rad, COLORS['bg_elevated'],
                                     COLORS['border_light'], 1.0)
                    # Positions not available for deposition draw no circle.

        # Total deposited droplets, at the bottom-right corner.
        total = sum(1 for v in self._status.values() if v[0] is not None)
        plates_bottom = (self._num_plates * plate_h
                         + max(0, self._num_plates - 1) * self.PLATE_GAP)
        footer_font = QFont("Segoe UI", 9); footer_font.setBold(True)
        p.setFont(footer_font)
        p.setPen(QColor(COLORS['text_primary']))
        p.drawText(QRectF(0, plates_bottom + 2, self.PAD_L + 24 * CELL, self.FOOTER),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"Total droplets: {total}")


class DeckMap(ScalableMap):
    """Source tube racks on the deck (24-tube racks in slots 1/4/7/10)."""

    TUBE = 36.0
    RACK_PAD = 22.0   # inner margin; also holds the A-D / 1-6 coordinate labels
    LABEL_W = 0.0
    RACK_GAP = 22.0
    COL_GAP = 34.0
    HEADER = 20.0   # space above each rack for its "Slot N" label

    def __init__(self, parent=None):
        super().__init__("Set Total Samples to preview the deck", parent)
        self._samples = 0
        self._start_s = 1
        self._palette = []

    def set_data(self, samples, start_s, palette):
        self._samples = samples
        self._start_s = start_s
        self._palette = palette
        self.update()

    def has_data(self):
        return self._samples > 0

    def _active_slots(self):
        req = math.ceil(self._samples / 24) if self._samples else 0
        return [1, 4, 7, 10][:min(req, 4)]

    def _rack_size(self):
        return (6 * self.TUBE + 2 * self.RACK_PAD, 4 * self.TUBE + 2 * self.RACK_PAD)

    def content_size(self):
        n = max(1, len(self._active_slots()))
        cols, rows = grid_dims(n)
        rack_w, rack_h = self._rack_size()
        cell_w = self.LABEL_W + rack_w
        cell_h = rack_h + self.HEADER
        w = cols * cell_w + (cols - 1) * self.COL_GAP + 8
        h = rows * cell_h + (rows - 1) * self.RACK_GAP + 8
        return (w, h)

    def draw_content(self, p):
        TUBE = self.TUBE
        rad = TUBE * 0.34
        active = self._active_slots()   # ascending, matches the frame-title slot list
        n = len(active)
        cols, _ = grid_dims(n)
        rack_w, rack_h = self._rack_size()
        cell_w = self.LABEL_W + rack_w
        cell_h = rack_h + self.HEADER
        slot_to_rack = {1: 0, 4: 1, 7: 2, 10: 3}
        title_font = QFont("Segoe UI", 10); title_font.setBold(True)
        coord_font = QFont("Segoe UI", 8); coord_font.setBold(True)

        for vis_idx, slot in enumerate(active):
            gc = vis_idx % cols
            gr = vis_idx // cols
            cell_x = gc * (cell_w + self.COL_GAP)
            cell_y = gr * (cell_h + self.RACK_GAP)
            ox = cell_x
            oy = cell_y + self.HEADER

            # Slot label above the rack
            p.setFont(title_font)
            p.setPen(QColor(COLORS['text_primary']))
            p.drawText(QRectF(cell_x, cell_y, rack_w, self.HEADER),
                       Qt.AlignmentFlag.AlignCenter, f"Slot {slot}")

            # Rack body
            p.setBrush(QBrush(QColor(COLORS['bg_medium'])))
            p.setPen(QPen(QColor(COLORS['border_light']), 1.5))
            p.drawRoundedRect(QRectF(ox, oy, rack_w, rack_h), 6, 6)

            # Coordinate labels in the rack margin: columns 1-6 across the top,
            # rows A-D down the left.
            p.setFont(coord_font)
            p.setPen(QColor(COLORS['text_secondary']))
            for col in range(6):
                lx = ox + self.RACK_PAD + col * TUBE + TUBE / 2
                p.drawText(QRectF(lx - TUBE / 2, oy + 2, TUBE, self.RACK_PAD - 4),
                           Qt.AlignmentFlag.AlignCenter, str(col + 1))
            for row in range(4):
                ly = oy + self.RACK_PAD + row * TUBE + TUBE / 2
                p.drawText(QRectF(ox + 2, ly - TUBE / 2, self.RACK_PAD - 5, TUBE),
                           Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                           chr(65 + row))

            rack_index = slot_to_rack.get(slot, 0)
            for t_idx in range(24):
                gidx = rack_index * 24 + t_idx
                if gidx >= self._samples:
                    break
                row = t_idx % 4
                col = t_idx // 4
                cx = ox + self.RACK_PAD + col * TUBE + TUBE / 2
                cy = oy + self.RACK_PAD + row * TUBE + TUBE / 2
                skipped = (gidx + 1) < self._start_s
                if skipped:
                    self._circle(p, cx, cy, rad, '#555', '#777', 1.0,
                                 number=gidx + 1, text_color='white', num_pt=8)
                else:
                    color = self._palette[gidx] if gidx < len(self._palette) else '#888'
                    self._circle(p, cx, cy, rad, color, 'white', 1.0,
                                 number=gidx + 1, text_color='black', num_pt=8)
                self._register(cx, cy, rad, gidx)


class TipRackMap(ScalableMap):
    """96-tip racks; tips consumed for the run are highlighted, with the
    starting tip ringed. Tips are picked column-major (A1,B1..H1,A2..)."""

    CELL = 18.0
    PAD_L = 22.0
    PAD_T = 32.0   # room for the "Slot N" label + the column numbers
    LABEL_W = 16.0
    RACK_GAP = 30.0
    COL_GAP = 30.0

    def __init__(self, parent=None):
        super().__init__("Select an action to preview tip usage", parent)
        # racks: list of {slot, colors:{well->hex}, done:set, start_well:int|None}
        self._racks = []

    def set_data(self, racks):
        self._racks = racks
        self.update()

    def has_data(self):
        return bool(self._racks)

    def _rack_size(self):
        return (self.PAD_L + 12 * self.CELL + 6, self.PAD_T + 8 * self.CELL)

    def content_size(self):
        n = max(1, len(self._racks))
        cols, rows = grid_dims(n)
        rack_w, rack_h = self._rack_size()
        w = cols * rack_w + (cols - 1) * self.COL_GAP + 8
        h = rows * rack_h + (rows - 1) * self.RACK_GAP + 8
        return (w, h)

    def draw_content(self, p):
        CELL = self.CELL
        rad = CELL * 0.38
        label_color = QColor(COLORS['text_secondary'])
        label_font = QFont("Segoe UI", 8); label_font.setBold(True)
        title_font = QFont("Segoe UI", 10); title_font.setBold(True)
        cols, _ = grid_dims(len(self._racks))
        rack_w, rack_h = self._rack_size()

        for ri, rack in enumerate(self._racks):
            gc = ri % cols
            gr = ri // cols
            ox = gc * (rack_w + self.COL_GAP)
            oy = gr * (rack_h + self.RACK_GAP)
            colors = rack['colors']
            numbers = rack.get('numbers', {})
            info = rack.get('info', {})
            done = rack['done']

            # Slot label above the rack
            p.setFont(title_font)
            p.setPen(QColor(COLORS['text_primary']))
            p.drawText(QRectF(ox, oy, self.PAD_L + 12 * CELL, 15),
                       Qt.AlignmentFlag.AlignCenter, f"Slot {rack['slot']}")

            grid_top = oy + self.PAD_T

            # Column numbers 1-12, row letters A-H
            p.setFont(label_font)
            p.setPen(label_color)
            for c in range(12):
                cx = ox + self.PAD_L + c * CELL
                p.drawText(QRectF(cx, grid_top - 14, CELL, 12),
                           Qt.AlignmentFlag.AlignCenter, str(c + 1))
            for r in range(8):
                cy = grid_top + r * CELL
                p.drawText(QRectF(ox, cy, self.PAD_L - 5, CELL),
                           Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                           chr(65 + r))

            for c in range(12):
                for r in range(8):
                    idx = c * 8 + r
                    cx = ox + self.PAD_L + c * CELL + CELL / 2
                    cy = grid_top + r * CELL + CELL / 2
                    if idx in colors:
                        self._circle(p, cx, cy, rad, colors[idx], 'white', 1.0,
                                     number=numbers.get(idx),
                                     text_color=_contrast_text(colors[idx]),
                                     num_pt=6)
                        if idx in info:
                            s_i, kind, drop = info[idx]
                            self._register(cx, cy, rad, s_i, drop, kind)
                    elif idx in done:
                        # tips already used before the resume point
                        self._circle(p, cx, cy, rad, '#333', COLORS['border_dark'], 1.0)
                    else:
                        self._circle(p, cx, cy, rad, COLORS['bg_elevated'],
                                     COLORS['border_light'], 1.0)


# =============================================================================
# DIALOGS
# =============================================================================
# =============================================================================
# MAIN WINDOW
# =============================================================================
class SpottingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Opentrons Spotting Manager")
        self.setMinimumSize(1500, 950)

        self.last_path = ""
        self.sample_entries = []   # QLineEdit per sample
        self.sample_rows = []      # container widget per sample
        self._prefix_labels = []   # the "<prefix>_" label shown on each row
        self.sorted_wells_cache = list(DEPOSITION_ORDER)
        # Default protocol base name offered in the save dialog (updated when a
        # template is loaded).
        self._default_protocol_name = f"{datetime.datetime.now().strftime('%Y%m%d')}_OT2"

        # Capacity validation: last accepted values + a re-entrancy guard used
        # when reverting an out-of-capacity edit.
        self._prev_samples = 16
        self._prev_droplets = 1
        self._validating = False

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # Load-template button lives in a top bar above all columns.
        outer.addLayout(self._build_top_bar())

        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self._build_controls_panel())
        row.addWidget(self._build_samples_panel())
        row.addWidget(self._build_maps_panel(), 1)
        outer.addLayout(row, 1)

        # Actions live in a bottom bar (bottom-right corner).
        outer.addLayout(self._build_bottom_bar())

        self._rebuild_sample_entries()
        self._refresh_visuals()

    # ---- small builders -----------------------------------------------------
    def _section(self, title):
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 8)
        lay.setSpacing(6)
        return box, lay

    def _row(self, layout, label, widget):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        row.addWidget(lbl)
        row.addStretch()
        widget.setFixedWidth(72)   # compact inputs -> narrower controls column
        row.addWidget(widget)
        layout.addLayout(row)

    def _spin(self, lo, hi, val, step=1):
        s = NoWheelSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSingleStep(step)
        return s

    def _dspin(self, lo, hi, val, step=0.5, decimals=1):
        s = NoWheelDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setValue(val)
        s.setSingleStep(step)
        return s

    # ---- controls panel -----------------------------------------------------
    def _build_controls_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(286)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content.setObjectName("scrollContent")
        v = QVBoxLayout(content)
        v.setContentsMargins(2, 2, 8, 2)
        v.setSpacing(8)

        # Sample: total samples + droplets/sample
        box, lay = self._section("Sample")
        self.num_samples = self._spin(1, MAX_SAMPLES, 16)
        self.num_samples.setKeyboardTracking(False)
        self.num_samples.setToolTip(
            f"Max {MAX_SAMPLES} (4 tube racks x 24 tubes). "
            f"Samples x droplets must stay <= {PLATE_CAPACITY}.")
        self.num_samples.valueChanged.connect(self._on_samples_changed)
        self.droplets_per_sample = self._spin(1, PLATE_CAPACITY, 1)
        self.droplets_per_sample.setKeyboardTracking(False)
        self.droplets_per_sample.setToolTip(
            f"Samples x droplets must stay <= {PLATE_CAPACITY} (one plate)")
        self.droplets_per_sample.valueChanged.connect(self._on_droplets_changed)
        self._row(lay, "Total Samples:", self.num_samples)
        self._row(lay, "Droplets/Sample:", self.droplets_per_sample)
        v.addWidget(box)

        # Mixing toggle — sits directly above the mixing parameter panel.
        # Spotting is always performed; mixing is the optional step.
        self.do_mix = QCheckBox("Enable Mixing")
        self.do_mix.setChecked(True)
        self.do_mix.toggled.connect(self._on_mix_toggled)
        v.addWidget(self.do_mix)

        # Mixing parameters (enabled only when "Enable Mixing" is checked)
        self._mix_group, lay = self._section("Mixing Settings")
        self.mix_volume = self._dspin(1.0, 20.0, 5.0, step=1.0)
        self.mix_cycles = self._spin(1, 99, 10)
        self.mix_rate = self._dspin(0.1, 25.0, 5.0, step=0.5)
        self._row(lay, "Mix Vol (1-20 µL):", self.mix_volume)
        self._row(lay, "Mix Cycles:", self.mix_cycles)
        self._row(lay, "Mix Rate (µL/s):", self.mix_rate)
        mix_note = QLabel("Note: using separate pipette tips")
        mix_note.setProperty("secondary", True)
        mix_note.setWordWrap(True)
        lay.addWidget(mix_note)
        v.addWidget(self._mix_group)
        self._mix_group.setVisible(self.do_mix.isChecked())

        # Deposition Setting: dispense volume + aspiration/dispense rates
        box, lay = self._section("Deposition Settings")
        self.droplet_volume = self._dspin(0.0, 10.0, 2.0, step=0.1, decimals=2)
        self.asp_rate = self._dspin(0.1, 25.0, 5.0, step=0.5)
        self.disp_rate = self._dspin(0.1, 25.0, 5.0, step=0.5)
        self.travel_speed = self._spin(10, 500, 400)
        self.travel_speed.setToolTip("Gantry travel speed between labware "
                                     "(Opentrons default is 400 mm/s)")
        self._row(lay, "Dispense Vol (µL):", self.droplet_volume)
        self._row(lay, "Asp Rate (µL/s):", self.asp_rate)
        self._row(lay, "Disp Rate (µL/s):", self.disp_rate)
        self._row(lay, "Travel Speed (mm/s):", self.travel_speed)
        note = QLabel("Use slower disp rates for more viscous samples")
        note.setProperty("secondary", True)
        note.setWordWrap(True)
        lay.addWidget(note)
        v.addWidget(box)

        # Blowout toggle — folds the parameter panel like mixing.
        self.do_blowout = QCheckBox("Enable Blowout")
        self.do_blowout.setChecked(False)
        self.do_blowout.toggled.connect(self._on_blowout_toggled)
        v.addWidget(self.do_blowout)

        # Blowout parameters (enabled only when "Enable Blowout" is checked)
        self._blowout_group, lay = self._section("Blowout Settings")
        warn = QLabel("Warning: can introduce bubbles to the sample!")
        warn.setProperty("warn", True)
        warn.setWordWrap(True)
        lay.addWidget(warn)
        self.blowout_height = self._dspin(0.1, 10.0, 1.0, step=0.5)
        self.blowout_rate = self._dspin(0.1, 25.0, 5.0, step=0.5)
        self._row(lay, "Blowout Ht (mm):", self.blowout_height)
        self._row(lay, "Blowout Rate:", self.blowout_rate)
        v.addWidget(self._blowout_group)
        self._blowout_group.setVisible(self.do_blowout.isChecked())

        # Crash recovery — "Skip to Sample #" auto-fills the tip/deposition
        # resume points, which the user can still override.
        box, lay = self._section("Starting Positions")
        self.skip_sample = self._spin(1, MAX_SAMPLES, 1)
        self.skip_sample.setKeyboardTracking(False)
        self.skip_sample.valueChanged.connect(self._on_skip_sample_changed)
        # Tip / deposition resume points are given as well locations (e.g. A1,
        # B3). The tip well is within the first tip rack.
        self.skip_tip = QLineEdit("A1")
        self.skip_tip.setToolTip("Tip-rack well to resume from, e.g. A1")
        self.skip_tip.editingFinished.connect(self._refresh_visuals)
        self.skip_deposition = QLineEdit(self.sorted_wells_cache[0])
        self.skip_deposition.setToolTip("Destination well to resume from, e.g. B3")
        self.skip_deposition.editingFinished.connect(self._refresh_visuals)
        self._row(lay, "Skip to Sample #:", self.skip_sample)
        self._row(lay, "Skip to Pipette Tip:", self.skip_tip)
        self._row(lay, "Skip to Deposition:", self.skip_deposition)
        v.addWidget(box)

        v.addStretch()
        scroll.setWidget(content)
        return scroll

    # ---- top bar ------------------------------------------------------------
    def _build_top_bar(self):
        bar = QHBoxLayout()
        load_btn = QPushButton("Load Template Protocol…")
        load_btn.setToolTip("Load a previously generated protocol (.py) and "
                            "use its settings as a starting point")
        load_btn.clicked.connect(self._load_template)
        bar.addWidget(load_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Reset all parameters back to their defaults")
        clear_btn.clicked.connect(self._clear_template)
        bar.addWidget(clear_btn)

        self._template_label = QLabel("")
        self._template_label.setProperty("secondary", True)
        bar.addWidget(self._template_label)
        bar.addStretch()
        return bar

    # ---- bottom bar ---------------------------------------------------------
    def _build_bottom_bar(self):
        bar = QHBoxLayout()
        bar.addStretch()

        gen_btn = QPushButton("Generate Protocols")
        gen_btn.setProperty("primary", True)
        gen_btn.setMinimumHeight(36)
        gen_btn.clicked.connect(self.check_and_generate)
        bar.addWidget(gen_btn)

        if REPORTLAB_AVAILABLE:
            pdf_btn = QPushButton("Export PDF Report")
            pdf_btn.setProperty("secondaryAction", True)
            pdf_btn.setMinimumHeight(36)
            pdf_btn.clicked.connect(self.export_pdf)
            bar.addWidget(pdf_btn)

        return bar

    # ---- load template ------------------------------------------------------
    def _load_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Template Protocol", "", "Python protocol (*.py)")
        if not path:
            return
        try:
            with open(path, "r") as f:
                text = f.read()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read file:\n{e}")
            return

        # Pull module-level constant assignments out of the generated protocol.
        consts = {}
        try:
            for node in ast.parse(text).body:
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)):
                    try:
                        consts[node.targets[0].id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Not a valid protocol file:\n{e}")
            return

        if "NUM_SAMPLES" not in consts:
            QMessageBox.warning(self, "Load Template",
                                "This file doesn't look like a generated spotting protocol.")
            return

        def val(name, default, cast):
            try:
                return cast(consts[name])
            except Exception:
                return default

        n = max(1, min(MAX_SAMPLES, val("NUM_SAMPLES", 1, int)))
        d = max(1, min(PLATE_CAPACITY, val("DROPLETS", 1, int)))
        names = consts.get("SAMPLE_NAMES", [])

        # Samples + droplets (block signals so capacity checks / auto-fill don't
        # fire mid-load); rebuild the name fields manually.
        self.num_samples.blockSignals(True)
        self.num_samples.setValue(n)
        self.num_samples.blockSignals(False)
        self.droplets_per_sample.blockSignals(True)
        self.droplets_per_sample.setValue(d)
        self.droplets_per_sample.blockSignals(False)
        self._prev_samples, self._prev_droplets = n, d
        self._rebuild_sample_entries()
        # Split a shared "<prefix>_" off the loaded names back into the prefix
        # box, leaving just the per-sample suffix in each entry.
        prefix, suffixes = self._split_prefix([str(x) for x in names])
        self._set_text_silently(self._prefix_edit, prefix)
        for lbl in self._prefix_labels:
            lbl.setText(self._prefix_label_text())
        for i, sfx in enumerate(suffixes):
            if i < len(self.sample_entries):
                self.sample_entries[i].setText(sfx)

        self.droplet_volume.setValue(val("VOL", 2.0, float))
        self.mix_volume.setValue(val("MIX_VOL", 5.0, float))
        self.mix_cycles.setValue(val("MIX_CYCLES", 10, int))
        self.mix_rate.setValue(val("MIX_RATE", 5.0, float))
        self.travel_speed.setValue(val("TRAVEL_SPEED", 400, int))
        self.do_mix.setChecked(bool(consts.get("ENABLE_MIX", True)))
        self.do_blowout.setChecked(bool(consts.get("ENABLE_BLOWOUT", False)))
        self.blowout_height.setValue(val("BLOWOUT_HEIGHT", 1.0, float))
        self.blowout_rate.setValue(val("BLOWOUT_RATE", 5.0, float))

        # Aspiration / dispense rates: new templates embed ASP_RATE / DISP_RATE;
        # older ones only set the flow_rate lines, so fall back to those.
        if "ASP_RATE" in consts:
            self.asp_rate.setValue(val("ASP_RATE", 5.0, float))
            self.disp_rate.setValue(val("DISP_RATE", 5.0, float))
        else:
            m = re.search(r"flow_rate\.aspirate\s*=\s*([\d.]+)", text)
            if m:
                self.asp_rate.setValue(float(m.group(1)))
            m = re.search(r"flow_rate\.dispense\s*=\s*([\d.]+)", text)
            if m:
                self.disp_rate.setValue(float(m.group(1)))

        # Crash-recovery resume points (block skip_sample so it doesn't
        # auto-overwrite the loaded well locations).
        self.skip_sample.blockSignals(True)
        self.skip_sample.setValue(max(1, min(MAX_SAMPLES, val("START_SAMPLE_INDEX", 1, int))))
        self.skip_sample.blockSignals(False)
        if consts.get("START_DEPOSITION_WELL"):
            self._set_text_silently(self.skip_deposition, str(consts["START_DEPOSITION_WELL"]))
        if consts.get("START_TIP_WELL"):
            self._set_text_silently(self.skip_tip, str(consts["START_TIP_WELL"]))

        # Default save name = template base name (minus mode suffix + .py).
        base = os.path.basename(path)
        if base.endswith(".py"):
            base = base[:-3]
        for suf in ("_Combined", "_Spot", "_Mix"):
            if base.endswith(suf):
                base = base[:-len(suf)]
                break
        self._default_protocol_name = base
        self._template_label.setText(f"Template: {os.path.basename(path)}")

        self._refresh_visuals()

    def _clear_template(self):
        """Reset every parameter back to its default value."""
        self._set_text_silently(self._prefix_edit, "patient")
        # Samples + droplets (block signals so capacity checks don't fire).
        self.num_samples.blockSignals(True)
        self.num_samples.setValue(16)
        self.num_samples.blockSignals(False)
        self.droplets_per_sample.blockSignals(True)
        self.droplets_per_sample.setValue(1)
        self.droplets_per_sample.blockSignals(False)
        self._prev_samples, self._prev_droplets = 16, 1
        self._rebuild_sample_entries()
        for e in self.sample_entries:
            e.blockSignals(True)
            e.clear()
            e.blockSignals(False)
        for lbl in self._prefix_labels:
            lbl.setText(self._prefix_label_text())

        self.droplet_volume.setValue(2.0)
        self.mix_volume.setValue(5.0)
        self.mix_cycles.setValue(10)
        self.mix_rate.setValue(5.0)
        self.travel_speed.setValue(400)
        self.do_mix.setChecked(True)
        self.do_blowout.setChecked(False)
        self.blowout_height.setValue(1.0)
        self.blowout_rate.setValue(5.0)
        self.asp_rate.setValue(5.0)
        self.disp_rate.setValue(5.0)

        self.skip_sample.blockSignals(True)
        self.skip_sample.setValue(1)
        self.skip_sample.blockSignals(False)
        self._set_text_silently(self.skip_deposition, self.sorted_wells_cache[0])
        self._set_text_silently(self.skip_tip, "A1")

        self._default_protocol_name = f"{datetime.datetime.now().strftime('%Y%m%d')}_OT2"
        self._template_label.setText("")

        self._update_map_names()
        self._refresh_visuals()

    # ---- samples panel ------------------------------------------------------
    def _build_samples_panel(self):
        box = QGroupBox("Sample Names")
        box.setFixedWidth(340)
        outer = QVBoxLayout(box)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(6)

        # Editable name prefix applied to every sample (default "sample").
        prefix_row = QHBoxLayout()
        prefix_lbl = QLabel("Prefix:")
        prefix_lbl.setProperty("secondary", True)
        prefix_row.addWidget(prefix_lbl)
        self._prefix_edit = QLineEdit("patient")
        self._prefix_edit.textChanged.connect(self._on_prefix_changed)
        # Enter in the prefix box jumps to the first sample name.
        self._prefix_edit.returnPressed.connect(lambda: self._focus_sample_entry(0))
        prefix_row.addWidget(self._prefix_edit, 1)
        outer.addLayout(prefix_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._sample_container = QWidget()
        self._sample_container.setObjectName("scrollContent")
        self.sample_list_layout = QVBoxLayout(self._sample_container)
        self.sample_list_layout.setContentsMargins(0, 0, 4, 0)
        self.sample_list_layout.setSpacing(4)
        self.sample_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self._sample_container)
        outer.addWidget(scroll, 1)
        self._sample_scroll = scroll

        return box

    # ---- maps panel ---------------------------------------------------------
    def _build_maps_panel(self):
        panel = QWidget()
        grid = QGridLayout(panel)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)

        def wrap(title, widget):
            gb = QGroupBox(title)
            lay = QVBoxLayout(gb)
            lay.setContentsMargins(8, 10, 8, 8)
            lay.addWidget(widget)
            return gb

        self.dest_map = DestinationMap()
        self.deck_map = DeckMap()
        self.tip_map = TipRackMap()

        # Slot numbers live in the frame titles (set dynamically in
        # _refresh_visuals); the maps themselves no longer draw slot labels.
        self._tube_group = wrap("Sample Tube Rack", self.deck_map)
        self._tip_group = wrap("Pipette Tip Rack", self.tip_map)
        self._dest_group = wrap("Sample Deposition Map", self.dest_map)

        # Top row: tube racks + tip racks side by side. Bottom row: deposition
        # map spanning both columns (it is the largest, with 16x24 wells).
        grid.addWidget(self._tube_group, 0, 0)
        grid.addWidget(self._tip_group, 0, 1)
        grid.addWidget(self._dest_group, 1, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 2)
        return panel

    # ---- sample entry management -------------------------------------------
    def _on_samples_changed(self):
        if self._enforce_capacity("samples"):
            return
        self._prev_samples = self.num_samples.value()
        self._rebuild_sample_entries()
        self._refresh_visuals()

    def _on_droplets_changed(self):
        if self._enforce_capacity("droplets"):
            return
        self._prev_droplets = self.droplets_per_sample.value()
        self._refresh_visuals()

    def _enforce_capacity(self, which):
        """If samples x droplets would exceed one plate (192 spots), warn and
        revert the just-edited field to its last accepted value.

        Returns True if the edit was rejected.
        """
        if self._validating:
            return False
        s = self.num_samples.value()
        d = self.droplets_per_sample.value()
        if s * d <= PLATE_CAPACITY:
            return False

        self._validating = True
        try:
            if which == "samples":
                max_s = max(1, PLATE_CAPACITY // d)
                QMessageBox.warning(
                    self, "Capacity Exceeded",
                    f"{s} samples x {d} droplets/sample = {s * d} spots, which "
                    f"exceeds the {PLATE_CAPACITY}-spot plate capacity.\n\n"
                    f"With {d} droplets/sample, the maximum is {max_s} samples.")
                self.num_samples.setValue(self._prev_samples)
            else:
                max_d = max(1, PLATE_CAPACITY // s)
                QMessageBox.warning(
                    self, "Capacity Exceeded",
                    f"{s} samples x {d} droplets/sample = {s * d} spots, which "
                    f"exceeds the {PLATE_CAPACITY}-spot plate capacity.\n\n"
                    f"With {s} samples, the maximum is {max_d} droplets/sample.")
                self.droplets_per_sample.setValue(self._prev_droplets)
        finally:
            self._validating = False
        return True

    def _on_mix_toggled(self, checked):
        # Fold (hide) the mixing parameter panel when mixing is disabled.
        self._mix_group.setVisible(checked)
        self._refresh_visuals()

    def _on_blowout_toggled(self, checked):
        # Fold (hide) the blowout parameter panel when blowout is disabled.
        self._blowout_group.setVisible(checked)

    @staticmethod
    def _set_text_silently(line_edit, text):
        line_edit.blockSignals(True)
        line_edit.setText(text)
        line_edit.blockSignals(False)

    def _skip_tip_index(self):
        """First resume tip as a 0-based index within the first tip rack."""
        try:
            idx = tip_index(self.skip_tip.text())
            return idx if 0 <= idx <= 95 else 0
        except Exception:
            return 0

    def _skip_deposition_index(self):
        """First resume deposition spot as a 0-based index in the snake order."""
        well = self.skip_deposition.text().strip().upper()
        try:
            return self.sorted_wells_cache.index(well)
        except ValueError:
            return 0

    def _on_skip_sample_changed(self):
        """Auto-fill the tip/deposition resume locations from the sample number.

        Skipping to sample S means samples 1..S-1 are already done, which used
        (S-1) x droplets deposition spots and (S-1) x (droplets + mix) tips.
        The derived well locations can still be overridden by the user.
        """
        s = self.skip_sample.value()
        d = self.droplets_per_sample.value()
        per_sample_tips = d + (1 if self.do_mix.isChecked() else 0)
        dep_i = min(max(0, (s - 1) * d), len(self.sorted_wells_cache) - 1)
        tip_i = max(0, min(95, (s - 1) * per_sample_tips))
        self._set_text_silently(self.skip_deposition, self.sorted_wells_cache[dep_i])
        self._set_text_silently(self.skip_tip, tip_well(tip_i))
        self._refresh_visuals()

    def _rebuild_sample_entries(self):
        n = self.num_samples.value()
        cur = len(self.sample_entries)
        if n > cur:
            for i in range(cur, n):
                row = QWidget()
                h = QHBoxLayout(row)
                h.setContentsMargins(0, 0, 0, 0)
                h.setSpacing(6)
                lbl = QLabel(f"{i + 1}:")
                lbl.setStyleSheet(
                    f"color: {COLORS['text_secondary']}; font-size: 15px; font-weight: 600;")
                lbl.setFixedWidth(34)
                prefix = QLabel(self._prefix_label_text())
                prefix.setStyleSheet(f"color: {COLORS['text_secondary']};")
                e = QLineEdit()
                e.setPlaceholderText(f"{i + 1}")
                e.textChanged.connect(self._update_map_names)
                # Enter jumps to the next sample name box for fast typing.
                e.returnPressed.connect(lambda idx=i: self._focus_sample_entry(idx + 1))
                h.addWidget(lbl)
                h.addWidget(prefix)
                h.addWidget(e, 1)
                self.sample_list_layout.addWidget(row)
                self.sample_entries.append(e)
                self.sample_rows.append(row)
                self._prefix_labels.append(prefix)
        elif n < cur:
            for _ in range(cur - n):
                self.sample_entries.pop()
                self._prefix_labels.pop()
                row = self.sample_rows.pop()
                row.setParent(None)
                row.deleteLater()

    def _focus_sample_entry(self, index):
        """Move keyboard focus to sample entry `index` (if it exists), select
        its text, and scroll it into view."""
        if not (0 <= index < len(self.sample_entries)):
            return
        e = self.sample_entries[index]
        e.setFocus(Qt.FocusReason.TabFocusReason)
        e.selectAll()
        self._sample_scroll.ensureWidgetVisible(e)

    @staticmethod
    def _split_prefix(names):
        """Split a shared '<prefix>_' off a list of names. Returns
        (prefix, suffixes). If there is no common underscore-terminated prefix,
        the prefix is '' and the names are returned unchanged."""
        if not names:
            return "patient", []
        common = os.path.commonprefix(names)
        if "_" in common:
            cut = common.rfind("_")
            return common[:cut], [n[cut + 1:] for n in names]
        return "", list(names)

    def _prefix(self):
        return self._prefix_edit.text().strip()

    def _prefix_label_text(self):
        p = self._prefix()
        return f"{p}_" if p else ""

    def _full_name(self, suffix):
        p = self._prefix()
        return f"{p}_{suffix}" if p else f"{suffix}"

    def _on_prefix_changed(self):
        text = self._prefix_label_text()
        for lbl in self._prefix_labels:
            lbl.setText(text)
        self._update_map_names()

    def _sample_names(self):
        return [self._full_name(e.text() if e.text() else str(i + 1))
                for i, e in enumerate(self.sample_entries)]

    @staticmethod
    def _tube_position(i):
        """Tube-rack position for the 0-based sample index i, e.g. 'Slot_1_A1'.
        Racks fill column-major (A1, B1, C1, D1, A2, ...) across slots 1/4/7/10."""
        rack_idx, t = divmod(i, 24)
        row, col = t % 4, t // 4
        slot = [1, 4, 7, 10][rack_idx] if rack_idx < 4 else "?"
        return f"Slot_{slot}_{chr(ord('A') + row)}{col + 1}"

    def _update_map_names(self):
        """Push current sample names to the maps so hover tooltips stay current
        as the user edits the name fields (no repaint needed)."""
        names = self._sample_names()
        self.dest_map.set_names(names)
        self.deck_map.set_names(names)
        self.tip_map.set_names(names)

    # ---- visual refresh -----------------------------------------------------
    def _compute_tip_usage(self):
        """Per-rack tip state, colored to match the sample each tip serves.

        Tips before the "Skip to Pipette Tip" location are already used
        (dimmed); the resume tips (one MIX tip per processed sample if mixing
        is on, plus one per droplet to SPOT) are colored and numbered by the
        sample they serve; the rest are unused.
        """
        s = self.num_samples.value()
        d = self.droplets_per_sample.value()
        skip_sample = self.skip_sample.value()
        mix = self.do_mix.isChecked()
        palette = sample_palette(s)

        # Per resume tip, in pick-up order: (sample_index, kind, droplet).
        seq = []
        for i in range(s):
            if (i + 1) < skip_sample:
                continue
            if mix:
                seq.append((i, "mix", None))          # mixing tip
            for di in range(d):
                seq.append((i, "spot", di + 1))       # one spotting tip per droplet
        total = len(seq)

        start_global = self._skip_tip_index()   # 0-based first resume tip (rack 0)
        end_global = start_global + total        # exclusive

        slots = [2, 5, 8, 11]
        racks_needed = max(1, math.ceil(end_global / 96)) if end_global else 1
        racks_needed = min(racks_needed, len(slots))

        racks = []
        for r in range(racks_needed):
            lo = r * 96
            colors = {}
            numbers = {}
            info = {}
            done = set()
            for well_i in range(96):
                g = lo + well_i
                if start_global <= g < end_global:
                    sample_i, kind, drop = seq[g - start_global]
                    colors[well_i] = palette[sample_i]
                    numbers[well_i] = sample_i + 1      # 1-based sample number
                    info[well_i] = (sample_i, kind, drop)
                elif g < start_global:
                    done.add(well_i)
            start_well = (start_global - lo) if lo <= start_global < lo + 96 else None
            racks.append({'slot': slots[r], 'colors': colors, 'numbers': numbers,
                          'info': info, 'used': set(colors.keys()), 'done': done,
                          'start_well': start_well})
        return racks

    def _refresh_visuals(self):
        s = self.num_samples.value()
        d = self.droplets_per_sample.value()
        skip_sample = self.skip_sample.value()
        dep_start = self._skip_deposition_index()

        # Samples x droplets is capped to one plate (192 spots), so there is
        # always exactly one destination plate (slot 3) and no overflow.
        status = self.get_well_status(s, d, skip_sample, dep_start)
        num_plates = (max(pl for pl, _ in status.keys()) + 1) if status else 0
        self.dest_map.set_data(status, num_plates)
        self.deck_map.set_data(s, skip_sample, sample_palette(s))

        racks = self._compute_tip_usage()
        self.tip_map.set_data(racks)

        # Sample names power the hover tooltips on every map's circles.
        names = self._sample_names()
        self.dest_map.set_names(names)
        self.deck_map.set_names(names)
        self.tip_map.set_names(names)
        # Slot numbers are shown above each rack/plate in the maps, so the
        # frame titles stay as plain names (set once at construction).

    def get_well_status(self, samples, droplets, skip_sample, dep_start):
        """Destination spot status keyed by (plate, well).

        ``dep_start`` is the 0-based deposition spot to resume at. Value is
        (sample_index, color, done, droplet_number): spots before the resume
        point are marked done (sample_index = None); spots from the resume
        point on are colored per the sample that fills them, with the 1-based
        droplet number for that sample.
        """
        status = {}
        cap = len(self.sorted_wells_cache)
        palette = sample_palette(samples)
        cursor = max(0, dep_start)

        # Already-deposited spots before the resume point.
        for g in range(min(cursor, cap)):
            status[(0, self.sorted_wells_cache[g])] = (None, "#444", True, None)

        for s in range(samples):
            if (s + 1) < skip_sample:
                continue
            for di in range(droplets):
                if 0 <= cursor < cap:
                    status[(0, self.sorted_wells_cache[cursor])] = (s, palette[s], False, di + 1)
                cursor += 1
        return status

    # ---- generation ---------------------------------------------------------
    def check_and_generate(self):
        try:
            s = self.num_samples.value()
            d = self.droplets_per_sample.value()
            vol = self.droplet_volume.value()
            asp = self.asp_rate.value()
            disp = self.disp_rate.value()
            m_vol = self.mix_volume.value()
            b_ht = self.blowout_height.value()
            b_rt = self.blowout_rate.value()

            if not (0.2 <= vol <= 3.0): raise ValueError("Dispense Volume must be between 0.2 and 3.0 µL")
            if not (0.1 <= asp <= 25.0): raise ValueError("Physics error (Asp Rate)")
            if not (0.1 <= disp <= 25.0): raise ValueError("Physics error (Disp Rate)")
            if self.do_mix.isChecked() and not (1.0 <= m_vol <= 20.0):
                raise ValueError("Mix Volume must be between 1 and 20 µL")
            if not (0.1 <= b_ht <= 10.0): raise ValueError("Blowout Height error")
            if not (0.1 <= b_rt <= 25.0): raise ValueError("Blowout Rate error")

            # Mixing is the optional add-on: Combined = mix + spot, else spot only.
            mode = "Combined" if self.do_mix.isChecked() else "Spot"
            self.generate_files([mode])
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def generate_files(self, modes):
        names = self._sample_names()
        base_name = self._default_protocol_name or "Protocol"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Protocol", base_name, "Python (*.py)")
        if not path:
            return

        base_dir = os.path.dirname(path)
        chosen_filename = os.path.basename(path)
        if chosen_filename.endswith(".py"):
            chosen_filename = chosen_filename[:-3]

        # Use the user's chosen name exactly — no mode suffix appended.
        generated_files = []
        for mode in modes:
            full_path = os.path.join(base_dir, f"{chosen_filename}.py")
            code = self.get_protocol_code(names, mode, chosen_filename)
            with open(full_path, "w") as f:
                f.write(code)
            if mode in ["Spot", "Combined"]:
                self.generate_json_manifest(full_path, names, mode)
            generated_files.append(chosen_filename + ".py")
            self.last_path = full_path

        QMessageBox.information(self, "Success",
                                "Generated Protocols:\n" + "\n".join(generated_files))

    def generate_json_manifest(self, filepath, names, mode):
        s = self.num_samples.value()
        d = self.droplets_per_sample.value()
        skip_sample = self.skip_sample.value()
        manifest = []
        ordered_wells = list(DEPOSITION_ORDER)
        cursor = self._skip_deposition_index()

        droplet_index = 1   # global 1..N deposition sequence / unique id
        for i in range(s):
            if (i + 1) < skip_sample:
                continue

            sample_name = names[i]
            for drop_idx in range(d):
                well = ordered_wells[cursor] if cursor < len(ordered_wells) else None
                manifest.append({
                    "droplet_index": droplet_index,
                    "droplet_position": well,
                    "sample_tube_position": self._tube_position(i),
                    "sample_name": sample_name,
                    "droplet_number": drop_idx + 1,
                })
                droplet_index += 1
                cursor += 1

        json_path = filepath.replace(".py", ".json")
        with open(json_path, "w") as jf:
            json.dump(manifest, jf, indent=4)

    def get_protocol_code(self, sample_names, mode, final_protocol_name):
        s = self.num_samples.value()
        d = self.droplets_per_sample.value()
        v = self.droplet_volume.value()
        asp = self.asp_rate.value()
        disp = self.disp_rate.value()
        m_vol = self.mix_volume.value()
        m_cyc = self.mix_cycles.value()
        m_rate = self.mix_rate.value()
        travel_speed = self.travel_speed.value()
        skip_sample = self.skip_sample.value()
        skip_tip_well = self.skip_tip.text().strip().upper() or "A1"
        skip_dep_well = self.skip_deposition.text().strip().upper() or self.sorted_wells_cache[0]
        b_ht = self.blowout_height.value()
        b_rt = self.blowout_rate.value()

        do_mix = mode in ["Mix", "Combined"]
        do_spot = mode in ["Spot", "Combined"]
        do_blowout = self.do_blowout.isChecked()

        # One gist_rainbow color per sample, embedded into the protocol so the
        # robot's liquid display colors match the GUI maps.
        colors_py = str(sample_palette(s))

        tip_slots = [2, 5, 8, 11]

        return f"""
from opentrons import protocol_api
import math

metadata = {{'protocolName': '{final_protocol_name}', 'apiLevel': '2.14'}}

SAMPLE_NAMES = {sample_names}
NUM_SAMPLES = {s}
DROPLETS = {d}
VOL = {v}
VALID_WELLS = {VALID_WELLS}
ORDERED_WELLS = {DEPOSITION_ORDER}
COLORS = {colors_py}
START_SAMPLE_INDEX = {skip_sample}
START_DEPOSITION_WELL = '{skip_dep_well}'
START_TIP_WELL = '{skip_tip_well}'
ENABLE_MIX = {do_mix}
ENABLE_SPOT = {do_spot}
MIX_VOL = {m_vol}
MIX_CYCLES = {m_cyc}
MIX_RATE = {m_rate}
ASP_RATE = {asp}
DISP_RATE = {disp}
TRAVEL_SPEED = {travel_speed}   # gantry travel speed (mm/s)

# Blowout Configuration
ENABLE_BLOWOUT = {do_blowout}
BLOWOUT_HEIGHT = {b_ht}
BLOWOUT_RATE = {b_rt}

def run(protocol: protocol_api.ProtocolContext):
    plate1 = protocol.load_labware('spotting_384_wellplate_10ul', 3) if ENABLE_SPOT else None

    total_tips = 0
    if ENABLE_MIX: total_tips += NUM_SAMPLES
    if ENABLE_SPOT: total_tips += (NUM_SAMPLES * DROPLETS)

    tip_slots = {tip_slots}
    req_tip_racks = math.ceil(total_tips / 96)
    tips_to_load = []
    for i in range(min(req_tip_racks, len(tip_slots))):
        tips_to_load.append(protocol.load_labware('opentrons_96_tiprack_20ul', tip_slots[i]))

    p20 = protocol.load_instrument('p20_single_gen2', 'left', tip_racks=tips_to_load)
    p20.default_speed = TRAVEL_SPEED   # gantry travel speed between labware
    # Resume from a specific tip well in the first tip rack (e.g. 'A1').
    if tips_to_load and START_TIP_WELL:
        p20.starting_tip = tips_to_load[0].wells_by_name()[START_TIP_WELL]

    src_slots = [1, 4, 7, 10]
    req_src_racks = math.ceil(NUM_SAMPLES / 24)
    sources = [protocol.load_labware('opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap', src_slots[i]) for i in range(min(req_src_racks, 4))]

    for i in range(NUM_SAMPLES):
        rack_idx = i // 24
        tube_idx = i % 24
        well = sources[rack_idx].wells()[tube_idx]
        name = SAMPLE_NAMES[i]
        color = COLORS[i % len(COLORS)]
        liq = protocol.define_liquid(name=name, description=f"Sample {{i+1}}", display_color=color)
        well.load_liquid(liquid=liq, volume=500)

    p20.flow_rate.aspirate = ASP_RATE
    p20.flow_rate.dispense = DISP_RATE
    p20.flow_rate.blow_out = BLOWOUT_RATE

    ordered = ORDERED_WELLS
    # Resume at the given deposition well (its position in the fill order).
    cursor = ordered.index(START_DEPOSITION_WELL) if START_DEPOSITION_WELL in ordered else 0

    for i in range(NUM_SAMPLES):
        if (i + 1) < START_SAMPLE_INDEX:
            continue

        rack_idx = i // 24
        tube_idx = i % 24
        src = sources[rack_idx].wells()[tube_idx]

        protocol.comment(f"Processing {{SAMPLE_NAMES[i]}} (Sample {{i+1}})")

        if ENABLE_MIX:
            p20.flow_rate.aspirate = MIX_RATE
            p20.flow_rate.dispense = MIX_RATE
            p20.pick_up_tip()
            # Descend into the sample, then mix in place (no extra lift between
            # the descent and the mixing cycles).
            p20.move_to(src.bottom(2), speed=125)
            p20.mix(MIX_CYCLES, MIX_VOL)
            p20.drop_tip()
            p20.flow_rate.aspirate = ASP_RATE
            p20.flow_rate.dispense = DISP_RATE

        if ENABLE_SPOT:
            for _ in range(DROPLETS):
                p20.pick_up_tip()
                dest = plate1.wells_by_name()[ordered[cursor]]

                # Standard Aspiration
                p20.move_to(src.bottom(1), speed=125)
                protocol.delay(seconds=0)
                p20.aspirate(VOL, src.bottom(1))
                protocol.delay(seconds=1)
                p20.move_to(src.top(), speed=125)

                # Dispense
                p20.move_to(dest.bottom(0.2), speed=125)
                protocol.delay(seconds=0)
                p20.dispense(VOL, dest.bottom(0.2))

                if ENABLE_BLOWOUT:
                    protocol.delay(seconds=1)
                    p20.move_to(dest.bottom(BLOWOUT_HEIGHT), speed=10)
                    p20.blow_out()

                protocol.delay(seconds=1)
                p20.move_to(dest.bottom(3.0), speed=10)

                p20.drop_tip()
                cursor += 1
"""

    # ---- PDF export ---------------------------------------------------------
    def export_pdf(self):
        if not REPORTLAB_AVAILABLE:
            QMessageBox.critical(self, "Error", "ReportLab not installed")
            return
        try:
            s = self.num_samples.value()
            d = self.droplets_per_sample.value()
            vol = self.droplet_volume.value()
            asp = self.asp_rate.value()
            disp = self.disp_rate.value()
            b_ht = self.blowout_height.value()
            b_rt = self.blowout_rate.value()
            skip_sample = self.skip_sample.value()
            skip_tip_well = self.skip_tip.text().strip().upper() or "A1"
            skip_dep_well = self.skip_deposition.text().strip().upper() or self.sorted_wells_cache[0]
            dep_start = self._skip_deposition_index()
            base_name = self._default_protocol_name or "Protocol_Report"
            names = self._sample_names()
            palette = sample_palette(s)
        except Exception:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export PDF", base_name, "PDF (*.pdf)")
        if not filename:
            return
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"

        c = pdf_canvas.Canvas(filename, pagesize=landscape(letter))
        w, h = landscape(letter)

        well_status = self.get_well_status(s, d, skip_sample, dep_start)
        num_plates = max([p for p, _ in well_status.keys()]) + 1 if well_status else 0

        for p in range(num_plates):
            c.setFont("Helvetica-Bold", 18)
            title = f"Opentrons Report: {base_name}" if p == 0 else f"Overflow Destination Locations (Plate {p+1})"
            c.drawString(30, h - 40, title)

            c.setFont("Helvetica", 10)
            if p == 0:
                c.drawString(30, h - 60, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
                c.drawString(30, h - 75, f"Config: {s} Samples | {d} Drops | {vol} µL")
                blowout_str = f"ON (Ht: {b_ht} mm, Rt: {b_rt} µL/s)" if self.do_blowout.isChecked() else "OFF"
                c.drawString(30, h - 90, f"Physics: Asp {asp} | Disp {disp} | Blowout: {blowout_str}")
                c.drawString(30, h - 105, f"Recovery: Skip to Sample {skip_sample} | Tip {skip_tip_well} | Deposition {skip_dep_well}")

            cell_size = 20
            start_x = (w - (24 * cell_size)) / 2
            start_y = h - 150

            for r in range(16):
                row_char = chr(ord('A') + r)
                y_pos = start_y - (r * cell_size)
                for col in range(24):
                    col_num = col + 1
                    w_name = f"{row_char}{col_num}"
                    x_pos = start_x + (col * cell_size)

                    if (p, w_name) in well_status:
                        s_idx, hex_c, done, _drop = well_status[(p, w_name)]
                        c.setFillColor(pdf_colors.HexColor(hex_c))
                        c.circle(x_pos, y_pos, 7, stroke=1, fill=1)
                        if s_idx is not None:
                            c.setFillColor(pdf_colors.black)
                            c.setFont("Helvetica-Bold", 6)
                            c.drawCentredString(x_pos, y_pos - 2, str(s_idx + 1))
                    elif w_name in VALID_WELLS_SET:
                        c.setFillColor(pdf_colors.lightgrey)
                        c.circle(x_pos, y_pos, 7, stroke=1, fill=1)
                    else:
                        c.setFillColor(pdf_colors.whitesmoke)
                        c.circle(x_pos, y_pos, 2, stroke=0, fill=1)

            c.setFillColor(pdf_colors.black)
            c.setFont("Helvetica", 6)
            for col in range(24):
                c.drawCentredString(start_x + (col * cell_size), start_y + 15, str(col + 1))
            for r in range(16):
                c.drawString(start_x - 20, start_y - (r * cell_size) - 3, chr(ord('A') + r))
            c.showPage()

        c.setPageSize(portrait(letter))
        w, h = portrait(letter)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(30, h - 40, "Opentrons Spotting Report - Source Deck Layout")
        rack_capacity = 24
        req_racks = math.ceil(s / rack_capacity)
        active_slots = [1, 4, 7, 10][:min(req_racks, 4)]
        display_order = active_slots[::-1]
        rack_x = 100
        start_y = h - 100

        for visual_idx, slot in enumerate(display_order):
            visual_y = start_y - (visual_idx * 200)
            c.setStrokeColor(pdf_colors.black)
            c.rect(rack_x, visual_y - 150, 300, 150, fill=0)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(rack_x - 60, visual_y - 20, f"Slot {slot}")
            rack_index = {1: 0, 4: 1, 7: 2, 10: 3}.get(slot, 0)

            for t_idx in range(24):
                global_sample_idx = (rack_index * 24) + t_idx
                if global_sample_idx >= s:
                    break
                row = t_idx % 4
                col = t_idx // 4
                tx = rack_x + 30 + (col * 40)
                ty = visual_y - 30 - (row * 30)
                skipped = (global_sample_idx + 1) < skip_sample
                hex_c = palette[global_sample_idx] if not skipped else "#999999"
                c.setFillColor(pdf_colors.HexColor(hex_c))
                c.circle(tx, ty, 10, stroke=1, fill=1)
                c.setFillColor(pdf_colors.black if not skipped else pdf_colors.white)
                c.setFont("Helvetica-Bold", 8)
                c.drawCentredString(tx, ty - 3, str(global_sample_idx + 1))
        c.showPage()

        c.setPageSize(portrait(letter))
        w, h = portrait(letter)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(30, h - 40, "Sample Key")
        c.setFont("Helvetica", 10)
        key_start_x = 40
        key_start_y = h - 80
        col_width = 180
        line_height = 15
        max_y = 50
        current_x = key_start_x
        current_y = key_start_y

        for i, name in enumerate(names):
            col = pdf_colors.HexColor(palette[i] if i < len(palette) else "#888888")
            c.setFillColor(col)
            c.circle(current_x, current_y + 3, 5, fill=1)
            c.setFillColor(pdf_colors.black)
            c.drawString(current_x + 10, current_y, f"{i+1}: {name}")
            current_y -= line_height
            if current_y < max_y:
                current_y = key_start_y
                current_x += col_width
                if current_x + col_width > w:
                    c.showPage()
                    c.setFont("Helvetica-Bold", 18)
                    c.drawString(30, h - 40, "Sample Key (Continued)")
                    c.setFont("Helvetica", 10)
                    current_x = key_start_x
                    current_y = key_start_y

        c.save()
        QMessageBox.information(self, "Exported", f"Report saved to:\n{filename}")


def main():
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    win = SpottingApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
