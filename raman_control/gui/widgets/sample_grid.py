"""
384 well plate grid widget for selecting acquisition positions.

Displays only the valid sample wells, grouped into 12 quartz slide
clusters (3 row-groups x 4 column-groups, 16 spots per slide) with
visual spacing between slides. The plate is rotated 90 degrees CCW
so rows run left-to-right and columns run top-to-bottom.
"""

import json
from typing import Optional, List, Set, Tuple
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QGraphicsView, QGraphicsScene,
    QGraphicsEllipseItem, QFileDialog, QMessageBox, QToolTip,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QBrush, QPen, QColor, QFont, QMouseEvent, QTransform, QPolygonF
from PyQt6.QtCore import QPointF

from raman_control.gui.styles import COLORS
from raman_control.config.logging_config import get_logger

logger = get_logger('sample_grid')

# 384 well plate: 16 rows (A-P) x 24 columns (1-24)
ROWS = 16
COLS = 24
ROW_LABELS = [chr(ord('A') + i) for i in range(ROWS)]

# Valid wells for this specific plate layout
VALID_WELLS = {
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
    "P4", "P5", "P10", "P11", "P16", "P17", "P22", "P23",
}

# Slide clusters: 3 row-groups x 4 column-groups = 12 slides, 16 spots each
ROW_GROUPS = [
    [0, 1, 2, 3, 4],      # A-E
    [6, 7, 8, 9],          # G-J
    [11, 12, 13, 14, 15],  # L-P
]
COL_GROUPS = [
    [3, 4, 5, 6],
    [9, 10, 11, 12],
    [15, 16, 17, 18],
    [21, 22, 23, 24],
]

# Grid drawing constants
WELL_RADIUS = 10
WELL_SPACING = 26
SLIDE_GAP = 14
LABEL_OFFSET = 34


def well_name(row: int, col: int) -> str:
    """Convert row/col indices to well name (e.g., 0,0 -> 'A1')."""
    return f"{ROW_LABELS[row]}{col + 1}"


class WellCircle(QGraphicsEllipseItem):
    """Interactive circle representing a well position."""

    def __init__(self, row: int, col: int, x: float, y: float, radius: float):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.row = row
        self.col = col
        self.well = well_name(row, col)
        self.setPos(x, y)

        self._selected = False
        self._hovered = False
        self._disabled = False
        self.setAcceptHoverEvents(True)
        self._update_style()

    def _update_style(self):
        if self._disabled:
            c = QColor('#404040')
            c.setAlpha(160)
            self.setBrush(QBrush(c))
            self.setPen(QPen(QColor('#505050'), 1))
        elif self._selected:
            self.setBrush(QBrush(QColor(COLORS['warning'])))
            self.setPen(QPen(QColor('#ffb74d'), 1.5))
        elif self._hovered:
            self.setBrush(QBrush(QColor(COLORS['accent_hover'])))
            self.setPen(QPen(QColor(COLORS['accent_primary']), 1.5))
        else:
            c = QColor(COLORS['info'])
            c.setAlpha(180)
            self.setBrush(QBrush(c))
            self.setPen(QPen(QColor('#1976d2'), 1))

    def set_selected(self, selected: bool):
        self._selected = selected
        self._update_style()

    def set_disabled(self, disabled: bool):
        self._disabled = disabled
        self._update_style()

    def hoverEnterEvent(self, event):
        if not self._disabled:
            self._hovered = True
            self._update_style()
        owner = getattr(self, '_owner', None)
        if owner is not None:
            owner._on_circle_hover_enter(self, event.screenPos())
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self._update_style()
        owner = getattr(self, '_owner', None)
        if owner is not None:
            owner._on_circle_hover_leave()
        super().hoverLeaveEvent(event)


class SampleGridWidget(QWidget):
    """
    384 well plate grid for position selection.

    Only valid sample wells are drawn, grouped by quartz slide with
    gaps between clusters. Clicking a well highlights it and emits
    position_selected(row, col). The main window uses this to move
    the stage to that well.
    """

    position_selected = pyqtSignal(int, int)  # row, col (single-click mode)
    selection_changed = pyqtSignal(list)       # multi-select mode

    def __init__(self, multi_select: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._multi_select = multi_select
        self._grid_enabled = True
        self._circles: dict[Tuple[int, int], WellCircle] = {}
        self._current_position: Optional[Tuple[int, int]] = None
        self._selected_positions: Set[Tuple[int, int]] = set()
        self._initial_scroll_done = False
        # droplet_position (well name) -> manifest entry, from a loaded config
        self._droplet_map: dict = {}

        # Custom 0.5 s hover-tooltip (Qt's built-in delay isn't tunable here).
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(500)
        self._hover_timer.timeout.connect(self._show_hover_tooltip)
        self._hover_text = ""
        self._hover_pos = None

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        grid_group = QGroupBox("Sample Droplet Map")
        self._grid_group = grid_group
        grid_layout = QVBoxLayout(grid_group)
        grid_layout.setContentsMargins(4, 4, 4, 4)
        grid_layout.setSpacing(4)

        self._scene = QGraphicsScene()
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(self._view.renderHints().Antialiasing)
        self._view.setStyleSheet(f'''
            QGraphicsView {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['border_dark']};
                border-radius: 4px;
            }}
        ''')
        # Scrollbars appear when grid is larger than viewport
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._view.setMinimumHeight(160)

        self._create_grid()
        grid_layout.addWidget(self._view)

        # Controls / info row
        controls = QHBoxLayout()
        controls.setSpacing(6)

        if self._multi_select:
            select_all_btn = QPushButton("Select All")
            select_all_btn.setFixedHeight(24)
            select_all_btn.clicked.connect(self._on_select_all)
            controls.addWidget(select_all_btn)

            clear_btn = QPushButton("Clear")
            clear_btn.setFixedHeight(24)
            clear_btn.clicked.connect(self._on_clear_selection)
            controls.addWidget(clear_btn)

            controls.addStretch()

            self._count_label = QLabel("0 selected")
            self._count_label.setStyleSheet(f'color: {COLORS["accent_primary"]}; font-size: 11px;')
            controls.addWidget(self._count_label)
        else:
            self._info_label = QLabel("Click a well to move stage")
            self._info_label.setStyleSheet(f'color: {COLORS["text_secondary"]}; font-size: 11px;')
            controls.addWidget(self._info_label)
            controls.addStretch()

        # Load Config (far-right / bottom-right corner): loads an Opentrons
        # droplet manifest (.json) so hovering a well shows its sample name.
        self._load_config_btn = QPushButton("Load Config")
        self._load_config_btn.setFixedHeight(24)
        self._load_config_btn.clicked.connect(self._on_load_config)
        controls.addWidget(self._load_config_btn)

        grid_layout.addLayout(controls)
        main_layout.addWidget(grid_group)

    def _create_grid(self):
        """Create the well plate grid rotated 90 degrees CCW.

        Only valid wells are drawn. Slide clusters are separated by gaps.
        The grid is drawn at a fixed pixel size — scrollbars appear if the
        view is too small to show everything.

        Each cluster has a large slide number rendered behind the circles.
        Numbering follows a snake pattern starting from the bottom-left:
        row 0 (bottom) L->R: 1,2,3; row 1 R->L: 4,5,6; etc.
        """
        self._scene.clear()
        self._circles.clear()

        s = WELL_SPACING
        r = WELL_RADIUS
        gap = SLIDE_GAP

        label_font = QFont("Segoe UI", 8)
        label_font.setBold(True)
        label_color = QColor(COLORS['text_secondary'])

        # X positions: rows grouped with gaps
        row_x = {}
        x = LABEL_OFFSET
        for gi, rg in enumerate(ROW_GROUPS):
            if gi > 0:
                x += gap
            for row_idx in rg:
                row_x[row_idx] = x
                x += s

        # Y positions: column groups reversed (highest at top) with gaps
        col_y = {}
        y = 0
        for gi, cg in enumerate(reversed(COL_GROUPS)):
            if gi > 0:
                y += gap
            for col_num in reversed(cg):
                col_y[col_num] = y
                y += s

        # Bottom labels: row letters (below the last row of wells)
        max_y = max(col_y.values()) + s
        for row_idx in row_x:
            lx = row_x[row_idx]
            text = self._scene.addText(ROW_LABELS[row_idx], label_font)
            text.setDefaultTextColor(label_color)
            text.setPos(lx - text.boundingRect().width() / 2, max_y)

        # Left labels: column numbers
        for col_num, ly in col_y.items():
            text = self._scene.addText(str(col_num), label_font)
            text.setDefaultTextColor(label_color)
            text.setPos(-LABEL_OFFSET - text.boundingRect().width() / 2,
                        ly - text.boundingRect().height() / 2)

        # Slide number labels (behind circles)
        # Snake pattern: bottom row L->R, next row R->L, etc.
        # cgi=0 is COL_GROUPS[0] (cols 3-6, bottom of display)
        slide_font = QFont("Segoe UI", 48)
        slide_font.setBold(True)
        slide_color = QColor('#606060')

        slide_num = 1
        for cgi in range(len(COL_GROUPS)):
            rgi_order = (range(len(ROW_GROUPS)) if cgi % 2 == 0
                         else range(len(ROW_GROUPS) - 1, -1, -1))
            for rgi in rgi_order:
                rg = ROW_GROUPS[rgi]
                cg = COL_GROUPS[cgi]

                # Center of this cluster
                cx = sum(row_x[ri] for ri in rg if ri in row_x) / max(1, sum(1 for ri in rg if ri in row_x))
                cy = sum(col_y[cn] for cn in cg if cn in col_y) / max(1, sum(1 for cn in cg if cn in col_y))

                text = self._scene.addText(str(slide_num), slide_font)
                text.setDefaultTextColor(slide_color)
                text.setPos(cx - text.boundingRect().width() / 2,
                            cy - text.boundingRect().height() / 2)
                text.setZValue(-1)  # behind circles

                slide_num += 1

        # Valid well circles only
        for row_idx in row_x:
            for col_num in col_y:
                col_idx = col_num - 1
                name = well_name(row_idx, col_idx)
                if name not in VALID_WELLS:
                    continue

                circle = WellCircle(row_idx, col_idx, row_x[row_idx], col_y[col_num], r)
                circle._owner = self
                circle._tip_text = ""
                self._scene.addItem(circle)
                self._circles[(row_idx, col_idx)] = circle

        # Home fiducial: L-shaped mark at the bottom-left corner of slide 1
        # One arm points up, the other points right
        slide1_rg = ROW_GROUPS[0]
        slide1_cg = COL_GROUPS[0]
        corner_x = row_x[slide1_rg[0]] - r - 4  # left edge of leftmost well
        corner_y = col_y[slide1_cg[0]] + r + 4   # bottom edge of bottommost well
        arm_len = 20
        thickness = 4
        l_shape = QPolygonF([
            QPointF(corner_x, corner_y),                          # outer bottom-left
            QPointF(corner_x + arm_len, corner_y),                # right arm end (bottom)
            QPointF(corner_x + arm_len, corner_y - thickness),    # right arm end (top)
            QPointF(corner_x + thickness, corner_y - thickness),  # inner corner
            QPointF(corner_x + thickness, corner_y - arm_len),    # up arm end (right)
            QPointF(corner_x, corner_y - arm_len),                # up arm end (left)
        ])
        fiducial = self._scene.addPolygon(
            l_shape,
            QPen(Qt.PenStyle.NoPen),
            QBrush(QColor(COLORS['warning']))
        )
        fiducial.setZValue(2)  # above circles

        # Set scene rect with padding
        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-10, -10, 10, 10))

        # Initial fit — sets the baseline scale
        self._view.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        # Remember this as the minimum (1:1) scale — never go smaller
        self._base_transform = self._view.transform()

        # Mouse events
        self._view.mouseReleaseEvent = self._on_view_mouse_release

    def _on_view_mouse_release(self, event: QMouseEvent):
        """Handle click — ignore if grid is disabled."""
        if not self._grid_enabled:
            return

        scene_pos = self._view.mapToScene(event.pos())
        item = self._scene.itemAt(scene_pos, self._view.transform())

        if not isinstance(item, WellCircle):
            return

        pos = (item.row, item.col)

        if self._multi_select:
            if pos in self._selected_positions:
                self._selected_positions.discard(pos)
                item.set_selected(False)
            else:
                self._selected_positions.add(pos)
                item.set_selected(True)
            self._count_label.setText(f"{len(self._selected_positions)} selected")
            self.selection_changed.emit(list(self._selected_positions))
        else:
            if self._current_position and self._current_position in self._circles:
                self._circles[self._current_position].set_selected(False)
            item.set_selected(True)
            self._current_position = pos
            self._info_label.setText(f"Moving to {well_name(*pos)}")
            self.position_selected.emit(item.row, item.col)

    def _on_select_all(self):
        if not self._grid_enabled:
            return
        self._selected_positions.clear()
        for pos, circle in self._circles.items():
            circle.set_selected(True)
            self._selected_positions.add(pos)
        self._count_label.setText(f"{len(self._selected_positions)} selected")
        self.selection_changed.emit(list(self._selected_positions))

    def _on_clear_selection(self):
        for circle in self._circles.values():
            circle.set_selected(False)
        self._selected_positions.clear()
        self._count_label.setText("0 selected")
        self.selection_changed.emit([])

    def get_selected_positions(self) -> List[Tuple[int, int]]:
        return list(self._selected_positions)

    # --- droplet config (Opentrons manifest) --------------------------------
    def _on_load_config(self):
        """Load an Opentrons droplet manifest (.json) and attach sample-name
        tooltips to the matching wells."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Config", "", "Droplet manifest (*.json)")
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config '{path}': {e}")
            QMessageBox.critical(self, "Load Config",
                                 f"Could not read the config file:\n{e}")
            return
        if not isinstance(data, list):
            QMessageBox.warning(
                self, "Load Config",
                "This file doesn't look like an Opentrons droplet manifest.")
            return
        self._apply_droplet_config(data)

    def _apply_droplet_config(self, entries: list):
        """Map each manifest droplet_position to its sample and set the
        matching well-circle tooltips to the sample name. Well names align
        directly: 'A4' in this panel == droplet_position 'A4' in the manifest."""
        self._droplet_map = {}
        for e in entries:
            if not isinstance(e, dict):
                continue
            well = str(e.get("droplet_position", "")).strip().upper()
            if well:
                self._droplet_map[well] = e

        for (row, col), circle in self._circles.items():
            info = self._droplet_map.get(well_name(row, col))
            circle._tip_text = self._droplet_tooltip(info) if info else ""

        n = len(self._droplet_map)
        logger.info(f"Loaded droplet config: {n} droplets")
        if not self._multi_select and hasattr(self, "_info_label"):
            self._info_label.setText(f"Config loaded: {n} droplets")

    @staticmethod
    def _droplet_tooltip(info: dict) -> str:
        parts = [str(info.get("sample_name", "?"))]
        num = info.get("droplet_number")
        if num is not None:
            parts.append(f"droplet {num}")
        return "  |  ".join(parts)

    # --- custom 0.5 s hover tooltip -----------------------------------------
    def _on_circle_hover_enter(self, circle, screen_pos):
        text = getattr(circle, '_tip_text', '')
        if not text:
            self._hover_timer.stop()
            QToolTip.hideText()
            return
        self._hover_text = text
        self._hover_pos = screen_pos
        self._hover_timer.start()   # shows after 500 ms (see _show_hover_tooltip)

    def _on_circle_hover_leave(self):
        self._hover_timer.stop()
        QToolTip.hideText()

    def _show_hover_tooltip(self):
        if self._hover_text and self._hover_pos is not None:
            pos = self._hover_pos
            if hasattr(pos, 'toPoint'):
                pos = pos.toPoint()
            QToolTip.showText(pos, self._hover_text, self._view)

    def set_current_position(self, row: int, col: int):
        """Set the currently highlighted well (e.g., after stage arrives)."""
        if self._current_position and self._current_position in self._circles:
            self._circles[self._current_position].set_selected(False)
        pos = (row, col)
        if pos in self._circles:
            self._current_position = pos
            self._circles[pos].set_selected(True)
            if not self._multi_select:
                self._info_label.setText(f"At {well_name(row, col)}")

    def set_grid_enabled(self, enabled: bool):
        """Enable or disable the grid.

        When disabled, all circles turn grey and clicks are ignored,
        but the view remains scrollable/pannable.
        """
        self._grid_enabled = enabled
        for circle in self._circles.values():
            circle.set_disabled(not enabled)
        if not self._multi_select:
            if enabled:
                self._info_label.setText("Click a well to move stage")
            else:
                self._info_label.setText("Reference the stage first")

    def showEvent(self, event):
        """Scroll to bottom on first show so slide 1 is visible."""
        super().showEvent(event)
        if not self._initial_scroll_done:
            self._initial_scroll_done = True
            QTimer.singleShot(50, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        """Scroll the view so the bottom of the grid is visible."""
        vbar = self._view.verticalScrollBar()
        vbar.setValue(vbar.maximum())

    def resizeEvent(self, event):
        """On resize, scale up to fit but never shrink below the base scale.

        If the view is large enough, the grid scales up to fill it.
        If the view is too small, the grid keeps its base size and
        scrollbars appear automatically.
        """
        super().resizeEvent(event)
        if not self._scene.sceneRect().isValid():
            return

        # Compute the scale that would fit the scene into the current viewport
        vp = self._view.viewport().size()
        sr = self._scene.sceneRect()
        if sr.width() <= 0 or sr.height() <= 0:
            return

        scale_x = vp.width() / sr.width()
        scale_y = vp.height() / sr.height()
        fit_scale = min(scale_x, scale_y)

        # Base scale = the transform set on first layout (full-size grid)
        base_scale = self._base_transform.m11()  # horizontal scale factor

        if fit_scale >= base_scale:
            # View is large enough — scale up to fit
            self._view.fitInView(sr, Qt.AspectRatioMode.KeepAspectRatio)
        else:
            # View is too small — keep base scale, let scrollbars handle overflow
            self._view.setTransform(self._base_transform)
