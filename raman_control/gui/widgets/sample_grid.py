"""
Sample drop grid widget for selecting acquisition positions.

Features:
- Interactive grid of sample positions
- Hover effects and selection highlighting
- Position coordinate display
- Multi-selection support for protocols
"""

from typing import Optional, List, Tuple, Set
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QGraphicsView, QGraphicsScene,
    QGraphicsEllipseItem, QGraphicsTextItem, QFrame,
    QGraphicsRectItem, QRubberBand
)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF, QRect, QPoint
from PyQt6.QtGui import QBrush, QPen, QColor, QFont, QMouseEvent

from raman_control.gui.styles import COLORS
from raman_control.config.hardware_config import SampleDropConfig
from raman_control.config.logging_config import get_logger

logger = get_logger('sample_grid')


class SampleCircle(QGraphicsEllipseItem):
    """Interactive circle representing a sample drop position."""

    def __init__(self, row: int, col: int, x: float, y: float, radius: float):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.row = row
        self.col = col
        self.setPos(x, y)

        self._selected = False
        self._hovered = False
        self._active = True

        # Enable hover events
        self.setAcceptHoverEvents(True)

        # Set initial style
        self._update_style()

    def _update_style(self):
        """Update visual style based on state."""
        if not self._active:
            brush = QBrush(QColor('#404040'))
            pen = QPen(QColor('#505050'), 1)
        elif self._selected:
            brush = QBrush(QColor(COLORS['warning']))
            pen = QPen(QColor('#ffb74d'), 2)
        elif self._hovered:
            brush = QBrush(QColor(COLORS['accent_hover']))
            pen = QPen(QColor(COLORS['accent_primary']), 2)
        else:
            brush = QBrush(QColor(COLORS['info']))
            pen = QPen(QColor('#1976d2'), 1)

        self.setBrush(brush)
        self.setPen(pen)

    def set_selected(self, selected: bool):
        self._selected = selected
        self._update_style()

    def set_active(self, active: bool):
        self._active = active
        self._update_style()

    def hoverEnterEvent(self, event):
        if self._active:
            self._hovered = True
            self._update_style()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self._update_style()
        super().hoverLeaveEvent(event)


class SampleGridWidget(QWidget):
    """
    Interactive sample drop grid for position selection.

    Signals:
        position_selected: Emitted when a position is clicked (row, col)
        position_hovered: Emitted when hovering over a position (row, col)
        selection_changed: Emitted when selection set changes (list of (row, col))
    """

    position_selected = pyqtSignal(int, int)
    position_hovered = pyqtSignal(int, int)
    selection_changed = pyqtSignal(list)

    def __init__(
        self,
        config: Optional[SampleDropConfig] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self._config = config or SampleDropConfig()
        self._circles: dict[Tuple[int, int], SampleCircle] = {}
        self._selected_positions: Set[Tuple[int, int]] = set()
        self._current_position: Optional[Tuple[int, int]] = None
        self._multi_select = False

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Grid display
        grid_group = QGroupBox("Sample Positions")
        grid_layout = QVBoxLayout(grid_group)

        # Graphics view for the grid
        self._scene = QGraphicsScene()
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(self._view.renderHints().Antialiasing)
        self._view.setStyleSheet(f'''
            QGraphicsView {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['border_dark']};
                border-radius: 6px;
            }}
        ''')
        self._view.setMinimumSize(250, 200)

        # Create the grid
        self._create_grid()

        grid_layout.addWidget(self._view)

        # Position info
        self._info_label = QLabel("Click a position to select")
        self._info_label.setStyleSheet(f'color: {COLORS["text_secondary"]}; font-size: 12px;')
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(self._info_label)

        main_layout.addWidget(grid_group)

        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)

        # Select all button
        self._select_all_btn = QPushButton("Select All")
        self._select_all_btn.clicked.connect(self._on_select_all)
        controls_layout.addWidget(self._select_all_btn)

        # Clear selection button
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._on_clear_selection)
        controls_layout.addWidget(self._clear_btn)

        controls_layout.addStretch()

        # Selection count
        self._count_label = QLabel("0 selected")
        self._count_label.setStyleSheet(f'color: {COLORS["accent_primary"]};')
        controls_layout.addWidget(self._count_label)

        main_layout.addLayout(controls_layout)

    def _create_grid(self):
        """Create the sample position grid."""
        self._scene.clear()
        self._circles.clear()

        pattern = self._config.grid_pattern
        spacing = self._config.spacing
        radius = self._config.circle_radius

        # Skip positions (corners for typical slide layout)
        skip_positions = {(0, 0), (3, 0), (0, 4), (3, 4)}

        max_rows = max(pattern)

        for col, num_rows in enumerate(pattern):
            # Vertical offset to center columns with fewer circles
            v_offset = (max_rows - num_rows) * spacing / 2

            for row in range(num_rows):
                true_row = row + int(v_offset / spacing)

                # Skip corner positions
                if (true_row, col) in skip_positions:
                    continue

                x = col * spacing + spacing / 2
                y = v_offset + row * spacing + spacing / 2

                circle = SampleCircle(true_row, col, x, y, radius)
                self._scene.addItem(circle)
                self._circles[(true_row, col)] = circle

        # Fit view to content
        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-20, -20, 20, 20))
        self._view.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        # Drag selection state
        self._drag_start: Optional[QPoint] = None
        self._rubber_band: Optional[QRubberBand] = None

        # Connect mouse events for drag selection
        self._view.mousePressEvent = self._on_view_mouse_press
        self._view.mouseMoveEvent = self._on_view_mouse_move
        self._view.mouseReleaseEvent = self._on_view_mouse_release

    def _on_view_mouse_press(self, event: QMouseEvent):
        """Handle mouse press - start drag or click selection."""
        self._drag_start = event.pos()

        # Create rubber band for drag selection
        if self._multi_select:
            self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self._view)
            self._rubber_band.setGeometry(QRect(self._drag_start, self._drag_start))
            self._rubber_band.show()

    def _on_view_mouse_move(self, event: QMouseEvent):
        """Handle mouse move - update rubber band."""
        if self._rubber_band and self._drag_start:
            self._rubber_band.setGeometry(QRect(self._drag_start, event.pos()).normalized())

    def _on_view_mouse_release(self, event: QMouseEvent):
        """Handle mouse release - finish drag or click selection."""
        if self._rubber_band:
            # Drag selection
            rect = self._rubber_band.geometry()
            self._rubber_band.hide()
            self._rubber_band = None

            # Check if it was a drag (not just a click)
            if rect.width() > 5 and rect.height() > 5:
                self._select_items_in_rect(rect)
                self._drag_start = None
                return

        # Single click - handle as before
        if self._drag_start:
            scene_pos = self._view.mapToScene(self._drag_start)
            item = self._scene.itemAt(scene_pos, self._view.transform())

            if isinstance(item, SampleCircle):
                pos = (item.row, item.col)

                if self._multi_select:
                    # Toggle selection
                    if pos in self._selected_positions:
                        self._selected_positions.discard(pos)
                        item.set_selected(False)
                    else:
                        self._selected_positions.add(pos)
                        item.set_selected(True)
                else:
                    # Single selection - clear others first
                    for p, c in self._circles.items():
                        c.set_selected(p == pos)
                    self._selected_positions = {pos}

                self._current_position = pos
                self._update_info()
                self.position_selected.emit(item.row, item.col)
                self.selection_changed.emit(list(self._selected_positions))

        self._drag_start = None

    def _select_items_in_rect(self, rect: QRect):
        """Select all items within the given rectangle."""
        # Convert rect to scene coordinates
        top_left = self._view.mapToScene(rect.topLeft())
        bottom_right = self._view.mapToScene(rect.bottomRight())
        scene_rect = QRectF(top_left, bottom_right).normalized()

        # Find all circles in the rect
        for pos, circle in self._circles.items():
            if circle._active and scene_rect.contains(circle.pos()):
                circle.set_selected(True)
                self._selected_positions.add(pos)

        self._update_info()
        self.selection_changed.emit(list(self._selected_positions))

    def _on_select_all(self):
        """Select all active positions."""
        self._selected_positions.clear()
        for pos, circle in self._circles.items():
            if circle._active:
                circle.set_selected(True)
                self._selected_positions.add(pos)

        self._update_info()
        self.selection_changed.emit(list(self._selected_positions))

    def _on_clear_selection(self):
        """Clear all selections."""
        for circle in self._circles.values():
            circle.set_selected(False)

        self._selected_positions.clear()
        self._current_position = None
        self._update_info()
        self.selection_changed.emit([])

    def _update_info(self):
        """Update the info label."""
        count = len(self._selected_positions)
        self._count_label.setText(f"{count} selected")

        if self._current_position:
            row, col = self._current_position
            self._info_label.setText(f"Position: Row {row}, Column {col}")
        else:
            self._info_label.setText("Click a position to select")

    def set_current_position(self, row: int, col: int):
        """Set the current active position (e.g., where stage is)."""
        # Reset previous current
        if self._current_position and self._current_position in self._circles:
            self._circles[self._current_position].set_selected(
                self._current_position in self._selected_positions
            )

        # Set new current
        pos = (row, col)
        if pos in self._circles:
            self._current_position = pos
            self._circles[pos].set_selected(True)
            self._update_info()

    def get_selected_positions(self) -> List[Tuple[int, int]]:
        """Get list of selected positions."""
        return list(self._selected_positions)

    def set_multi_select(self, enabled: bool):
        """Enable or disable multi-selection mode."""
        self._multi_select = enabled

    def set_positions_active(self, positions: List[Tuple[int, int]], active: bool):
        """Set active state for specific positions."""
        for pos in positions:
            if pos in self._circles:
                self._circles[pos].set_active(active)

    def resizeEvent(self, event):
        """Handle resize to maintain grid fit."""
        super().resizeEvent(event)
        if self._scene.sceneRect().isValid():
            self._view.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
