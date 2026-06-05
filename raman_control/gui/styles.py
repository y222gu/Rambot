"""
Dark theme styles for the Raman Control application.

Provides a modern, professional dark theme with:
- Dark backgrounds for reduced eye strain
- Cyan/teal accents for active elements
- Clear visual hierarchy
- Hover and pressed states for buttons
"""

# Color palette
COLORS = {
    # Backgrounds
    'bg_dark': '#1a1a1a',
    'bg_medium': '#242424',
    'bg_light': '#2d2d2d',
    'bg_elevated': '#363636',

    # Borders
    'border_dark': '#3d3d3d',
    'border_light': '#4d4d4d',
    'border_focus': '#00a0a0',

    # Text
    'text_primary': '#e0e0e0',
    'text_secondary': '#a0a0a0',
    'text_disabled': '#606060',

    # Accents
    'accent_primary': '#00b4b4',  # Cyan/teal
    'accent_hover': '#00d4d4',
    'accent_pressed': '#008080',

    # Status colors
    'success': '#4caf50',
    'warning': '#ff9800',
    'error': '#f44336',
    'info': '#2196f3',

    # Special
    'selection': '#264f78',
    'highlight': '#3d5a80',
}


DARK_STYLESHEET = f'''
/* ============================================
   MAIN WINDOW & CONTAINERS
   ============================================ */

QMainWindow {{
    background-color: {COLORS['bg_dark']};
}}

QFrame {{
    border: none;
}}

/* ============================================
   GROUP BOXES
   ============================================ */

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

/* ============================================
   BUTTONS
   ============================================ */

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

/* Primary action buttons */
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

QPushButton[primary="true"]:pressed {{
    background-color: {COLORS['accent_pressed']};
}}

/* Danger buttons */
QPushButton[danger="true"] {{
    background-color: {COLORS['error']};
    border-color: {COLORS['error']};
    color: white;
}}

QPushButton[danger="true"]:hover {{
    background-color: #ff5252;
}}

/* Direction buttons (compact) */
QPushButton[direction="true"] {{
    padding: 4px;
    min-width: 36px;
    min-height: 36px;
    font-size: 16px;
    font-weight: bold;
}}

/* ============================================
   INPUT FIELDS
   ============================================ */

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

QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_disabled']};
}}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 0;
    height: 0;
    border: none;
}}

/* ============================================
   COMBO BOX
   ============================================ */

QComboBox {{
    background-color: {COLORS['bg_light']};
    border: 1px solid {COLORS['border_dark']};
    border-radius: 4px;
    padding: 5px 10px;
    color: {COLORS['text_primary']};
    min-height: 18px;
}}

QComboBox:hover {{
    border-color: {COLORS['accent_primary']};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {COLORS['text_secondary']};
    margin-right: 6px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_elevated']};
    border: 1px solid {COLORS['border_light']};
    color: {COLORS['text_primary']};
    selection-background-color: {COLORS['selection']};
    outline: none;
}}

/* ============================================
   LABELS
   ============================================ */

QLabel, QCheckBox, QRadioButton {{
    background: transparent;
    border: none;
    padding: 0px;
    color: {COLORS['text_primary']};
}}

/* ---- Checkbox indicators ---- */
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 2px solid {COLORS['text_primary']};
    border-radius: 3px;
    background-color: transparent;
}}

QCheckBox::indicator:hover {{
    border-color: white;
}}

QCheckBox::indicator:checked {{
    border-color: {COLORS['text_primary']};
    background-color: transparent;
}}

/* ---- Radio button indicators ---- */
QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border: 2px solid {COLORS['border_light']};
    border-radius: 9px;
    background-color: transparent;
}}

QRadioButton::indicator:hover {{
    border-color: {COLORS['accent_primary']};
}}

QRadioButton::indicator:checked {{
    background-color: {COLORS['accent_primary']};
    border-color: {COLORS['accent_primary']};
}}

QLabel[heading="true"] {{
    font-size: 15px;
    font-weight: 600;
    color: {COLORS['accent_primary']};
}}

QLabel[secondary="true"] {{
    color: {COLORS['text_secondary']};
    font-size: 12px;
}}

QLabel[status="connected"] {{
    color: {COLORS['success']};
    font-weight: 600;
}}

QLabel[status="disconnected"] {{
    color: {COLORS['error']};
}}

QLabel[status="warning"] {{
    color: {COLORS['warning']};
}}

/* ============================================
   SLIDERS
   ============================================ */

QSlider::groove:horizontal {{
    border: none;
    height: 6px;
    background-color: {COLORS['bg_light']};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background-color: {COLORS['accent_primary']};
    border: none;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background-color: {COLORS['accent_hover']};
}}

QSlider::sub-page:horizontal {{
    background-color: {COLORS['accent_primary']};
    border-radius: 3px;
}}

/* ============================================
   TAB WIDGET
   ============================================ */

QTabWidget::pane {{
    background-color: {COLORS['bg_medium']};
    border: 1px solid {COLORS['border_dark']};
    border-radius: 8px;
    padding: 8px;
}}

QTabBar::tab {{
    background-color: {COLORS['bg_light']};
    border: 1px solid {COLORS['border_dark']};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 10px 20px;
    margin-right: 2px;
    color: {COLORS['text_secondary']};
}}

QTabBar::tab:selected {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['accent_primary']};
    font-weight: 600;
}}

QTabBar::tab:hover:!selected {{
    background-color: {COLORS['bg_elevated']};
    color: {COLORS['text_primary']};
}}

/* ============================================
   STATUS BAR
   ============================================ */

QStatusBar {{
    background-color: {COLORS['bg_medium']};
    border-top: 1px solid {COLORS['border_dark']};
    padding: 4px;
}}

QStatusBar::item {{
    border: none;
}}

/* ============================================
   MENU BAR
   ============================================ */

QMenuBar {{
    background-color: {COLORS['bg_medium']};
    border-bottom: 1px solid {COLORS['border_dark']};
    padding: 4px;
}}

QMenuBar::item {{
    background-color: transparent;
    padding: 6px 12px;
    border-radius: 4px;
}}

QMenuBar::item:selected {{
    background-color: {COLORS['bg_elevated']};
}}

QMenu {{
    background-color: {COLORS['bg_elevated']};
    border: 1px solid {COLORS['border_dark']};
    border-radius: 6px;
    padding: 4px;
}}

QMenu::item {{
    padding: 8px 24px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background-color: {COLORS['selection']};
}}

QMenu::separator {{
    height: 1px;
    background-color: {COLORS['border_dark']};
    margin: 4px 8px;
}}

/* ============================================
   SPLITTER
   ============================================ */

QSplitter::handle {{
    background-color: {COLORS['border_dark']};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

QSplitter::handle:hover {{
    background-color: {COLORS['accent_primary']};
}}

/* ============================================
   SCROLL BARS
   ============================================ */

QScrollBar:vertical {{
    background-color: {COLORS['bg_dark']};
    width: 12px;
    border: none;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['border_light']};
    border-radius: 6px;
    min-height: 30px;
    margin: 2px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['text_secondary']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {COLORS['bg_dark']};
    height: 12px;
    border: none;
}}

QScrollBar::handle:horizontal {{
    background-color: {COLORS['border_light']};
    border-radius: 6px;
    min-width: 30px;
    margin: 2px;
}}

/* ============================================
   MESSAGE BOX
   ============================================ */

QMessageBox {{
    background-color: {COLORS['bg_medium']};
}}

QMessageBox QLabel {{
    color: {COLORS['text_primary']};
}}

/* ============================================
   TOOLTIPS
   ============================================ */

QToolTip {{
    background-color: {COLORS['bg_elevated']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border_light']};
    border-radius: 4px;
    padding: 6px 10px;
}}

/* ============================================
   PROGRESS BAR
   ============================================ */

QProgressBar {{
    background-color: {COLORS['bg_light']};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {COLORS['accent_primary']};
    border-radius: 4px;
}}

/* ============================================
   CANVAS (for sample grid)
   ============================================ */

QGraphicsView {{
    background-color: {COLORS['bg_light']};
    border: 1px solid {COLORS['border_dark']};
    border-radius: 6px;
}}
'''


def _generate_checkmark_icon(size: int = 18) -> str:
    """Generate a white checkmark icon on transparent background.

    Returns the file path (forward slashes) for use in stylesheet
    ``image: url(...)`` rules.
    """
    import tempfile, os
    from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor
    from PyQt6.QtCore import Qt, QPoint

    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor('white'))
    pen.setWidth(3)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    # Checkmark path scaled to size
    p.drawLine(QPoint(3, 9), QPoint(7, 13))
    p.drawLine(QPoint(7, 13), QPoint(14, 4))
    p.end()

    path = os.path.join(tempfile.gettempdir(), 'raman_checkmark.png')
    pixmap.save(path)
    return path.replace('\\', '/')


def apply_dark_theme(app):
    """Apply dark theme to the entire application.

    Uses QPalette for base window/widget colours so child widgets
    inherit properly, then layers the stylesheet on top for borders,
    spacing, and accent colours.  This avoids the dark-box artefact
    caused by the old ``QWidget { background-color }`` stylesheet rule
    fighting with ``QLabel { background: transparent }``.
    """
    from PyQt6.QtGui import QPalette, QColor, QFont

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLORS['bg_dark']))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLORS['text_primary']))
    palette.setColor(QPalette.ColorRole.Base, QColor(COLORS['bg_light']))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLORS['bg_medium']))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLORS['text_primary']))
    palette.setColor(QPalette.ColorRole.Button, QColor(COLORS['bg_elevated']))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLORS['text_primary']))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLORS['accent_primary']))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(COLORS['bg_dark']))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(COLORS['bg_elevated']))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(COLORS['text_primary']))

    # Disabled state
    palette.setColor(QPalette.ColorGroup.Disabled,
                     QPalette.ColorRole.WindowText, QColor(COLORS['text_disabled']))
    palette.setColor(QPalette.ColorGroup.Disabled,
                     QPalette.ColorRole.Text, QColor(COLORS['text_disabled']))
    palette.setColor(QPalette.ColorGroup.Disabled,
                     QPalette.ColorRole.ButtonText, QColor(COLORS['text_disabled']))

    app.setPalette(palette)
    app.setFont(QFont("Segoe UI", 10))

    # Generate white checkmark icon and inject into stylesheet
    checkmark_path = _generate_checkmark_icon()
    stylesheet = DARK_STYLESHEET + f'''
    QCheckBox::indicator:checked {{
        image: url({checkmark_path});
    }}
    '''
    app.setStyleSheet(stylesheet)


def get_pyqtgraph_theme():
    """Get PyQtGraph theme settings for dark mode."""
    return {
        'background': COLORS['bg_medium'],
        'foreground': COLORS['text_primary'],
        'antialias': True,
    }
