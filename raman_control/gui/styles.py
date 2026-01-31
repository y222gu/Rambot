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

QWidget {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_primary']};
    font-family: "Segoe UI", "SF Pro Display", -apple-system, sans-serif;
    font-size: 13px;
}}

QFrame {{
    background-color: {COLORS['bg_dark']};
    border: none;
}}

/* ============================================
   GROUP BOXES
   ============================================ */

QGroupBox {{
    background-color: {COLORS['bg_medium']};
    border: 1px solid {COLORS['border_dark']};
    border-radius: 8px;
    margin-top: 16px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
    font-size: 13px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: 4px;
    padding: 0 8px;
    background-color: {COLORS['bg_medium']};
    color: {COLORS['accent_primary']};
}}

/* ============================================
   BUTTONS
   ============================================ */

QPushButton {{
    background-color: {COLORS['bg_elevated']};
    border: 1px solid {COLORS['border_dark']};
    border-radius: 6px;
    padding: 8px 16px;
    min-height: 20px;
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
    background-color: {COLORS['bg_medium']};
    border-color: {COLORS['border_dark']};
    color: {COLORS['text_disabled']};
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

/* ============================================
   LABELS
   ============================================ */

QLabel {{
    background-color: transparent;
    color: {COLORS['text_primary']};
    padding: 2px;
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


def apply_dark_theme(app):
    """Apply dark theme to the entire application."""
    app.setStyleSheet(DARK_STYLESHEET)


def get_pyqtgraph_theme():
    """Get PyQtGraph theme settings for dark mode."""
    return {
        'background': COLORS['bg_medium'],
        'foreground': COLORS['text_primary'],
        'antialias': True,
    }
