"""
Application-wide guard that stops the mouse wheel from changing the value
of entry widgets (spin boxes and combo boxes).

By default, scrolling the wheel while the pointer is over a QSpinBox /
QDoubleSpinBox / QComboBox increments, decrements, or cycles its value.
This makes it far too easy to change a setting by accident while simply
scrolling a panel. This filter swallows wheel events targeted at those
widgets so their values can only be changed by typing or by clicking the
spin arrows / dropdown.

Install once on the QApplication; it then covers every window and dialog
in the process, including ones created later.
"""

from PyQt6.QtCore import QObject, QEvent
from PyQt6.QtWidgets import QAbstractSpinBox, QComboBox


class WheelGuard(QObject):
    """Event filter that consumes wheel events on value-entry widgets.

    QAbstractSpinBox covers QSpinBox and QDoubleSpinBox; QComboBox covers
    dropdown selectors. Returning True consumes the wheel event so the
    widget never sees it and its value is left unchanged.
    """

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.Type.Wheel
                and isinstance(obj, (QAbstractSpinBox, QComboBox))):
            event.ignore()
            return True  # consume — value can only change via typing/clicks
        return super().eventFilter(obj, event)


def install_wheel_guard(app) -> WheelGuard:
    """Install the wheel guard on the given QApplication.

    Returns the filter instance. It is parented to ``app`` so it stays
    alive for the application's lifetime.
    """
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    return guard
