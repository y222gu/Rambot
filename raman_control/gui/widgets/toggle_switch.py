"""
iOS-style sliding toggle switch (a checkable button drawn as a pill track
with a round knob that slides between the two ends).

Behaves like any ``QAbstractButton``: ``toggled(bool)``, ``isChecked()``,
``setChecked()``. The knob position is animated on every state change.
"""

from PyQt6.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, QSize, QRectF, pyqtProperty,
)
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen
from PyQt6.QtWidgets import QAbstractButton

from raman_control.gui.styles import COLORS


class ToggleSwitch(QAbstractButton):
    """Sliding on/off switch. Off = knob left, grey track; on = knob right,
    accent-coloured track."""

    _TRACK_W = 40
    _TRACK_H = 22
    _MARGIN = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._offset = 0.0  # 0.0 = knob fully left, 1.0 = fully right
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.toggled.connect(self._animate)
        self._on_color = QColor(COLORS.get('accent_primary', '#00a0a0'))
        self._off_color = QColor(COLORS.get('border_light', '#4d4d4d'))
        self._knob_color = QColor('#f0f0f0')
        self._disabled_color = QColor('#3a3a3a')

    # --- animated property ---------------------------------------------------

    def _get_offset(self) -> float:
        return self._offset

    def _set_offset(self, value: float):
        self._offset = float(value)
        self.update()

    offset = pyqtProperty(float, _get_offset, _set_offset)

    def _animate(self, checked: bool):
        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def setChecked(self, checked: bool):  # noqa: N802 (Qt API)
        # Keep the knob in sync when the state is set programmatically before
        # the widget is shown (no animation needed then).
        super().setChecked(checked)
        if not self.isVisible():
            self._offset = 1.0 if checked else 0.0
            self.update()

    # --- geometry / painting ---------------------------------------------------

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        return QSize(self._TRACK_W, self._TRACK_H)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        return self.sizeHint()

    def paintEvent(self, event):  # noqa: N802 (Qt API)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self._TRACK_W, self._TRACK_H
        x0 = (self.width() - w) / 2
        y0 = (self.height() - h) / 2
        track = QRectF(x0, y0, w, h)

        if not self.isEnabled():
            track_color = self._disabled_color
        else:
            # Blend grey -> accent with the knob position.
            t = max(0.0, min(1.0, self._offset))
            a, b = self._off_color, self._on_color
            track_color = QColor(
                int(a.red() + (b.red() - a.red()) * t),
                int(a.green() + (b.green() - a.green()) * t),
                int(a.blue() + (b.blue() - a.blue()) * t))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(track_color))
        p.drawRoundedRect(track, h / 2, h / 2)

        d = h - 2 * self._MARGIN
        travel = w - d - 2 * self._MARGIN
        kx = x0 + self._MARGIN + travel * max(0.0, min(1.0, self._offset))
        ky = y0 + self._MARGIN
        p.setPen(QPen(QColor(0, 0, 0, 60), 1))
        p.setBrush(QBrush(self._knob_color if self.isEnabled()
                          else QColor('#6a6a6a')))
        p.drawEllipse(QRectF(kx, ky, d, d))
        p.end()
