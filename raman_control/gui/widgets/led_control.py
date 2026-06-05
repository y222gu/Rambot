"""
LED illumination control widget (lives at the far-left of the status bar).

A compact toggle button ("Turn On" / "Turn Off") + brightness slider for the
COB illumination LED. All hardware I/O is dispatched to :class:`LEDThread`
(fire-and-forget).

The controls are ALWAYS clickable (like the original noodlepy GUI). Commands
are no-ops when the board isn't connected, and the thread (re)connects on
demand — so the user is never locked out of the slider/button. The bulb's
tooltip reflects the current connection state.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QSlider,
                             QWidget)

from raman_control.gui.styles import COLORS
from raman_control.gui.threads.led_thread import LEDThread


class LEDControlWidget(QWidget):
    """Toggle + brightness slider for the illumination LED (status-bar strip)."""

    def __init__(self, led_thread: LEDThread, default_brightness: int = 255,
                 parent=None):
        super().__init__(parent)
        self._led = led_thread
        self._default_brightness = int(default_brightness)
        self._is_on = False

        self._build()

        # Status only — the controls are never disabled.
        led_thread.connected.connect(self._on_connected)
        led_thread.error.connect(self._on_error)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build(self) -> None:
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 6, 0)  # flush to the far-left corner
        lay.setSpacing(6)

        # Toggle button. Checked = on; the text (with a 💡) shows the action.
        self._toggle_btn = QPushButton("\U0001F4A1 Turn On")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setFixedWidth(104)
        self._toggle_btn.setToolTip("Toggle the illumination LED")
        self._toggle_btn.clicked.connect(self._on_toggle)
        lay.addWidget(self._toggle_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 255)
        self._slider.setValue(self._default_brightness)
        self._slider.setFixedWidth(140)
        self._slider.setToolTip("LED brightness")
        self._slider.valueChanged.connect(self._on_slider)
        lay.addWidget(self._slider)

        self._pct_lbl = QLabel("100%")
        self._pct_lbl.setFixedWidth(38)
        self._pct_lbl.setAlignment(Qt.AlignmentFlag.AlignRight
                                   | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self._pct_lbl)
        # Absorb any extra width so the button/slider stay packed at the left
        # edge (otherwise the layout centres the fixed-width items in their
        # cells, pushing them off the corner).
        lay.addStretch(1)

        self._refresh_button()
        self._update_pct(self._slider.value())

    def _refresh_button(self) -> None:
        """Set the button text + style for the current on/off state."""
        on = self._toggle_btn.isChecked()
        self._toggle_btn.setText("\U0001F4A1 Turn Off" if on else "\U0001F4A1 Turn On")
        if on:
            self._toggle_btn.setStyleSheet(f'''
                QPushButton {{
                    background-color: {COLORS['accent_primary']};
                    border: 1px solid {COLORS['accent_hover']};
                    border-radius: 4px;
                    color: {COLORS['bg_dark']};
                    font-weight: 600;
                    padding: 4px 6px;
                }}
                QPushButton:hover {{ background-color: {COLORS['accent_hover']}; }}
            ''')
        else:
            self._toggle_btn.setStyleSheet(f'''
                QPushButton {{
                    background-color: transparent;
                    border: 1px solid #4d4d4d;
                    border-radius: 4px;
                    color: {COLORS['text_primary']};
                    padding: 4px 6px;
                }}
                QPushButton:hover {{ border-color: {COLORS['accent_primary']}; }}
            ''')

    def _update_pct(self, value: int) -> None:
        self._pct_lbl.setText(f"{round(value / 255 * 100)}%")

    # ------------------------------------------------------------------
    # Logic
    # ------------------------------------------------------------------

    def _on_toggle(self) -> None:
        self._is_on = self._toggle_btn.isChecked()
        if self._is_on:
            brightness = self._slider.value() or self._default_brightness
            if not self._slider.value():
                self._slider.setValue(brightness)  # reflect the fallback level
            self._led.request_on(brightness)
        else:
            self._led.request_off()
        self._refresh_button()

    def _on_slider(self, value: int) -> None:
        self._update_pct(value)
        if self._is_on:
            self._led.request_set_brightness(value)

    def reset_off(self) -> None:
        """Reflect the light being turned off (e.g. on global Disconnect)."""
        self._is_on = False
        self._toggle_btn.setChecked(False)
        self._refresh_button()

    # ------------------------------------------------------------------
    # Thread signals (status only — never disables the controls)
    # ------------------------------------------------------------------

    def _on_connected(self, port: str) -> None:
        self._toggle_btn.setToolTip(f"Illumination LED — connected on {port}")

    def _on_error(self, message: str) -> None:
        self._toggle_btn.setToolTip(f"Illumination LED — {message}")
