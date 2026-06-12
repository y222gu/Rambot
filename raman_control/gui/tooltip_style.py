"""
Application-wide tweak that makes tooltips appear almost immediately on hover.

Qt's default tooltip wake-up delay is roughly 700 ms, which feels sluggish for
a control-heavy panel where the tooltips explain what each button / entry box
does. This installs a proxy style that overrides only the tooltip wake-up delay
style hint, leaving every other style behaviour (and the dark stylesheet, which
is applied separately) untouched.

Install once on the QApplication; it then covers every window and dialog in the
process, including ones created later.
"""

from PyQt6.QtWidgets import QProxyStyle, QStyle


# Wake-up delay in milliseconds. Small enough to feel instant, but not zero so a
# quick sweep of the pointer across many controls does not flash every tooltip.
DEFAULT_WAKEUP_MS = 100


class FastTooltipStyle(QProxyStyle):
    """Proxy style that only shortens the tooltip wake-up delay.

    Every other style hint is delegated to the wrapped base style, so widget
    rendering is unchanged.
    """

    def __init__(self, wake_up_ms: int = DEFAULT_WAKEUP_MS, base_style=None):
        super().__init__(base_style)
        self._wake_up_ms = int(wake_up_ms)

    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.StyleHint.SH_ToolTip_WakeUpDelay:
            return self._wake_up_ms
        return super().styleHint(hint, option, widget, returnData)


def install_fast_tooltips(app, wake_up_ms: int = DEFAULT_WAKEUP_MS) -> FastTooltipStyle:
    """Make tooltips appear almost immediately on the given QApplication.

    Wraps the application's current style in a FastTooltipStyle so the only
    change is the tooltip wake-up delay. Returns the installed proxy style.
    """
    proxy = FastTooltipStyle(wake_up_ms, app.style())
    app.setStyle(proxy)
    return proxy
