"""
PyQt6 thread for the COB illumination LED strip.

Owns the LED serial connection (single owner, off the GUI thread). Brightness
commands are queued from the GUI and applied on this worker thread, so the GUI
never blocks on serial I/O — important because ``connect()`` resets the Arduino
and waits ~2 s for its bootloader.

Port discipline (this is what makes it reliable on the cheap CH340 board this
system uses):
  * Open the port once and keep it open for the whole session.
  * Close the port GRACEFULLY when the thread stops (purge, drop DTR/RTS, then
    pyserial close()). A clean close lets the CH340 driver release the port so
    the NEXT launch can reopen it. The opposite — relying on a hard
    process-exit handle release with no real close — is what used to leave the
    chip in the Windows "device not functioning" (error 31) state and forced a
    physical re-plug on every launch.
  * Don't churn the port during normal operation (no close/reopen on every
    write failure). A stuck open is retried a few times inside
    ``LEDController.connect()`` so error 31 can self-recover without a re-plug.

If the board isn't connected (e.g. it was unplugged), pressing the button or
moving the slider re-attempts open() — enough to pick the board back up.
"""

import queue
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from raman_control.config.hardware_config import LEDConfig
from raman_control.config.logging_config import get_logger
from raman_control.hardware.led_controller import LEDController

logger = get_logger('led_thread')


class LEDThread(QThread):
    """Worker thread owning the LED serial connection.

    Signals:
        connected:          (port) emitted whenever the board (re)connects
        disconnected:       emitted when the thread stops
        error:              (message) on connect or write failure
        brightness_changed: (0-255) after a brightness command is applied
    """

    connected = pyqtSignal(str)
    disconnected = pyqtSignal()
    error = pyqtSignal(str)
    brightness_changed = pyqtSignal(int)

    _QUIT = object()

    def __init__(self, config: Optional[LEDConfig] = None, parent=None):
        super().__init__(parent)
        self._config = config or LEDConfig()
        self._controller: Optional[LEDController] = None
        self._queue: "queue.Queue" = queue.Queue()
        self._running = False

    def _ensure_connected(self) -> bool:
        """Open the port if it isn't already open (single attempt, no close).
        Emits ``connected`` on success."""
        if self._controller is None:
            self._controller = LEDController(self._config)
        if self._controller.is_connected:
            return True
        if self._controller.connect():
            self.connected.emit(self._controller.port or "")
            return True
        return False

    def _apply(self, value: int) -> None:
        """Write a brightness value; (re)open on demand, never close."""
        if not self._ensure_connected():
            self.error.emit("not connected (click \U0001F4A1 to retry)")
            return
        try:
            self._controller.set_brightness(value)
            self.brightness_changed.emit(int(self._controller.brightness()))
        except Exception as e:  # noqa: BLE001
            logger.error(f"LED write failed: {e}")
            # No close/reopen here — that would risk hanging the CH340. If the
            # board was unplugged, re-plug it and press the button again.
            self.error.emit("write failed — re-plug the LED and press \U0001F4A1")

    def run(self) -> None:
        logger.info("LED thread starting")
        self._running = True
        self._controller = LEDController(self._config)

        if not self._ensure_connected():
            # Not fatal — the controls stay live; the next command retries.
            self.error.emit("not connected (click \U0001F4A1 to retry)")

        while self._running:
            try:
                cmd = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if cmd is self._QUIT:
                break
            kind, value = cmd
            if kind == 'set':
                self._apply(value)

        # Close the serial port cleanly. A graceful close is what keeps the
        # CH340 reopenable on the next launch (see the module docstring).
        if self._controller is not None:
            self._controller.disconnect()
        self.disconnected.emit()
        logger.info("LED thread stopped (port closed cleanly)")

    # --- GUI-thread API (fire-and-forget) ---

    def request_set_brightness(self, value: int) -> None:
        self._queue.put(('set', int(value)))

    def request_on(self, brightness: int = 255) -> None:
        self._queue.put(('set', int(brightness)))

    def request_off(self) -> None:
        self._queue.put(('set', 0))

    def stop(self) -> None:
        """Stop the worker thread and close the serial port gracefully (a clean
        close is what keeps this CH340 reopenable on the next launch)."""
        self._running = False
        self._queue.put(self._QUIT)
        self.wait(5000)
