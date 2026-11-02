"""
Qt thread that owns the Wasatch excitation laser when the Wasatch is NOT the
active spectrometer (i.e. the Kymera collects the spectra).

The thread opens the Wasatch unit through :class:`WasatchLaser` and services a
command queue: laser on/off and laser power. Callers (the spectrum thread, the
GUI) never touch the USB device; they post commands and return immediately.
A periodic health check reads the laser state so a lost USB link is noticed
and the laser is reported unavailable; the thread then retries the connection
in the background.

Started and stopped by :class:`SpectrumThread`, which forwards its public
``set_laser`` / ``set_laser_power_mw`` calls here, so the rest of the
application does not know (or care) which device hosts the laser.
"""

from __future__ import annotations

import queue
import time
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from raman_control.config.hardware_config import SpectrometerConfig
from raman_control.config.logging_config import get_logger
from raman_control.hardware.wasatch_laser import WasatchLaser

logger = get_logger('laser_thread')


class LaserThread(QThread):
    """Owner thread of the Wasatch laser (laser-only mode).

    Signals:
        available_changed(bool): laser hardware reachable / lost
        laser_state_changed(bool): the laser was switched on/off (applied)
        error(str)
    """

    available_changed = pyqtSignal(bool)
    laser_state_changed = pyqtSignal(bool)
    error = pyqtSignal(str)

    _RECONNECT_INTERVAL_S = 10.0
    _HEALTH_INTERVAL_S = 5.0

    def __init__(self, config: Optional[SpectrometerConfig] = None, parent=None,
                 laser: Optional[WasatchLaser] = None):
        super().__init__(parent)
        self._config = config or SpectrometerConfig()
        self._laser = laser or WasatchLaser(self._config)
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._running = False
        self._available = False
        self._laser_on = False
        self._power_mw = float(self._config.laser_power_mw)

    # --- public API (any thread) ----------------------------------------------

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def is_laser_on(self) -> bool:
        return self._laser_on

    def request_enable(self, on: bool) -> None:
        self._queue.put(('enable', bool(on)))

    def request_power_mw(self, mw: float) -> None:
        self._queue.put(('power', float(mw)))

    def stop(self, wait_ms: int = 5000) -> None:
        """Turn the laser off, close the device and end the thread."""
        if not self.isRunning():
            return
        self._running = False
        self._queue.put(('stop', None))
        self.wait(wait_ms)

    # --- thread body ----------------------------------------------------------

    def run(self):
        logger.info("Laser thread starting (Wasatch laser-only mode)")
        self._running = True
        self._set_available(self._try_connect())
        last_attempt = time.monotonic()
        last_health = time.monotonic()

        while self._running:
            try:
                cmd, arg = self._queue.get(timeout=0.2)
            except queue.Empty:
                cmd, arg = None, None

            if cmd == 'stop':
                break
            if cmd is not None:
                self._handle(cmd, arg)

            now = time.monotonic()
            if not self._available and now - last_attempt >= self._RECONNECT_INTERVAL_S:
                last_attempt = now
                if self._try_connect(quiet=True):
                    self._set_available(True)
                    # Re-apply the wanted state after a reconnect.
                    self._handle('power', self._power_mw)
                    if self._laser_on:
                        self._handle('enable', True)
            elif self._available and now - last_health >= self._HEALTH_INTERVAL_S:
                last_health = now
                self._health_check()

        self._shutdown()
        logger.info("Laser thread stopped")

    # --- internals (laser thread only) ----------------------------------------

    def _try_connect(self, quiet: bool = False) -> bool:
        try:
            try:
                return self._laser.connect(quiet=quiet)
            except TypeError:  # a laser object without the quiet flag (tests)
                return self._laser.connect()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Laser connect failed: {e}")
            return False

    def _set_available(self, available: bool) -> None:
        if available != self._available:
            self._available = available
            self.available_changed.emit(available)
            if not available:
                self.error.emit("Excitation laser (Wasatch unit) not available")

    def _handle(self, cmd: str, arg) -> None:
        if cmd == 'power':
            self._power_mw = float(arg)
            if self._available:
                self._guard(lambda: self._laser.set_laser_power_mw(self._power_mw),
                            f"set laser power {self._power_mw:.0f} mW")
        elif cmd == 'enable':
            want = bool(arg)
            self._laser_on = want
            if self._available:
                ok = self._guard(lambda: self._laser.set_laser_enable(want),
                                 f"laser {'ON' if want else 'OFF'}")
                if ok:
                    self.laser_state_changed.emit(want)
            elif want:
                self.error.emit("Cannot turn the laser on: Wasatch unit not found")
                self.laser_state_changed.emit(False)
                self._laser_on = False

    def _guard(self, fn, what: str) -> bool:
        try:
            ok = bool(fn())
        except Exception as e:  # noqa: BLE001
            logger.error(f"Laser command failed ({what}): {e}")
            ok = False
        if not ok:
            self._lose_link()
        else:
            logger.info(f"Laser: {what}")
        return ok

    def _health_check(self) -> None:
        state = self._laser.is_laser_on()
        if state is None:
            self._lose_link()
        elif state != self._laser_on:
            # The unit disagrees with us (interlock, key switch...). Trust it.
            logger.warning(f"Laser state from unit = {state}, expected {self._laser_on}")
            self._laser_on = state
            self.laser_state_changed.emit(state)

    def _lose_link(self) -> None:
        logger.error("Lost the Wasatch laser link; will retry")
        try:
            self._laser.disconnect()
        except Exception:  # noqa: BLE001
            pass
        self._set_available(False)

    def _shutdown(self) -> None:
        try:
            if self._available:
                self._laser.set_laser_enable(False)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._laser.disconnect()
        except Exception:  # noqa: BLE001
            pass
        self._laser_on = False
        self._set_available(False)
