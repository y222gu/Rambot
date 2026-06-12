"""
COB illumination LED strip controller (Arduino over serial).

The Arduino firmware accepts a brightness value 0-255 as an ASCII line
(``"<value>\\n"``); 0 turns the strip off, 255 is full brightness.

Adapted from the noodlepy ``LEDController`` with three changes for this system:

  * Safer port auto-detection — it only matches Arduino-class boards
    (CH340/FTDI/CP210x/genuine Arduino) and NEVER claims the Prusa stage port
    (VID 0x2C99 / "prusa"/"marlin" in the description).
  * Non-raising ``connect()`` — returns ``False`` instead of raising when no
    board is present, so the GUI still runs without the illuminator attached.
  * Clean shutdown — ``disconnect()`` purges, drops DTR/RTS and closes the
    port GRACEFULLY. A CH340 that is torn down by a hard process-exit handle
    release (instead of a real close) comes back in the Windows
    "device not functioning" (error 31) state, which is why the previous
    "never close" approach forced a physical re-plug on every launch. A proper
    close lets the driver release the port so the next launch reopens it.
"""

import subprocess
import sys
import time
from typing import Optional

import serial
import serial.tools.list_ports

from raman_control.config.hardware_config import LEDConfig
from raman_control.config.logging_config import get_logger

logger = get_logger('led')

# USB VIDs / description keywords for common Arduino-class USB-serial bridges.
_ARDUINO_VIDS = {
    0x2341,  # Arduino LLC
    0x2A03,  # Arduino SA (Genuino)
    0x1A86,  # QinHeng CH340/CH341
    0x0403,  # FTDI
    0x10C4,  # Silicon Labs CP210x
}
_ARDUINO_KEYWORDS = (
    'arduino', 'ch340', 'ch341', 'ftdi', 'cp210', 'usb serial', 'usb-serial',
)

# The Prusa stage is also a USB-serial device — never grab its port for the LED.
_PRUSA_VID = 0x2C99
_PRUSA_KEYWORDS = ('prusa', 'marlin')


class LEDController:
    """Controls the COB LED strip via an Arduino over serial (0-255)."""

    BRIGHTNESS_FULL = 255

    # A stuck CH340 sometimes refuses the first reconfigure with error 31 but
    # accepts it a moment later. A FEW gentle retries (not a tight loop) let it
    # self-recover without a physical re-plug.
    _OPEN_ATTEMPTS = 3
    _OPEN_RETRY_DELAY = 0.7  # seconds between open attempts

    def __init__(self, config: Optional[LEDConfig] = None):
        self.config = config or LEDConfig()
        self._serial: Optional[serial.Serial] = None
        self._is_on = False
        self._brightness = 0
        self.port: Optional[str] = None

    # ------------------------------------------------------------------
    # Port detection / connection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_prusa(p) -> bool:
        desc = (p.description or '').lower()
        return p.vid == _PRUSA_VID or any(k in desc for k in _PRUSA_KEYWORDS)

    def _auto_detect_port(self) -> Optional[str]:
        """Find an Arduino-class serial port, skipping the Prusa stage."""
        for p in serial.tools.list_ports.comports():
            if self._is_prusa(p):
                continue
            desc = (p.description or '').lower()
            if p.vid in _ARDUINO_VIDS or any(k in desc for k in _ARDUINO_KEYWORDS):
                logger.info(f"Auto-detected LED Arduino on {p.device}: "
                            f"{p.description}")
                return p.device
        logger.warning("No Arduino-class serial port found for the LED controller")
        return None

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def connect(self) -> bool:
        """Open the serial port. Returns True on success, False otherwise.

        ``serial.Serial(port, baud, timeout)`` asserts DTR on open, which pulses
        the Arduino's auto-reset so the board reboots into a clean state. If the
        first open fails with the CH340 "device not functioning" (error 31)
        state, we retry a FEW times with a short delay (not a tight loop) — that
        is usually enough for the driver to recover on its own, so a physical
        re-plug is no longer required.
        """
        if self.is_connected:
            return True
        port = self.config.port or self._auto_detect_port()
        if not port:
            return False

        if self._open_with_retries(port):
            return True

        # Persistent failure — the CH340 is most likely stuck in the Windows
        # "device not functioning" (error 31) state from a previous unclean
        # exit (e.g. a crash). Retrying the open can't clear that; the only
        # software way is to re-enumerate the USB device (the OS equivalent of
        # a physical re-plug). Try that once, then reopen.
        logger.warning(
            f"LED port {port} won't open, attempting to reset the device…")
        if self._attempt_port_recovery(port):
            time.sleep(2.0)  # let the device re-enumerate and re-create the port
            if self._open_with_retries(port, attempts=2):
                return True

        logger.error(
            f"Could not open LED port {port}. If this keeps happening, run the "
            f"app as administrator (so it can reset the device automatically) "
            f"or physically re-plug the LED once.")
        self._serial = None
        return False

    def _open_with_retries(self, port: str, attempts: Optional[int] = None) -> bool:
        """Try to open + configure the port a few times. Returns True on
        success (and sets self._serial), False if all attempts failed."""
        attempts = attempts or self._OPEN_ATTEMPTS
        last_err: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                # DTR asserted on open (pyserial default) -> resets the Arduino.
                ser = serial.Serial(
                    port, self.config.baudrate,
                    timeout=self.config.timeout,
                    write_timeout=self.config.timeout)
            except (serial.SerialException, OSError) as e:
                last_err = e
                logger.warning(
                    f"LED open attempt {attempt}/{attempts} on "
                    f"{port} failed: {e}")
                if attempt < attempts:
                    time.sleep(self._OPEN_RETRY_DELAY)
                continue

            self._serial = ser
            self.port = port
            # The board resets when the port opens — wait for the bootloader,
            # then clear any boot chatter from the input buffer.
            time.sleep(self.config.boot_delay)
            try:
                ser.reset_input_buffer()
            except Exception:  # noqa: BLE001
                pass
            logger.info(f"LED controller connected on {port}")
            return True

        if last_err is not None:
            logger.warning(
                f"LED port {port} did not open after {attempts} attempts: "
                f"{last_err}")
        return False

    def _attempt_port_recovery(self, port: str) -> bool:
        """Re-enumerate (disable + re-enable) the USB-serial device backing
        ``port`` — the software equivalent of a physical re-plug — to clear a
        CH340 stuck in the 'device not functioning' (error 31) state.

        Windows-only, best effort. ``Restart-PnpDevice`` usually needs
        administrator rights, so this is wrapped in try/except and simply
        returns False (with a hint logged) when it can't run."""
        if sys.platform != 'win32':
            return False
        # Plain (non-f) string so the PowerShell braces are literal; the port
        # is concatenated in.
        ps = (
            "$ErrorActionPreference='Stop'; "
            "$d = Get-PnpDevice -PresentOnly | "
            "Where-Object { $_.FriendlyName -like '*(" + port + ")*' } | "
            "Select-Object -First 1; "
            "if (-not $d) { throw 'COM device not found' }; "
            "Restart-PnpDevice -InstanceId $d.InstanceId -Confirm:$false"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                capture_output=True, text=True, timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode == 0:
                logger.info(f"Re-enumerated {port} (Restart-PnpDevice)")
                return True
            logger.warning(
                f"Could not auto-reset {port} (likely needs admin): "
                f"{(result.stderr or '').strip()[:200]}")
            return False
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Port recovery attempt for {port} failed: {e}")
            return False

    def disconnect(self) -> None:
        """Close the serial port GRACEFULLY so the CH340 stays reopenable.

        Turn the strip off, drop DTR/RTS, then run pyserial's normal close().
        A clean close is what lets the driver release the port — skipping it
        (or relying on a hard process-exit handle release) is what left this
        CH340 in the error-31 state that required a physical re-plug.
        """
        ser = self._serial
        self._serial = None
        self._is_on = False
        self._brightness = 0
        if ser is None:
            return
        try:
            if ser.is_open:
                # Best-effort: turn the strip off, and make sure the byte is
                # actually transmitted before we release the port.
                try:
                    ser.write(b'0\n')
                    ser.flush()              # block until the OFF byte is sent
                    time.sleep(0.05)
                except Exception:  # noqa: BLE001
                    pass
                # Clear any pending I/O so close() doesn't block / leave bytes
                # stuck in the driver.
                for reset in (ser.reset_output_buffer, ser.reset_input_buffer):
                    try:
                        reset()
                    except Exception:  # noqa: BLE001
                        pass
                # Drop the control lines and give the driver a moment to apply
                # the change before closing — this is what lets the CH340
                # release the port so the next launch can reopen it.
                try:
                    ser.dtr = False
                    ser.rts = False
                    time.sleep(0.05)
                except Exception:  # noqa: BLE001
                    pass
                ser.close()
            logger.info("LED controller disconnected (port closed cleanly)")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error while closing LED serial port: {e}")

    # ------------------------------------------------------------------
    # Brightness control
    # ------------------------------------------------------------------

    def set_brightness(self, value: int) -> None:
        """Set brightness 0-255 (0 = off, 255 = max); clamps out-of-range."""
        value = int(max(0, min(255, value)))
        self._brightness = value
        self._is_on = value > 0
        self._send(str(value))
        logger.debug(f"LED brightness set to {value}/255 "
                     f"({round(value / 255 * 100)}%)")

    def turn_on(self, brightness: int = 255) -> None:
        """Turn on at the given brightness (default full)."""
        self.set_brightness(brightness)

    def turn_off(self) -> None:
        """Turn the LED strip off."""
        self.set_brightness(0)

    def brightness(self) -> int:
        return int(self._brightness)

    def is_on(self) -> bool:
        return bool(self._is_on)

    def _send(self, cmd: str) -> None:
        if not self.is_connected:
            raise RuntimeError("LED serial port is not open")
        self._serial.write((cmd + '\n').encode())

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()
