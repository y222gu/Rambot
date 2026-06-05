import serial
import serial.tools.list_ports
import time
import logging

logger = logging.getLogger(__name__)


class LEDController:
    """
    Controls the COB LED strip via Arduino over serial.
    Supports on/off and variable brightness (0-255).
    """

    BRIGHTNESS_FULL     = 255   # 100%

    def __init__(self, port: str | None = None, baud: int = 9600, timeout: float = 2.0):
        self.port = port or self._auto_detect_port()
        self.baud = baud
        self._serial = None
        self._is_on = False
        self._brightness = 0
        self.connect(timeout)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _auto_detect_port(self) -> str:
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if any(kw in (p.description or "").lower()
                   for kw in ["arduino", "ch340", "ftdi", "usb serial"]):
                logger.info(f"Auto-detected Arduino on {p.device}")
                return p.device
        if ports:
            logger.warning(f"Could not confirm Arduino. Using first port: {ports[0].device}")
            return ports[0].device
        raise RuntimeError("No serial ports found. Is the Arduino plugged in?")

    def connect(self, timeout: float = 2.0):
        try:
            self._serial = serial.Serial(self.port, self.baud, timeout=timeout)
            # Arduino resets on serial connect — wait for it to boot
            time.sleep(2)
            logger.info(f"LED controller connected on {self.port}")
        except serial.SerialException as e:
            raise RuntimeError(f"Failed to connect to Arduino: {e}")

    def disconnect(self):
        self.turn_off()
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.info("LED controller disconnected")

    
    # main brightness control
    def set_brightness(self, value: int):
        """
        Sets brightness directly, values between 0-255, 0 = off, 255 = max brightness.
        """
        if not 0 <= value <= 255:
            raise ValueError(f"Brightness must be 0-255, got {value}")
        self._brightness = value
        self._is_on = value > 0
        self._send(str(value))
        logger.info(f"LED brightness set to {value}/255 ({round(value/255*100)}%)")

    def turn_on(self, brightness: int = 255):
        """Turn on at specified brightness, default is full"""
        self.set_brightness(brightness)

    def turn_off(self):
        """Turns LED strip off"""
        self.set_brightness(0)

    def brightness(self) -> int:
        return int(self._brightness)

    def is_on(self) -> bool:
        return bool(self._is_on)

    def _send(self, cmd: str):
        if not self._serial or not self._serial.is_open:
            raise RuntimeError("Serial port is not open")
        self._serial.write((cmd + '\n').encode())

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()

# integration for main

def on_calibration_complete(led: LEDController):
    """
    Light turns on after calibration so CV can detect sample edges.
    Uses full brightness by default — adjust if camera is overexposing.
    """
    logger.info("Calibration complete — enabling LED")
    led.turn_on(LEDController.BRIGHTNESS_FULL)


def on_edge_detected(led: LEDController):
    """
    Light turns off once edge is captured, before laser fires.
    """
    logger.info("Edge detected — disabling LED for laser")
    led.turn_off()


def on_scan_complete(led: LEDController):
    """Safety fallback — always off between scan positions."""
    if led.is_on():
        logger.warning("Scan ended with LED on — forcing off")
        led.turn_off()


def led_on(led: LEDController):
    led.turn_on()


def led_off(led: LEDController):
    led.turn_off()
