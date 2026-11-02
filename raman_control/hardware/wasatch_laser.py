"""
Laser-only control of the Wasatch Photonics unit.

Used when another spectrometer (the Andor Kymera) collects the Raman signal
but the Wasatch spectrometer-laser unit still provides the 785 nm excitation.
It opens the same USB device as :class:`WasatchSpectrometer` but only ever
calls the laser functions; it never reads a spectrum.

Threading: one object, one owner thread (see
:class:`raman_control.gui.threads.laser_thread.LaserThread`). The two Wasatch
classes must never be connected at the same time in one process, since both
would claim the same USB interface; ``SpectrumThread`` guarantees that by only
starting the laser thread when its active backend has no laser of its own.
"""

from __future__ import annotations

from typing import Optional

from raman_control.config.hardware_config import SpectrometerConfig
from raman_control.config.logging_config import get_logger
from raman_control.hardware.spectrometer_base import LaserController
from raman_control.hardware.wasatch_spectrometer import open_wasatch_device

logger = get_logger('wasatch_laser')


class WasatchLaser(LaserController):
    """Excitation laser of the Wasatch unit, with the spectrometer unused."""

    def __init__(self, config: Optional[SpectrometerConfig] = None):
        self._config = config or SpectrometerConfig()
        self._device = None
        self._fid = None
        self._connected = False
        self._serial = ""

    def connect(self, quiet: bool = False) -> bool:
        """Open the Wasatch unit for laser control. ``quiet`` is used by the
        laser thread's background retries so an absent unit does not spam the
        log."""
        try:
            self._device, self._fid = open_wasatch_device(logger, quiet=quiet)
        except ImportError:
            logger.error("Wasatch SDK (wasatch package) not installed; "
                         "the excitation laser cannot be controlled")
            return False
        except Exception as e:  # noqa: BLE001
            logger.error(f"Wasatch laser connection error: {e}")
            return False
        if self._device is None:
            return False
        try:
            try:
                self._serial = str(self._device.settings.eeprom.serial_number or "")
            except Exception:  # noqa: BLE001
                self._serial = ""
            self._fid.set_laser_power_high_resolution(True)
            self._fid.set_laser_power_mW(self._config.laser_power_mw)
            # Always start with the laser off; the user / a run turns it on.
            self._fid.set_laser_enable(False)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Wasatch laser initial configuration failed: {e}")
            self.disconnect()
            return False
        self._connected = True
        logger.info(f"Wasatch laser ready (unit {self._serial or 'unknown'}), "
                    f"power {self._config.laser_power_mw} mW")
        return True

    def disconnect(self) -> None:
        if self._fid is not None:
            try:
                self._fid.set_laser_enable(False)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to disable laser during disconnect: {e}")
        if self._device is not None:
            try:
                self._device.disconnect()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Wasatch laser disconnect error: {e}")
        self._fid = None
        self._device = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def serial(self) -> str:
        return self._serial

    # --- LaserController ----------------------------------------------------

    @property
    def laser_available(self) -> bool:
        return self._connected and self._fid is not None

    def set_laser_enable(self, on: bool) -> bool:
        if self._fid is None:
            return False
        self._fid.set_laser_enable(bool(on))
        return True

    def set_laser_power_mw(self, mw: float) -> bool:
        if self._fid is None:
            return False
        self._fid.set_laser_power_mW(mw)
        return True

    def is_laser_on(self) -> Optional[bool]:
        """Ask the unit whether the laser is firing (None if unknown). Used as
        a periodic health check of the USB link."""
        if self._fid is None:
            return None
        try:
            resp = self._fid.get_laser_enabled()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Laser status read failed: {e}")
            return None
        data = getattr(resp, 'data', resp)
        return None if data is None else bool(data)
