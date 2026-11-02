"""
Wasatch Photonics spectrometer backend (and its built-in excitation laser).

This is the hardware code that used to live inside ``SpectrumThread``,
unchanged in behaviour: ``WasatchBus`` -> ``WasatchDevice`` ->
``RealUSBDevice``, ``get_line()`` for a blocking exposure, and the
``FeatureIdentificationDevice`` (``fid``) setters for integration time and
laser control. Everything here runs on the spectrum thread only; the USB
device must never be touched from two threads (concurrent access causes
libusb timeouts).

The laser lives on the same USB device, so this backend is also the
:class:`LaserController` whenever the Wasatch is the active spectrometer.
When another spectrometer (the Kymera) is active, the laser is driven instead
by :class:`~raman_control.hardware.wasatch_laser.WasatchLaser`, which opens
the same device but only ever issues laser commands.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from raman_control.config.hardware_config import SpectrometerConfig
from raman_control.config.logging_config import get_logger
from raman_control.hardware.spectrometer_base import (
    BACKEND_WASATCH, LaserController, SpectrometerBackend, SpectrometerStatus,
)

logger = get_logger('wasatch_spectrometer')


#: USB vendor id of Wasatch Photonics units. The wasatch package's bus scan
#: also lists Ocean Optics and ANDOR cameras (its ENLIGHTEN app supports
#: them), so without this filter the Andor Newton would be mistaken for the
#: Wasatch unit.
WASATCH_VID = 0x24AA


def open_wasatch_device(log=logger, quiet: bool = False):
    """Find the first Wasatch Photonics unit on the USB bus and connect to it.

    Returns ``(device, fid)`` or ``(None, None)`` when no unit is present or
    the connection fails. Raises ``ImportError`` if the ``wasatch`` package is
    missing. Shared by the spectrometer backend and the laser-only controller
    so both open the hardware the same, proven way. ``quiet`` lowers the
    "not found" message to debug level (background retries).
    """
    from wasatch.WasatchBus import WasatchBus
    from wasatch.WasatchDevice import WasatchDevice
    from wasatch.RealUSBDevice import RealUSBDevice

    log.debug("Searching for Wasatch spectrometer...")
    bus = WasatchBus(use_sim=False)
    ids = [d for d in bus.device_ids
           if getattr(d, 'vid', None) in (None, WASATCH_VID)]
    skipped = len(bus.device_ids) - len(ids)
    if skipped:
        log.debug(f"Ignoring {skipped} non-Wasatch device(s) listed by the "
                  f"wasatch bus scan")
    if not ids:
        (log.debug if quiet else log.warning)(
            "No Wasatch spectrometer found on the USB bus")
        return None, None

    device_id = ids[0]
    log.info(f"Found Wasatch unit: {device_id}")
    device_id.device_type = RealUSBDevice(device_id)
    device = WasatchDevice(device_id)
    resp = device.connect()
    # wasatch >= 2.3 returns a SpectrometerResponse (always truthy); older
    # versions returned a bool.
    ok = bool(getattr(resp, 'data', resp))
    if not ok or device.hardware is None:
        log.error(f"Wasatch connect() failed: "
                  f"{getattr(resp, 'error_msg', None) or 'no hardware handle'}")
        try:
            device.disconnect()
        except Exception:  # noqa: BLE001
            pass
        return None, None
    return device, device.hardware


class WasatchSpectrometer(SpectrometerBackend, LaserController):
    """Wasatch spectrometer + laser on one USB device."""

    name = BACKEND_WASATCH
    model = "Wasatch Photonics"

    def __init__(self, config: SpectrometerConfig):
        self._config = config
        self._device = None
        self._fid = None  # FID = Feature Identification Device
        self._connected = False
        self._wavelengths_nm: Optional[np.ndarray] = None
        self._wavenumbers: Optional[np.ndarray] = None
        self._pixels = 0
        self._serial = ""

    # --- SpectrometerBackend ------------------------------------------------

    def connect(self) -> bool:
        try:
            self._device, self._fid = open_wasatch_device(logger)
        except ImportError:
            logger.error("Wasatch SDK (wasatch package) not installed")
            return False
        except Exception as e:  # noqa: BLE001
            logger.error(f"Wasatch connection error: {e}")
            return False
        if self._device is None:
            return False

        try:
            settings = self._device.settings
            try:
                self._serial = str(settings.eeprom.serial_number or "")
            except Exception:  # noqa: BLE001
                self._serial = ""

            wl = getattr(settings, 'wavelengths', None)
            wn = getattr(settings, 'wavenumbers', None)
            if wl is not None and len(wl) > 0:
                self._wavelengths_nm = np.array(wl, dtype=np.float64)
                logger.info(f"Wavelength range: {self._wavelengths_nm[0]:.1f} - "
                            f"{self._wavelengths_nm[-1]:.1f} nm "
                            f"({len(self._wavelengths_nm)} pixels)")
            if wn is not None and len(wn) > 0:
                # The device's own Raman-shift calibration (uses its laser line).
                self._wavenumbers = np.array(wn, dtype=np.float64)
                logger.info(f"Wavenumber range: {self._wavenumbers[0]:.1f} - "
                            f"{self._wavenumbers[-1]:.1f} cm^-1")
            self._pixels = (len(self._wavelengths_nm)
                            if self._wavelengths_nm is not None
                            else len(self._wavenumbers)
                            if self._wavenumbers is not None else 0)
            if self._pixels == 0:
                logger.warning("No wavelength/wavenumber calibration from the "
                               "Wasatch device; using pixel indices")
                self._pixels = int(getattr(settings.eeprom, 'active_pixels_horizontal',
                                           0) or 2048)

            # Initial settings (same as before the refactor).
            self._fid.set_integration_time_ms(self._config.integration_time)
            self._fid.set_laser_power_high_resolution(True)
            self._fid.set_laser_power_mW(self._config.laser_power_mw)
            logger.info(f"Integration time: {self._config.integration_time} ms, "
                        f"Laser power: {self._config.laser_power_mw} mW")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Wasatch initial configuration failed: {e}")
            self.disconnect()
            return False

        self._connected = True
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
                logger.warning(f"Wasatch disconnect error: {e}")
        self._fid = None
        self._device = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def wavelengths_nm(self) -> Optional[np.ndarray]:
        return None if self._wavelengths_nm is None else self._wavelengths_nm.copy()

    @property
    def device_wavenumbers(self) -> Optional[np.ndarray]:
        """The unit's own Raman-shift axis (cm⁻¹), if it provided one."""
        return None if self._wavenumbers is None else self._wavenumbers.copy()

    @property
    def pixels(self) -> int:
        return self._pixels

    def set_integration_time_ms(self, ms: float) -> None:
        if self._fid is None:
            return
        self._fid.set_integration_time_ms(ms)

    def acquire(self) -> Optional[np.ndarray]:
        if self._fid is None:
            return None
        # get_line() blocks for the integration time.
        response = self._fid.get_line()
        if response is None:
            logger.warning("get_line() returned None")
            return None
        # Response layouts seen across wasatch versions.
        if hasattr(response, 'data') and hasattr(response.data, 'spectrum'):
            return np.array(response.data.spectrum, dtype=np.float32)
        if hasattr(response, 'data') and isinstance(response.data, (list, np.ndarray)):
            return np.array(response.data, dtype=np.float32)
        if isinstance(response, (list, np.ndarray)):
            return np.array(response, dtype=np.float32)
        logger.warning(f"Unexpected get_line() response type: {type(response)}")
        return None

    def status(self) -> SpectrometerStatus:
        return SpectrometerStatus(model=self.model, serial=self._serial,
                                  pixels=self._pixels)

    # --- LaserController ----------------------------------------------------

    @property
    def has_laser(self) -> bool:
        return True

    @property
    def laser_available(self) -> bool:
        return self._fid is not None

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
