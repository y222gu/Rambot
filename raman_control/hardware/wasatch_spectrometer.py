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

2D readout ("area scan")
------------------------
The detector behind a Wasatch unit is a 2D sensor (the lab's WP-785 has a
Hamamatsu S7031-1006 CCD, 1024 x 64 active pixels; Series-XS units have a
Sony IMX CMOS). Normally the FPGA bins the rows of its vertical ROI into the
one spectrum ``get_line()`` returns. When the FPGA advertises
``has_area_scan`` (``SpectrometerSettings.fpga_options``), ``SET_AREA_SCAN_
ENABLE`` (opcode 0xeb) switches the unit to stream the sensor row by row
instead after each ACQUIRE (opcode 0xad). Newer firmware tags each row with
a start-of-line marker (0xffff) and its row index; the lab unit's FPGA
(030-107) streams 69 rows of plain pixel data in order, without either. The
driver's own ``fid.get_area_scan()`` only accepts marked rows and assumes
exactly 69, so :meth:`WasatchSpectrometer._read_area_scan_frame` reads the
rows itself: ACQUIRE, then bulk reads until the pipe runs dry. One frame
takes roughly ``integration + 0.3 s`` on Hamamatsu units. This is the same
feature ENLIGHTEN's "Area Scan" view uses.

:meth:`WasatchSpectrometer.acquire_image` enables the area scan (once) and
returns the frame as ``(rows, pixels)``; the next :meth:`acquire` disables it
again before reading a spectrum, so the two readouts are never mixed in one
exposure. The rows binned into the spectrum are the unit's vertical ROI
(EEPROM ``roi_vertical_region_1_start/end``, 0-based inclusive); they can
only be changed where the firmware supports ``set_vertical_roi`` (XS, or a
microcontroller >= 10.0.0.47 on Hamamatsu units). The lab unit (firmware
10.0.0.10) cannot, so :attr:`supports_bin_rows` is False there and the GUI
shows the factory rows as fixed lines.

Detector temperature
--------------------
Cooled units (EEPROM ``has_cooling``; the lab's WP-785 has a TEC behind its
S7031 CCD) report the detector temperature over a control transfer
(``GET_CCD_TEMP``, opcode 0xd7). Wasatch.PY itself writes the EEPROM start-up
set point to the TEC and switches it on while connecting (``post_connect``;
-15 C on the lab unit), so nothing has to be configured here.
:meth:`WasatchSpectrometer.poll`, called by the spectrum thread about once a
second between exposures, reads the temperature into
:class:`SpectrometerStatus` exactly like the Kymera backend does for the
Newton CCD: the status bar shows it next to the spectrometer name, with
"(cooling)" until it is within ``TEC_STABLE_BAND_C`` of the set point, and
capture metadata records it. Uncooled units leave the temperature ``None``.
"""

from __future__ import annotations

import time
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

#: Start-of-line marker the FPGA writes into pixel 0 of every area-scan row.
AREA_SCAN_MARKER = 0xFFFF

#: Consecutive empty / failed frames after which the area scan is given up
#: (each failed frame costs one bulk-read timeout, ~2 x integration + 1 s).
AREA_SCAN_MAX_FAILURES = 3

#: After the first row of a frame arrived, a bulk read that stays empty this
#: long means the unit has streamed its last row (rows follow each other in
#: a few ms). Paid once per frame.
AREA_SCAN_DRAIN_TIMEOUT_MS = 300

#: Timeout of the bulk reads that clear stale rows from the pipe BEFORE each
#: ACQUIRE (a previous frame that broke off mid-way leaves rows behind).
AREA_SCAN_STALE_TIMEOUT_MS = 50
# Detector counts as "at temperature" within this band around the TEC set point.
TEC_STABLE_BAND_C = 1.0
# Minimum spacing of detector temperature reads (the thread polls about 1/s).
TEMPERATURE_POLL_S = 0.5


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


def _response_failed(resp) -> Optional[str]:
    """Error text if a wasatch ``SpectrometerResponse`` reports a failure
    (``error_msg`` set, poison pill, or ``data is False`` for "not
    connected"), else None. A bare bool / None from an old driver passes."""
    if resp is None or isinstance(resp, bool):
        return None if resp is not False else "command refused"
    msg = getattr(resp, 'error_msg', '') or ''
    if msg:
        return str(msg)
    if getattr(resp, 'poison_pill', False):
        return "USB control transfer failed"
    if getattr(resp, 'data', None) is False:
        return "device not connected"
    return None


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
        # 2D readout (see module docstring). Probed once in connect().
        self._detector = ""
        self._micro_fw = ""
        self._fpga_fw = ""
        self._invert_x = False
        self._area_scan_supported = False   # FPGA advertises area scan
        self._area_scan_on = False          # unit currently streams rows
        self._area_scan_failures = 0
        self._left_area_scan = False        # next spectrum follows an area scan
        self._rows = 0                      # rows of the area-scan frame
        self._rows_settable = False         # firmware takes set_vertical_roi
        self._roi = (0, 0)                  # vertical ROI, 0-based inclusive
        # Detector TEC (cooled units only; see module docstring).
        self._has_cooling = False
        self._temperature_c: Optional[float] = None
        self._tec_setpoint_c: Optional[float] = None
        self._tec_enabled: Optional[bool] = None
        self._last_temp_poll = 0.0
        self._temp_read_failed = False

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
            try:
                unit_model = str(getattr(settings.eeprom, 'model', '') or '').strip()
                if unit_model:
                    self.model = f"Wasatch Photonics {unit_model}"
            except Exception:  # noqa: BLE001
                pass

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

            self._probe_detector(settings)
            self._probe_cooling(settings)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Wasatch initial configuration failed: {e}")
            self.disconnect()
            return False

        self._connected = True
        return True

    def disconnect(self) -> None:
        if self._fid is not None:
            if self._area_scan_on:
                try:
                    self._set_area_scan(False)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to leave area scan during disconnect: {e}")
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
        self._area_scan_on = False
        self._left_area_scan = False
        self._temperature_c = None
        self._tec_enabled = None

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
        # A spectrum is the FPGA's binned readout: leave the row-streaming
        # area scan first (the GUI's Spectrograph mode may have turned it on).
        if self._area_scan_on and not self._set_area_scan(False):
            return None
        spectrum = self._read_spectrum()
        if spectrum is not None and self._left_area_scan:
            self._left_area_scan = False
            if len(spectrum) > 1 and spectrum[0] >= AREA_SCAN_MARKER:
                # A row that was still in the USB pipe, not a spectrum.
                logger.warning("Discarding one readout that still carried the "
                               "area-scan line marker")
                spectrum = self._read_spectrum()
        return spectrum

    def _read_spectrum(self) -> Optional[np.ndarray]:
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

    # --- 2D readout: area scan ------------------------------------------------

    def _probe_detector(self, settings) -> None:
        """Record, once after connect, what the unit's sensor and firmware
        allow: whole-sensor readout (area scan) and a settable vertical ROI."""
        eeprom = settings.eeprom
        self._detector = str(getattr(eeprom, 'detector', '') or '').strip()
        self._invert_x = bool(getattr(eeprom, 'invert_x_axis', False))
        self._micro_fw = str(getattr(settings, 'microcontroller_firmware_version', '') or '')
        self._fpga_fw = str(getattr(settings, 'fpga_firmware_version', '') or '')

        fpga = getattr(settings, 'fpga_options', None)
        has_area_scan = bool(getattr(fpga, 'has_area_scan', False))
        try:
            ingaas = bool(settings.is_ingaas())   # single-line array: no rows
        except Exception:  # noqa: BLE001
            ingaas = False
        self._area_scan_supported = (has_area_scan and not ingaas
                                     and hasattr(self._fid, 'set_area_scan_enable')
                                     and hasattr(self._fid, '_send_code')
                                     and getattr(self._fid, 'device_type', None) is not None)
        self._area_scan_failures = 0

        # The driver's frame buffer is the sensor height it expects to stream
        # (70 rows on Hamamatsu units, 1080 on XS); the EEPROM's active rows
        # are the fallback.
        frame = getattr(self._fid, 'area_scan_frame', None)
        rows = len(frame) if frame else 0
        if rows <= 0:
            rows = int(getattr(eeprom, 'actual_pixels_vertical', 0)
                       or getattr(eeprom, 'active_pixels_vertical', 0) or 0)
        self._rows = rows

        # Vertical ROI = the rows the FPGA bins into the spectrum.
        self._roi = self._read_roi(eeprom)
        try:
            self._rows_settable = bool(settings.is_xs()) or bool(
                settings.supports_feature('hamamatsu_vertical_roi'))
        except Exception:  # noqa: BLE001
            self._rows_settable = False
        if not hasattr(self._fid, 'set_vertical_roi'):
            self._rows_settable = False

        start, end = self.bin_rows
        logger.info(f"Wasatch detector {self._detector or '?'}: "
                    f"{self._pixels} px x {self._rows} rows, firmware "
                    f"micro {self._micro_fw or '?'} / FPGA {self._fpga_fw or '?'}, "
                    f"area scan {'supported' if self._area_scan_supported else 'NOT supported'}, "
                    f"bin rows {start}-{end} "
                    f"({'settable' if self._rows_settable else 'fixed by firmware'})")

        # Configured rows (Settings > Spectrometer > Wasatch): 0/0 = factory.
        w = self._config.wasatch
        if int(w.bin_row_start) > 0 or int(w.bin_row_end) > 0:
            if self._rows_settable:
                self.set_bin_rows(int(w.bin_row_start), int(w.bin_row_end))
            else:
                logger.warning(f"Configured Wasatch bin rows {w.bin_row_start}-"
                               f"{w.bin_row_end} ignored: this unit's firmware "
                               f"does not allow changing the vertical ROI "
                               f"(factory rows {start}-{end} are used)")

    @staticmethod
    def _read_roi(eeprom) -> tuple:
        try:
            return (int(eeprom.roi_vertical_region_1_start),
                    int(eeprom.roi_vertical_region_1_end))
        except Exception:  # noqa: BLE001
            return (0, 0)

    @property
    def supports_image_readout(self) -> bool:
        return self._connected and self._area_scan_supported

    @property
    def detector_rows(self) -> int:
        return self._rows if self._connected else 0

    @property
    def supports_bin_rows(self) -> bool:
        return self.supports_image_readout and self._rows_settable

    @property
    def bin_rows(self) -> tuple:
        """Rows the FPGA bins into the spectrum, 1-based inclusive, from the
        unit's vertical ROI (0-based inclusive). An unusable ROI (end before
        start, or outside the sensor) reads as the whole sensor."""
        rows = self._rows
        if rows <= 0:
            return (0, 0)
        start, end = self._roi
        if end < start or start < 0 or end >= rows or (start, end) == (0, 0):
            return (1, rows)
        return (start + 1, end + 1)

    def _clamp_rows(self, start: int, end: int) -> tuple:
        rows = self._rows
        start, end = int(start), int(end)
        if rows <= 0:
            return (max(1, start), max(1, end))
        if end <= 0 or end > rows:
            end = rows
        start = min(max(1, start), rows)
        if start > end:
            start, end = end, start
        return (start, end)

    def set_bin_rows(self, start: int, end: int) -> None:
        """Change the vertical ROI (rows binned into the spectrum), 1-based
        inclusive, ``end`` 0 = last row. Only possible where the firmware
        supports it (:attr:`supports_bin_rows`); otherwise the request is
        recorded in the config but the factory rows stay in force."""
        start, end = self._clamp_rows(start, end)
        self._config.wasatch.bin_row_start = start
        self._config.wasatch.bin_row_end = end
        if self._fid is None:
            return
        if not self._rows_settable:
            cur = self.bin_rows
            logger.warning(f"Wasatch bin rows {start}-{end} not applied: the "
                           f"unit's firmware fixes the vertical ROI "
                           f"(rows {cur[0]}-{cur[1]})")
            return
        # The driver takes 0-based lines and insists on stop > start, so a
        # single row is sent as (row, row + 1).
        lo, hi = start - 1, end - 1
        if hi <= lo:
            hi = lo + 1
        try:
            resp = self._fid.set_vertical_roi((lo, hi))
        except Exception as e:  # noqa: BLE001
            logger.error(f"set_vertical_roi({lo}, {hi}) failed: {e}")
            return
        err = _response_failed(resp)
        if err:
            logger.warning(f"Wasatch refused vertical ROI {lo}-{hi}: {err}")
            return
        self._roi = (lo, hi)
        logger.info(f"Wasatch bin rows set to {start}-{end} "
                    f"(vertical ROI lines {lo}-{hi})")

    def _set_area_scan(self, on: bool) -> bool:
        """Switch the unit between the binned spectrum and streaming the
        sensor row by row. Idempotent; returns False if the unit refused."""
        if self._fid is None:
            return False
        if self._area_scan_on == on:
            return True
        try:
            resp = self._fid.set_area_scan_enable(bool(on))
        except Exception as e:  # noqa: BLE001
            logger.error(f"set_area_scan_enable({on}) failed: {e}")
            return False
        err = _response_failed(resp)
        if err:
            logger.error(f"Wasatch refused area scan {'on' if on else 'off'}: {err}")
            if on:
                self._area_scan_supported = False
            return False
        self._area_scan_on = on
        # Drop any partial row left over from the other readout.
        if hasattr(self._fid, 'extra_area_scan_data'):
            self._fid.extra_area_scan_data = []
        if on and hasattr(self._fid, 'reset_area_scan_frame'):
            self._fid.reset_area_scan_frame()
        self._left_area_scan = not on
        logger.info(f"Wasatch area scan {'enabled' if on else 'disabled'}")
        return True

    @staticmethod
    def _is_timeout(exc: BaseException) -> bool:
        """pyusb's libusb0 backend raises a plain USBError for timeouts, so
        classify by message as well as by type."""
        name = type(exc).__name__
        return 'Timeout' in name or 'timeout' in str(exc).lower() \
            or 'timed out' in str(exc).lower()

    def _read_bulk_line(self, line_len: int, endpoints: list, timeout_ms: int) -> bytearray:
        """Read exactly one ``line_len``-byte row from the bulk endpoint(s),
        gathering partial transfers. Raises what pyusb raises (a timeout
        before the first byte means "no more rows")."""
        fid = self._fid
        data = bytearray()
        while len(data) < line_len:
            remaining = line_len - len(data)
            # 2048-px FX2 units deliver 1024 px per endpoint (0x82 then 0x86).
            ep = endpoints[1] if len(endpoints) > 1 and len(data) >= line_len // 2 else endpoints[0]
            chunk = fid.device_type.read(fid.device, ep, remaining, timeout=timeout_ms)
            if chunk is None or len(chunk) == 0:
                raise TimeoutError("empty bulk read")
            data.extend(bytes(chunk))
        return data

    def _read_area_scan_frame(self) -> Optional[np.ndarray]:
        """Trigger ONE exposure in area-scan mode and read every row the unit
        streams, as (rows, pixels) float32.

        This replaces the driver's ``get_area_scan_hamamatsu()``, which only
        accepts rows that start with a 0xffff marker followed by the row index
        and assumes exactly 69 of them. The lab unit's FPGA (030-107) streams
        69 rows of plain pixel data without marker or index, which the driver
        drops wholesale, and any row count other than 69 leaves bytes in the
        pipe that break the next ACQUIRE. Here rows are read until the pipe
        runs dry (a short timeout after the last row), the marker / index are
        honoured when present and arrival order is used otherwise.
        """
        fid = self._fid
        try:
            pixels = int(fid.settings.pixels()) or self._pixels
        except Exception:  # noqa: BLE001
            pixels = self._pixels
        if pixels <= 0:
            return None
        line_len = pixels * 2
        endpoints = [0x82]
        try:
            if pixels == 2048 and not fid.settings.is_arm():
                endpoints = [0x82, 0x86]
        except Exception:  # noqa: BLE001
            pass
        try:
            first_timeout = int(fid.generate_timeout_ms()) + 10
        except Exception:  # noqa: BLE001
            first_timeout = int(2 * self._config.integration_time + 1000)
        drain_timeout = AREA_SCAN_DRAIN_TIMEOUT_MS
        max_rows = max(self._rows, 64) * 2 + 16   # safety cap only

        # Rows a broken-off previous frame left in the pipe would be read as
        # part of this one (and an ACQUIRE on top of them upset the FX2 on the
        # lab unit): drop them first. Costs one short timeout per frame.
        stale = 0
        for _ in range(max_rows):
            try:
                stale += len(self._read_bulk_line(line_len, endpoints,
                                                  AREA_SCAN_STALE_TIMEOUT_MS))
            except Exception as e:  # noqa: BLE001
                if not self._is_timeout(e):
                    logger.debug(f"Wasatch area scan: draining stale rows: {e}")
                break
        if stale:
            logger.warning(f"Wasatch area scan: dropped {stale} stale bytes "
                           f"({stale // line_len} row(s)) before the exposure")

        fid._send_code(0xad, label="ACQUIRE_SPECTRUM")

        rows: list = []        # (index or None, np.ndarray)
        n_marked = 0
        while len(rows) < max_rows:
            timeout = first_timeout if not rows else drain_timeout
            try:
                raw = self._read_bulk_line(line_len, endpoints, timeout)
            except Exception as e:  # noqa: BLE001
                if self._is_timeout(e):
                    if not rows:
                        logger.warning("Wasatch area scan: no row arrived within "
                                       f"{timeout} ms")
                    break          # the frame is complete (pipe drained)
                logger.warning(f"Wasatch area scan: USB read error after "
                               f"{len(rows)} row(s): {e}")
                self._clear_bulk_halt(endpoints)
                break
            row = np.frombuffer(bytes(raw), dtype='<u2').astype(np.float32)
            index = None
            if int(row[0]) == AREA_SCAN_MARKER and 1 < len(row):
                # Newer firmware: marker + row index in the first two pixels.
                index = int(row[1])
                row[0] = row[1] = row[2] if len(row) > 2 else 0
                n_marked += 1
            rows.append((index, row))

        if not rows:
            return None
        if n_marked == len(rows) and max(i for i, _ in rows) < max_rows:
            height = max(i for i, _ in rows) + 1
            img = np.zeros((height, pixels), dtype=np.float32)
            for i, row in rows:
                img[i] = row
        else:
            img = np.stack([row for _, row in rows]).astype(np.float32)
        if self._invert_x:
            # Match the spectrum's pixel order (the driver reverses spectra,
            # but not area-scan rows, on such units).
            img = img[:, ::-1]
        return img

    def _clear_bulk_halt(self, endpoints: list) -> None:
        """Best effort: clear a stalled bulk endpoint after a pipe error so
        the next ACQUIRE can read again."""
        dev = getattr(self._fid, 'device', None)
        if dev is None or not hasattr(dev, 'clear_halt'):
            return
        for ep in endpoints:
            try:
                dev.clear_halt(ep)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"clear_halt(0x{ep:02x}) failed: {e}")

    def acquire_image(self) -> Optional[np.ndarray]:
        """One whole-sensor frame (rows, pixels) through the area scan, or
        None if the unit cannot / did not deliver one."""
        if self._fid is None or not self._area_scan_supported:
            return None
        if not self._set_area_scan(True):
            return None
        try:
            img = self._read_area_scan_frame()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Wasatch area scan read failed: {e}")
            img = None
        if img is not None and (img.ndim != 2 or not img.size):
            img = None
        if img is None:
            self._area_scan_failures += 1
            logger.warning(f"Wasatch area scan returned no rows "
                           f"({self._area_scan_failures}/{AREA_SCAN_MAX_FAILURES})")
            if self._area_scan_failures >= AREA_SCAN_MAX_FAILURES:
                logger.error("Giving up on the Wasatch area scan: the unit "
                             "advertises it but streams no rows. Spectrograph "
                             "mode is unavailable until reconnect.")
                self._set_area_scan(False)
                self._area_scan_supported = False
            return None
        self._area_scan_failures = 0
        if img.shape[0] != self._rows:
            self._rows = int(img.shape[0])
        return np.ascontiguousarray(img)

    # --- detector temperature -----------------------------------------------

    @staticmethod
    def _response_value(resp):
        """Payload of a wasatch ``SpectrometerResponse`` (or a bare value from
        an old driver); None when the response reports a failure."""
        if _response_failed(resp):
            return None
        return getattr(resp, "data", resp)

    def _probe_cooling(self, settings) -> None:
        """Record, once after connect, whether the unit has a detector TEC and
        what the driver set it to (Wasatch.PY writes the EEPROM start-up set
        point and enables the TEC in its own connect)."""
        eeprom = settings.eeprom
        self._has_cooling = bool(getattr(eeprom, "has_cooling", False))
        self._temperature_c = None
        self._tec_setpoint_c = None
        self._tec_enabled = None
        self._temp_read_failed = False
        if not self._has_cooling:
            logger.info("Wasatch detector has no TEC (EEPROM has_cooling False): "
                        "no temperature readout")
            return
        try:
            resp = self._fid.get_tec_enabled()
            msg = getattr(resp, "error_msg", "") or ""
            if msg:
                logger.warning(f"Could not read the detector TEC state: {msg}")
            else:
                self._tec_enabled = bool(getattr(resp, "data", resp))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not read the detector TEC state: {e}")
        if getattr(self._fid, "detector_tec_setpoint_has_been_set", False):
            try:
                value = self._response_value(self._fid.get_detector_tec_setpoint_degC())
                if value is not None:
                    self._tec_setpoint_c = float(value)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not read the detector TEC set point: {e}")
        if self._tec_setpoint_c is None and self._tec_enabled:
            # TEC on but the driver did not record a set point: it writes the
            # EEPROM start-up value when it connects.
            startup = getattr(eeprom, "startup_temp_degC", None)
            if startup is not None:
                self._tec_setpoint_c = float(startup)
        self._poll_temperature(force=True)
        setpoint = (f"{self._tec_setpoint_c:.0f} C" if self._tec_setpoint_c is not None
                    else "unknown")
        now = (f"{self._temperature_c:.1f} C" if self._temperature_c is not None
               else "not read")
        state = {True: "on", False: "OFF", None: "state unknown"}[self._tec_enabled]
        logger.info(f"Wasatch detector TEC {state}, set point {setpoint}, "
                    f"detector now {now}")

    def _poll_temperature(self, force: bool = False) -> None:
        if self._fid is None or not self._has_cooling:
            return
        now = time.time()
        if not force and now - self._last_temp_poll < TEMPERATURE_POLL_S:
            return
        self._last_temp_poll = now
        try:
            value = self._response_value(self._fid.get_detector_temperature_degC())
        except Exception as e:  # noqa: BLE001
            if not self._temp_read_failed:
                logger.warning(f"Detector temperature read failed: {e}")
            self._temp_read_failed = True
            return
        if value is None:
            return
        try:
            degc = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(degc) or not -80.0 <= degc <= 120.0:
            logger.debug(f"Ignoring implausible detector temperature {degc}")
            return
        if self._temp_read_failed:
            logger.info("Detector temperature readout recovered")
            self._temp_read_failed = False
        self._temperature_c = degc

    def poll(self) -> None:
        self._poll_temperature()

    @property
    def has_cooling(self) -> bool:
        """True if the unit has a detector TEC (EEPROM ``has_cooling``)."""
        return self._has_cooling

    @property
    def temperature_c(self) -> Optional[float]:
        """Last detector temperature read (C); None on uncooled units."""
        return self._temperature_c

    @property
    def tec_setpoint_c(self) -> Optional[float]:
        """TEC set point (C) the driver applied at connect, if known."""
        return self._tec_setpoint_c

    @property
    def temperature_stable(self) -> Optional[bool]:
        """True once the detector is within ``TEC_STABLE_BAND_C`` of the TEC
        set point, False while it is still cooling, None when unknown (no
        reading, no set point, or the TEC is off)."""
        if (self._temperature_c is None or self._tec_setpoint_c is None
                or self._tec_enabled is False):
            return None
        return abs(self._temperature_c - self._tec_setpoint_c) <= TEC_STABLE_BAND_C

    def status(self) -> SpectrometerStatus:
        details = {}
        if self._detector:
            details["Detector"] = self._detector
        if self._micro_fw or self._fpga_fw:
            details["Firmware"] = f"micro {self._micro_fw or '?'} / FPGA {self._fpga_fw or '?'}"
        if self._connected:
            details["Area scan"] = ("supported" if self._area_scan_supported
                                    else "not supported")
            if self._rows > 0:
                start, end = self.bin_rows
                details["Bin rows"] = (f"{start}-{end} of {self._rows}"
                                       + ("" if self._rows_settable
                                          else " (vertical ROI fixed by firmware)"))
        if self._connected and self._has_cooling:
            tec = (f"set point {self._tec_setpoint_c:.0f} C"
                   if self._tec_setpoint_c is not None else "set point unknown")
            if self._tec_enabled is False:
                tec += ", off"
            details["Detector TEC"] = tec
        return SpectrometerStatus(model=self.model, serial=self._serial,
                                  pixels=self._pixels,
                                  detector_temperature_c=self._temperature_c,
                                  temperature_stable=self.temperature_stable,
                                  details=details)

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
