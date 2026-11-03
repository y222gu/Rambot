"""
Andor Kymera 328i spectrograph + Newton DU920P CCD backend.

Driven through Andor's own Python wrappers, ``pyAndorSDK2`` (``atmcd``, the
camera) and ``pyAndorSpectrograph`` (``ATSpectrograph``, the grating turret and
wavelength drive). The sequence mirrors the lab's working notebook code
(``utilityraman.Camera`` in the automation notebooks), with these changes:

* Cooling does not block. ``SetTemperature`` + ``CoolerON`` are issued at
  connect and the temperature is polled between frames; acquisition is allowed
  before the set point is reached (the status shows the live temperature).
* Every exposure waits with ``WaitForAcquisitionTimeOut`` (bounded) instead of
  the unbounded ``WaitForAcquisition``, and :meth:`abort` can interrupt a long
  exposure from another thread via ``CancelWait`` + ``AbortAcquisition``.
* The wavelength axis is read from the spectrograph calibration ONCE after the
  grating / centre wavelength are set, not after every frame.
* Readout geometry is configurable. The Newton is a 2D CCD (1024 x 256).
  :meth:`acquire` returns the spectrum: the on-chip sum of the rows
  ``KymeraConfig.bin_row_start..bin_row_end`` read in SINGLE TRACK mode
  (the whole chip, the default, is read as FULL VERTICAL BINNING, the same
  operation over all rows). :meth:`acquire_image` instead reads the whole
  sensor in IMAGE mode and returns the (rows, pixels) frame for the GUI's
  2D view; the two are never mixed in one exposure. Read-mode changes are
  made between exposures (the SDK rejects them while acquiring).

Units: the GUI works in milliseconds, the SDK in seconds. The spectrograph's
calibration is in nanometres; Raman shift is derived with the excitation
wavelength from :class:`SpectrometerConfig`.

Only one process can hold the Newton: Andor SOLIS must be closed.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import numpy as np

from raman_control.config.hardware_config import KymeraConfig, SpectrometerConfig
from raman_control.config.logging_config import get_logger
from raman_control.hardware.spectrometer_base import (
    BACKEND_KYMERA, SpectrometerBackend, SpectrometerStatus,
)

logger = get_logger('kymera_spectrometer')

# Andor SDK2 return codes (atmcd_errors.Error_Codes), listed here so the
# backend works even if an older wrapper lacks a name.
DRV_SUCCESS = 20002
DRV_TEMP_OFF = 20034
DRV_TEMP_NOT_STABILIZED = 20035
DRV_TEMP_STABILIZED = 20036
DRV_TEMP_NOT_REACHED = 20037
DRV_TEMP_DRIFT = 20040
DRV_NO_NEW_DATA = 20024
DRV_ACQUIRING = 20072
DRV_IDLE = 20073
DRV_NOT_INITIALIZED = 20075
# ATSpectrograph return codes
ATSPECTROGRAPH_SUCCESS = 20202
# atmcd_codes.Read_Mode values (stable across SDK2 versions)
READ_MODE_FVB = 0
READ_MODE_SINGLE_TRACK = 3
READ_MODE_IMAGE = 4


def _to_counts(arr) -> np.ndarray:
    """SDK ``*16`` readouts come back through a ``c_short`` buffer, so pixels
    above 32767 wrap negative; reinterpret them as the unsigned 16-bit counts
    they really are."""
    a = np.asarray(arr)
    if a.dtype == np.int16:
        a = a.view(np.uint16)
    return a.astype(np.float32)

_DEFAULT_SDK_DIRS = (
    r"C:\Program Files\Andor SDK",
    r"C:\Program Files\Andor SOLIS",
)


def _add_dll_dirs(extra: str = ""):
    """Make the Andor DLLs findable. The pip packages ship their own copies in
    ``<package>/libs``; the system SDK / SOLIS installs are a fallback."""
    dirs = []
    if extra:
        dirs.append(extra)
    for pkg in ('pyAndorSDK2', 'pyAndorSpectrograph'):
        try:
            mod = __import__(pkg)
            dirs.append(os.path.join(os.path.dirname(mod.__file__), 'libs'))
        except Exception:  # noqa: BLE001
            pass
    dirs.extend(_DEFAULT_SDK_DIRS)
    for d in dirs:
        if d and os.path.isdir(d):
            if d not in os.environ.get('PATH', ''):
                os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')
            try:
                os.add_dll_directory(d)
            except (AttributeError, OSError):
                pass


class KymeraSpectrometer(SpectrometerBackend):
    """Andor Kymera spectrograph with a Newton CCD, read out in FVB mode."""

    name = BACKEND_KYMERA
    model = "Andor Kymera 328i + Newton DU920P"

    def __init__(self, config: SpectrometerConfig,
                 kymera: Optional[KymeraConfig] = None):
        self._config = config
        self._k = kymera or getattr(config, 'kymera', None) or KymeraConfig()
        self._cam = None   # pyAndorSDK2.atmcd
        self._spc = None   # pyAndorSpectrograph.spectrograph.ATSpectrograph
        self._codes = None
        self._connected = False
        self._xpixels = 0
        self._ypixels = 0
        self._serial = ""
        self._spc_serial = ""
        self._wavelengths_nm: Optional[np.ndarray] = None
        self._exposure_s = max(0.0, float(config.integration_time) / 1000.0)
        self._applied_exposure_s: Optional[float] = None
        self._temperature_c: Optional[float] = None
        self._temp_code: Optional[int] = None
        self._last_temp_poll = 0.0
        self._acquiring = threading.Event()
        self._abort_requested = False
        self._grating = int(self._k.grating)
        self._center_nm = float(self._k.center_wavelength_nm)
        # Readout geometry. The CCD is 2D (1024 x 256 on the Newton DU920P):
        # the spectrum is the on-chip sum of rows _bin_start.._bin_end
        # (1-based, inclusive) read as a single track (full range: FVB).
        # acquire_image() reads the whole sensor instead.
        self._bin_start = int(self._k.bin_row_start)
        self._bin_end = int(self._k.bin_row_end)
        self._applied_readout: Optional[tuple] = None  # (mode, start, end)

    # --- helpers --------------------------------------------------------------

    def _spc_desc(self, code: int) -> str:
        try:
            return str(self._spc.GetFunctionReturnDescription(code, 64)[1])
        except Exception:  # noqa: BLE001
            return str(code)

    def _check(self, ret: int, what: str, fatal: bool = True) -> bool:
        if ret == DRV_SUCCESS:
            return True
        msg = f"Andor {what} returned {ret}"
        if fatal:
            logger.error(msg)
        else:
            logger.warning(msg)
        return False

    def _check_spc(self, ret: int, what: str, fatal: bool = True) -> bool:
        if ret == ATSPECTROGRAPH_SUCCESS:
            return True
        msg = f"Kymera {what} returned {ret} ({self._spc_desc(ret)})"
        if fatal:
            logger.error(msg)
        else:
            logger.warning(msg)
        return False

    # --- SpectrometerBackend --------------------------------------------------

    def connect(self) -> bool:
        try:
            _add_dll_dirs(self._k.sdk_dir)
            from pyAndorSDK2 import atmcd, atmcd_codes
            from pyAndorSpectrograph.spectrograph import ATSpectrograph
        except ImportError as e:
            logger.error(f"Andor SDK Python packages not installed "
                         f"(pyAndorSDK2 / pyAndorSpectrograph): {e}")
            return False
        except OSError as e:
            logger.error(f"Andor SDK DLLs could not be loaded: {e}")
            return False

        try:
            self._codes = atmcd_codes
            self._spc = ATSpectrograph()
            self._cam = atmcd()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Andor SDK library load failed: {e}")
            self._cam = self._spc = None
            return False

        # Spectrograph first (as the lab code does), then the camera. Both
        # Initialize calls can take several seconds; we are on the spectrum
        # thread so that is fine.
        t0 = time.time()
        shm = self._spc.Initialize("")
        if not self._check_spc(shm, "Initialize"):
            self._spc = None
            self._cam = None
            return False
        ret = self._cam.Initialize("")
        if ret != DRV_SUCCESS:
            logger.error(f"Newton Initialize returned {ret} (is Andor SOLIS "
                         f"still open? Only one program can use the camera)")
            try:
                self._spc.Close()
            except Exception:  # noqa: BLE001
                pass
            self._spc = None
            self._cam = None
            return False
        logger.info(f"Andor camera + spectrograph initialised in "
                    f"{time.time() - t0:.1f} s")

        try:
            ret, serial = self._cam.GetCameraSerialNumber()
            if ret == DRV_SUCCESS:
                self._serial = str(serial)
            try:
                shm, spc_serial = self._spc.GetSerialNumber(0, 64)
                if shm == ATSPECTROGRAPH_SUCCESS:
                    self._spc_serial = str(spc_serial)
            except Exception:  # noqa: BLE001
                pass
            logger.info(f"Newton serial {self._serial}, Kymera serial "
                        f"{self._spc_serial}")

            # --- cooling (non-blocking) ---
            if self._k.cooler_on:
                self._check(self._cam.SetTemperature(int(self._k.target_temperature_c)),
                            f"SetTemperature({self._k.target_temperature_c})",
                            fatal=False)
                # Cooler mode 1: keep the set point when ShutDown is called,
                # so disconnecting never leaves the CCD warming uncontrolled.
                try:
                    self._cam.SetCoolerMode(1 if self._k.keep_cooler_on_shutdown else 0)
                except Exception:  # noqa: BLE001
                    pass
                self._check(self._cam.CoolerON(), "CoolerON", fatal=False)
            self._poll_temperature(force=True)

            # --- readout configuration: single scan, binned rows / image ---
            codes = self._codes
            if not self._check(self._cam.SetAcquisitionMode(
                    codes.Acquisition_Mode.SINGLE_SCAN), "SetAcquisitionMode"):
                raise RuntimeError("SetAcquisitionMode failed")
            if not self._check(self._cam.SetTriggerMode(
                    codes.Trigger_Mode.INTERNAL), "SetTriggerMode"):
                raise RuntimeError("SetTriggerMode failed")
            ret, xpix, ypix = self._cam.GetDetector()
            if not self._check(ret, "GetDetector"):
                raise RuntimeError("GetDetector failed")
            self._xpixels, self._ypixels = int(xpix), int(ypix)
            logger.info(f"Newton detector {self._xpixels} x {self._ypixels} px")
            self._applied_readout = None
            if not self._apply_readout(image=False):
                raise RuntimeError("readout mode configuration failed")
            self._apply_exposure()
            self._check(self._cam.PrepareAcquisition(), "PrepareAcquisition",
                        fatal=False)

            # --- spectrograph: grating, centre wavelength, calibration ---
            if not self._configure_spectrograph(self._grating, self._center_nm):
                raise RuntimeError("spectrograph configuration failed")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Kymera/Newton configuration failed: {e}")
            self.disconnect()
            return False

        self._connected = True
        return True

    def _configure_spectrograph(self, grating: int, center_nm: float) -> bool:
        """Select the grating and centre wavelength, then read the per-pixel
        wavelength calibration for the current detector."""
        spc = self._spc
        shm, present = spc.IsGratingPresent(0)
        if shm == ATSPECTROGRAPH_SUCCESS and present:
            shm, current = spc.GetGrating(0)
            if shm != ATSPECTROGRAPH_SUCCESS or int(current) != int(grating):
                logger.info(f"Kymera: selecting grating {grating} (turret move)")
                if not self._check_spc(spc.SetGrating(0, int(grating)),
                                       f"SetGrating({grating})"):
                    return False
            shm, current = spc.GetGrating(0)
            if shm == ATSPECTROGRAPH_SUCCESS:
                self._grating = int(current)
        else:
            logger.warning("Kymera: no grating turret reported; using the "
                           "current grating")

        shm, present = spc.IsWavelengthPresent(0)
        if shm == ATSPECTROGRAPH_SUCCESS and present:
            if not self._check_spc(spc.SetWavelength(0, float(center_nm)),
                                   f"SetWavelength({center_nm})"):
                return False
            shm, wave = spc.GetWavelength(0)
            if shm == ATSPECTROGRAPH_SUCCESS:
                self._center_nm = float(wave)
            shm, wmin, wmax = spc.GetWavelengthLimits(0, self._grating)
            if shm == ATSPECTROGRAPH_SUCCESS:
                logger.info(f"Kymera grating {self._grating}: centre "
                            f"{self._center_nm:.1f} nm, range {wmin:.0f}-{wmax:.0f} nm")
        else:
            logger.warning("Kymera: no wavelength drive reported")

        # Calibration: tell the spectrograph the detector geometry, then ask it
        # for the wavelength of every pixel.
        ret, xsize_um, _ysize_um = self._cam.GetPixelSize()
        if not self._check(ret, "GetPixelSize", fatal=False):
            xsize_um = 26.0  # Newton DU920P pixel pitch
        self._check_spc(spc.SetNumberPixels(0, self._xpixels), "SetNumberPixels",
                        fatal=False)
        self._check_spc(spc.SetPixelWidth(0, float(xsize_um)), "SetPixelWidth",
                        fatal=False)
        shm, calib = spc.GetCalibration(0, self._xpixels)
        if not self._check_spc(shm, "GetCalibration"):
            return False
        wl = np.asarray(calib, dtype=np.float64)[:self._xpixels]
        if wl.size != self._xpixels or not np.all(np.isfinite(wl)):
            logger.error("Kymera calibration has the wrong length or NaNs")
            return False
        self._wavelengths_nm = wl
        logger.info(f"Kymera calibration: {wl[0]:.2f} - {wl[-1]:.2f} nm over "
                    f"{wl.size} px")
        return True

    def disconnect(self) -> None:
        self._connected = False
        cam, spc = self._cam, self._spc
        self._cam = self._spc = None
        if cam is not None:
            try:
                if self._acquiring.is_set():
                    cam.CancelWait()
                    cam.AbortAcquisition()
            except Exception:  # noqa: BLE001
                pass
            try:
                if not self._k.keep_cooler_on_shutdown:
                    cam.CoolerOFF()
                ret = cam.ShutDown()
                if ret != DRV_SUCCESS:
                    logger.warning(f"Newton ShutDown returned {ret}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Newton shutdown error: {e}")
        if spc is not None:
            try:
                spc.Close()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Kymera close error: {e}")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def wavelengths_nm(self) -> Optional[np.ndarray]:
        return None if self._wavelengths_nm is None else self._wavelengths_nm.copy()

    @property
    def pixels(self) -> int:
        return self._xpixels

    @property
    def grating(self) -> int:
        return self._grating

    @property
    def center_wavelength_nm(self) -> float:
        return self._center_nm

    def set_integration_time_ms(self, ms: float) -> None:
        # Applied right before the next StartAcquisition (SetExposureTime is
        # rejected while an acquisition is in progress).
        self._exposure_s = max(0.0, float(ms) / 1000.0)

    def _apply_exposure(self) -> None:
        if self._cam is None or self._applied_exposure_s == self._exposure_s:
            return
        if self._check(self._cam.SetExposureTime(float(self._exposure_s)),
                       f"SetExposureTime({self._exposure_s:.4f})", fatal=False):
            self._applied_exposure_s = self._exposure_s
            try:
                ret, exp, acc, kin = self._cam.GetAcquisitionTimings()
                if ret == DRV_SUCCESS and abs(exp - self._exposure_s) > 1e-6:
                    logger.info(f"Newton adjusted exposure to {exp * 1000:.2f} ms "
                                f"(requested {self._exposure_s * 1000:.2f} ms)")
            except Exception:  # noqa: BLE001
                pass

    # --- readout geometry (bin rows / 2D image) -------------------------------

    @property
    def supports_image_readout(self) -> bool:
        return True

    @property
    def detector_rows(self) -> int:
        return self._ypixels

    def set_bin_rows(self, start: int, end: int) -> None:
        """Rows (1-based, inclusive) summed into the spectrum; end 0 = last
        row. Takes effect before the next exposure (SetReadMode / SetImage are
        rejected while acquiring)."""
        self._bin_start = int(start)
        self._bin_end = int(end)
        self._k.bin_row_start = self._bin_start
        self._k.bin_row_end = self._bin_end

    @property
    def bin_rows(self) -> tuple:
        return self._effective_rows()

    def _effective_rows(self) -> tuple:
        """Clamp the configured rows to the detector: (start, end), 1-based
        inclusive, start <= end. Before connect the height is unknown and the
        raw values are returned."""
        y = self._ypixels
        start, end = self._bin_start, self._bin_end
        if y <= 0:
            return (start, end)
        if end <= 0 or end > y:
            end = y
        start = min(max(1, start), y)
        if start > end:
            start, end = end, start
        return (start, end)

    def _apply_readout(self, image: bool) -> bool:
        """Push the read mode + image geometry to the SDK if they changed.
        ``image`` selects the full-frame readout, otherwise the binned-rows
        spectrum readout. Runs on the owner thread between exposures."""
        cam = self._cam
        if cam is None or self._ypixels <= 0:
            return False
        start, end = self._effective_rows()
        x, y = self._xpixels, self._ypixels
        if image:
            wanted = ('image', 1, y)
        elif (start, end) == (1, y):
            wanted = ('fvb', 1, y)
        else:
            wanted = ('rows', start, end)
        if wanted == self._applied_readout:
            return True

        mode = wanted[0]
        if mode == 'fvb':
            ok = self._check(cam.SetReadMode(READ_MODE_FVB), "SetReadMode(FVB)")
            # Geometry is irrelevant in FVB but keep the SDK's image window
            # consistent with the whole chip.
            self._check(cam.SetImage(1, 1, 1, x, 1, y), "SetImage", fatal=False)
        elif mode == 'rows':
            # Single track: the chip sums only the rows of the track and
            # returns ONE row of x pixels. The SDK describes the track by its
            # centre row and height; for an even height the track is centred
            # between two rows, so the covered rows can be off by one from
            # (start, end) at one edge.
            height = end - start + 1
            centre = (start + end) // 2
            ok = (self._check(cam.SetReadMode(READ_MODE_SINGLE_TRACK),
                              "SetReadMode(SINGLE_TRACK)")
                  and self._check(cam.SetSingleTrack(centre, height),
                                  f"SetSingleTrack(centre {centre}, height {height})"))
        else:
            ok = (self._check(cam.SetReadMode(READ_MODE_IMAGE), "SetReadMode(IMAGE)")
                  and self._check(cam.SetImage(1, 1, 1, x, 1, y), "SetImage(full)"))
        if not ok:
            return False
        self._applied_readout = wanted
        logger.info(f"Newton readout: {mode}, bin rows {start}-{end} of {y}")
        return True

    def acquire(self) -> Optional[np.ndarray]:
        """One spectrum: the on-chip binned readout of the configured rows."""
        if self._cam is None or not self._connected:
            return None
        self._abort_requested = False
        self._apply_exposure()
        return self._expose(image=False)

    def acquire_image(self) -> Optional[np.ndarray]:
        """One full-sensor frame (rows, pixels) in IMAGE read mode."""
        if self._cam is None or not self._connected:
            return None
        self._abort_requested = False
        self._apply_exposure()
        return self._expose(image=True)

    def _expose(self, image: bool) -> Optional[np.ndarray]:
        """Take one exposure in the requested readout and return the data:
        (pixels,) for the spectrum readout, (rows, pixels) for the image."""
        cam = self._cam
        if cam is None or not self._apply_readout(image):
            return None
        exposure_s = self._applied_exposure_s or self._exposure_s
        timeout_ms = int((exposure_s * 2.0 + float(self._k.acquisition_timeout_margin_s))
                         * 1000)
        size = self._xpixels * self._ypixels if image else self._xpixels

        self._acquiring.set()
        try:
            if not self._check(cam.StartAcquisition(), "StartAcquisition", fatal=False):
                try:
                    cam.AbortAcquisition()
                except Exception:  # noqa: BLE001
                    pass
                return None
            ret = cam.WaitForAcquisitionTimeOut(timeout_ms)
            if ret != DRV_SUCCESS:
                if self._abort_requested:
                    logger.info("Newton exposure aborted")
                else:
                    logger.error(f"Newton exposure timed out after {timeout_ms} ms "
                                 f"(WaitForAcquisitionTimeOut returned {ret})")
                try:
                    cam.AbortAcquisition()
                except Exception:  # noqa: BLE001
                    pass
                return None
            ret, arr, _first, _last = cam.GetImages16(1, 1, size)
            if not self._check(ret, "GetImages16", fatal=False):
                return None
            data = _to_counts(arr).reshape(-1)[:size]
            if data.size != size:
                logger.warning(f"Newton returned {data.size} values, expected "
                               f"{size}")
                return None
            if image:
                return data.reshape(self._ypixels, self._xpixels)
            return data
        except Exception as e:  # noqa: BLE001
            logger.error(f"Newton acquisition error: {e}")
            return None
        finally:
            self._acquiring.clear()

    def abort(self) -> None:
        """Interrupt a long exposure (callable from another thread: Andor's
        CancelWait exists for exactly this)."""
        cam = self._cam
        if cam is None or not self._acquiring.is_set():
            return
        self._abort_requested = True
        try:
            cam.CancelWait()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"CancelWait failed: {e}")

    # --- temperature ------------------------------------------------------------

    def _poll_temperature(self, force: bool = False) -> None:
        if self._cam is None:
            return
        now = time.time()
        if not force and now - self._last_temp_poll < 1.0:
            return
        if self._acquiring.is_set():
            return  # GetTemperature returns DRV_ACQUIRING during an exposure
        self._last_temp_poll = now
        try:
            ret, temp = self._cam.GetTemperature()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"GetTemperature failed: {e}")
            return
        if ret in (DRV_ACQUIRING, DRV_NOT_INITIALIZED):
            return
        self._temp_code = int(ret)
        self._temperature_c = float(temp)

    def poll(self) -> None:
        self._poll_temperature()

    @property
    def temperature_c(self) -> Optional[float]:
        return self._temperature_c

    @property
    def temperature_stable(self) -> Optional[bool]:
        if self._temp_code is None:
            return None
        return self._temp_code == DRV_TEMP_STABILIZED

    def status(self) -> SpectrometerStatus:
        start, end = self._effective_rows()
        details = {
            "Grating": self._grating,
            "Center wavelength (nm)": f"{self._center_nm:.1f}",
            "Bin rows": f"{start}-{end}",
            "Spectrum readout": "FVB" if (start, end) == (1, self._ypixels)
                                else "single track",
        }
        if self._spc_serial:
            details["Spectrograph serial"] = self._spc_serial
        return SpectrometerStatus(
            model=self.model, serial=self._serial, pixels=self._xpixels,
            detector_temperature_c=self._temperature_c,
            temperature_stable=self.temperature_stable,
            details=details)
