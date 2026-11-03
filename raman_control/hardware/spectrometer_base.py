"""
Abstract spectrometer backend used by :class:`SpectrumThread`.

The thread owns exactly one backend and is the ONLY code that calls into it,
so every backend method here may block (an exposure, a USB round-trip, a DLL
initialisation) without any locking of its own. Backends never create threads,
never touch Qt and never emit signals; they report through return values and
the small :class:`SpectrometerStatus` snapshot the thread polls.

Two backends exist:

* :class:`~raman_control.hardware.wasatch_spectrometer.WasatchSpectrometer`:
  the Wasatch Photonics unit. The same USB device also hosts the excitation
  laser, so this backend is additionally a :class:`LaserController`.
* :class:`~raman_control.hardware.kymera_spectrometer.KymeraSpectrometer`:
  the Andor Kymera 328i spectrograph with the Newton DU920P CCD. It has no
  laser; the Wasatch laser is then driven by a separate
  :class:`~raman_control.hardware.wasatch_laser.WasatchLaser` on its own thread.

Axis convention (shared by all backends and by the GUI / acquisition code):

* ``wavelengths_nm`` is the detector's native calibration in nanometres.
* ``wavenumbers`` is the Raman shift in cm⁻¹, computed from the excitation
  wavelength: ``1e7 / lambda_0 - 1e7 / lambda``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# Backend identifiers used in SpectrometerConfig.backend and in log/status text.
BACKEND_AUTO = 'auto'
BACKEND_WASATCH = 'wasatch'
BACKEND_KYMERA = 'kymera'
BACKEND_CHOICES = (BACKEND_AUTO, BACKEND_WASATCH, BACKEND_KYMERA)

# Human-readable names for the status bar / settings dialog.
BACKEND_LABELS = {
    BACKEND_AUTO: "Auto (Wasatch, fall back to Kymera)",
    BACKEND_WASATCH: "Wasatch Photonics",
    BACKEND_KYMERA: "Andor Kymera + Newton",
}


def wavelength_nm_to_wavenumber(wavelength_nm: np.ndarray,
                                excitation_nm: float) -> np.ndarray:
    """Raman shift (cm⁻¹, Stokes side) of absolute wavelengths (nm)."""
    wl = np.asarray(wavelength_nm, dtype=np.float64)
    wl = np.where(wl == 0, np.nan, wl)
    return 1e7 / float(excitation_nm) - 1e7 / wl


def wavenumber_to_wavelength_nm(wavenumber: np.ndarray,
                                excitation_nm: float) -> np.ndarray:
    """Absolute (Stokes) wavelength (nm) of Raman shifts (cm⁻¹)."""
    wn0 = 1e7 / float(excitation_nm)
    scattered = wn0 - np.asarray(wavenumber, dtype=np.float64)
    scattered = np.where(scattered == 0, np.nan, scattered)
    return 1e7 / scattered


@dataclass
class SpectrometerStatus:
    """Snapshot of backend state for the GUI (polled by the thread).

    ``detector_temperature_c`` / ``temperature_stable`` are only meaningful
    for cooled CCDs (Kymera/Newton); the Wasatch leaves them ``None``.
    """
    model: str = ""
    serial: str = ""
    pixels: int = 0
    detector_temperature_c: Optional[float] = None
    temperature_stable: Optional[bool] = None
    # Free-form extras recorded in capture metadata (grating, centre nm...).
    details: dict = field(default_factory=dict)


class SpectrometerBackend(ABC):
    """One spectrometer. All methods run on the owning spectrum thread."""

    #: Short identifier (``'wasatch'`` / ``'kymera'``), used for config,
    #: metadata and the status bar.
    name: str = ""
    #: Human-readable model string for metadata / logs.
    model: str = ""

    @abstractmethod
    def connect(self) -> bool:
        """Open the device and apply initial settings. False if no device."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release the device. Must be safe to call twice."""

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @property
    @abstractmethod
    def wavelengths_nm(self) -> Optional[np.ndarray]:
        """Native wavelength axis (nm), or None if the device gave none."""

    @property
    @abstractmethod
    def pixels(self) -> int:
        """Number of pixels per spectrum."""

    @abstractmethod
    def set_integration_time_ms(self, ms: float) -> None:
        """Set the exposure / integration time in milliseconds."""

    @abstractmethod
    def acquire(self) -> Optional[np.ndarray]:
        """Take ONE fresh exposure and return the spectrum (pixels,) as
        float32, or None on failure. Blocks for the integration time."""

    def abort(self) -> None:
        """Interrupt an in-progress :meth:`acquire` from another thread.

        Optional; only backends whose acquire() can take a long time (long
        CCD exposures) need it. Default: no-op.
        """

    # --- 2D detectors (optional) ------------------------------------------

    @property
    def supports_image_readout(self) -> bool:
        """True if the detector is a 2D sensor that can return the full frame
        (the Newton CCD behind the Kymera). Default: False."""
        return False

    @property
    def detector_rows(self) -> int:
        """Rows of the 2D sensor (0 for a 1D detector)."""
        return 0

    def set_bin_rows(self, start: int, end: int) -> None:
        """Rows (1-based, inclusive) summed into the spectrum. ``end`` 0 =
        last row. Default: ignored (1D detectors)."""

    @property
    def bin_rows(self) -> tuple:
        """The effective (start, end) rows after clamping, or (0, 0)."""
        return (0, 0)

    def acquire_image(self) -> Optional[np.ndarray]:
        """Take ONE fresh exposure reading the whole 2D sensor and return it
        as a (rows, pixels) float32 frame, or None. Only for backends with
        :attr:`supports_image_readout`; default: None. The next
        :meth:`acquire` switches back to the spectrum readout by itself."""
        return None

    def status(self) -> SpectrometerStatus:
        """Current status snapshot (default: static identity only)."""
        return SpectrometerStatus(model=self.model, pixels=self.pixels)

    def poll(self) -> None:
        """Periodic housekeeping between frames (e.g. read the CCD
        temperature). Called by the thread roughly once per second."""

    @property
    def has_laser(self) -> bool:
        """True if this backend also controls the excitation laser (the
        Wasatch does). The thread then routes laser calls to it."""
        return False

    def metadata(self) -> dict:
        """Key/value pairs for capture / acquisition metadata files."""
        st = self.status()
        meta = {
            "Spectrometer": st.model or self.model or self.name,
        }
        if st.serial:
            meta["Spectrometer serial"] = st.serial
        if st.detector_temperature_c is not None:
            meta["Detector temperature (C)"] = f"{st.detector_temperature_c:.1f}"
        meta.update(st.details)
        return meta


class LaserController(ABC):
    """The excitation laser. Implemented by the Wasatch unit (either as part
    of :class:`WasatchSpectrometer` when it is the active spectrometer, or by
    the stand-alone :class:`WasatchLaser` when the Kymera is)."""

    @abstractmethod
    def set_laser_enable(self, on: bool) -> bool: ...

    @abstractmethod
    def set_laser_power_mw(self, mw: float) -> bool: ...

    @property
    @abstractmethod
    def laser_available(self) -> bool:
        """True when the laser hardware is reachable."""
