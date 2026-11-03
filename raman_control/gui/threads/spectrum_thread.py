"""
PyQt6 thread for spectrometer acquisition.

The thread owns ONE spectrometer backend (see ``hardware/spectrometer_base``)
and is the only code that touches it:

* :class:`WasatchSpectrometer` (Wasatch Photonics unit, spectrometer + laser)
* :class:`KymeraSpectrometer` (Andor Kymera 328i + Newton CCD)

Which one is used follows ``SpectrometerConfig.backend``: 'wasatch', 'kymera',
or 'auto' (Wasatch if detected, otherwise Kymera). The excitation laser is
always the Wasatch laser: when the Wasatch is the active spectrometer, laser
commands go through its backend on this thread; when the Kymera is active, a
separate :class:`LaserThread` opens the Wasatch unit for laser control only
and this thread forwards the laser calls to it.

The public API (signals, acquire_frames / acquire_average, hold_live /
release_live, set_integration_time, set_laser, set_laser_power_mw,
wavelengths / wavelengths_nm) is the same for both spectrometers, so the GUI,
autofocus and acquisition workers do not depend on the hardware in use.

Provides non-blocking spectrum acquisition with:
- Integration time control
- Laser power control (mW)
- Thread-safe spectrum delivery via Qt signals
- Debug fallback to simulated spectra (tests)
"""

import threading
import time
from typing import Optional
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

from raman_control.config.hardware_config import SpectrometerConfig
from raman_control.config.logging_config import get_logger
from raman_control.hardware.spectrometer_base import (
    BACKEND_AUTO, BACKEND_KYMERA, BACKEND_WASATCH, SpectrometerBackend,
    SpectrometerStatus, wavelength_nm_to_wavenumber, wavenumber_to_wavelength_nm,
)

logger = get_logger('spectrum_thread')


class SpectrumThread(QThread):
    """
    Qt thread for spectrometer acquisition.

    Signals:
        spectrum_ready: Emitted with new spectrum (wavelengths, intensities, timestamp)
        connected: Emitted when spectrometer connects
        disconnected: Emitted when spectrometer disconnects
        error: Emitted on errors (error_message)
        laser_state_changed: Emitted when laser state changes (is_on)
        laser_available_changed: Emitted when the excitation laser hardware
            becomes reachable / unreachable (is_available)
        status_changed: Emitted about once a second with a
            :class:`SpectrometerStatus` snapshot (detector temperature, ...)

    Usage:
        spectrometer = SpectrumThread(config)
        spectrometer.spectrum_ready.connect(self.on_spectrum)
        spectrometer.start()
        # Acquisition auto-starts on connect
        spectrometer.stop()
    """

    spectrum_ready = pyqtSignal(np.ndarray, np.ndarray, float)  # wavelengths, intensities, timestamp
    # Full 2D detector frame (rows, pixels) + timestamp, emitted by the live
    # loop instead of spectrum_ready while the image display is selected
    # (Kymera/Newton, see set_image_mode).
    image_ready = pyqtSignal(np.ndarray, float)
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)
    laser_state_changed = pyqtSignal(bool)
    laser_available_changed = pyqtSignal(bool)
    status_changed = pyqtSignal(object)  # SpectrometerStatus

    _STATUS_INTERVAL_S = 1.0

    def __init__(self, config: Optional[SpectrometerConfig] = None,
                 parent=None, laser_thread=None):
        super().__init__(parent)

        self._config = config or SpectrometerConfig()
        self._mutex = QMutex()

        # State
        self._running = False
        self._acquiring = False
        self._connected = False
        self._laser_on = False

        # Wavelength/wavenumber axis (set from hardware or default for simulation)
        self._wavelengths = np.linspace(200, 3500, 2048)  # cm^-1
        # Native wavelength axis (nm) from the device, used for capture/export.
        self._wavelengths_nm: Optional[np.ndarray] = None

        # Latest spectrum storage for on-demand access
        self._latest_spectrum: Optional[np.ndarray] = None

        # On-demand "acquire N fresh exposures" request, serviced by the run
        # loop (the only thread that may touch the hardware). A worker sets
        # _frames_request = N, waits on _frames_done, then reads _frames_result.
        self._frames_request: Optional[int] = None
        self._frames_result: Optional[np.ndarray] = None
        self._frames_done = threading.Event()

        # Pending hardware setting changes (integration time / laser power /
        # laser enable). Setters queue here instead of touching the device
        # directly; the run loop applies them between reads so the USB device is
        # only ever accessed from this one thread (concurrent access from a
        # caller thread + the run loop's get_line() causes libusb timeouts).
        self._pending_settings: dict = {}

        # Live-view pause count. While > 0 the free-running display loop takes
        # no frames of its own; only on-demand acquire_frames() requests run.
        # Autofocus / auto-acquisition hold it so the live loop doesn't slip an
        # extra single frame between their requests (which cost an extra
        # exposure of wait + laser time per step, and replaced the plotted
        # autofocus spectrum with a noisier one). Counted so nested holders
        # can't release each other's pause.
        self._live_holds = 0

        # Hardware backend (set in _connect_spectrometer).
        self._backend: Optional[SpectrometerBackend] = None
        self._backend_name = ""
        # Laser routing. When the backend hosts the laser (Wasatch) the laser
        # settings are applied on THIS thread through the backend. Otherwise the
        # optional LaserThread (Wasatch in laser-only mode) is started and laser
        # calls are forwarded to its queue straight from the setters, so a laser
        # OFF never waits behind a long CCD exposure.
        self._laser_thread = laser_thread
        self._use_laser_thread = False
        self._laser_available = False
        if self._laser_thread is not None:
            self._laser_thread.available_changed.connect(self._on_laser_thread_available)
            self._laser_thread.laser_state_changed.connect(self._on_laser_thread_state)
            self._laser_thread.error.connect(self.error)

        # Simulated mode flag
        self._simulated = False
        self._last_status_emit = 0.0
        # Live display mode for a 2D detector: False = binned spectrum,
        # True = whole-sensor image (see set_image_mode).
        self._image_mode = False
        self._sim_rows = 256  # simulated 2D detector height

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._connected

    @property
    def is_acquiring(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._acquiring

    @property
    def is_laser_on(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._laser_on

    @property
    def is_simulated(self) -> bool:
        return self._simulated

    @property
    def backend_name(self) -> str:
        """'wasatch' / 'kymera' once connected ('' before, 'simulated' in
        simulation)."""
        if self._simulated:
            return 'simulated'
        return self._backend_name

    @property
    def backend_label(self) -> str:
        """Short model name for the status bar."""
        if self._simulated:
            return "Simulated"
        if self._backend is not None:
            return {BACKEND_WASATCH: "Wasatch", BACKEND_KYMERA: "Kymera"}.get(
                self._backend.name, self._backend.name)
        return ""

    @property
    def laser_available(self) -> bool:
        """True when the excitation laser can be driven right now."""
        if self._simulated:
            return True
        if self._use_laser_thread:
            return self._laser_available
        return self._backend is not None and self._backend.has_laser \
            and self._backend.laser_available

    @property
    def supports_image_readout(self) -> bool:
        """True when the connected detector is 2D and can show its frame
        (Kymera/Newton; also True in simulation so the GUI can be tried)."""
        if self._simulated:
            return True
        return self._backend is not None and self._backend.supports_image_readout

    @property
    def detector_rows(self) -> int:
        """Rows of the 2D detector (0 for a 1D spectrometer)."""
        if self._simulated:
            return self._sim_rows
        return self._backend.detector_rows if self._backend is not None else 0

    @property
    def image_mode(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._image_mode

    @property
    def bin_rows(self) -> tuple:
        """Effective (start, end) rows binned into the spectrum, 1-based
        inclusive, as the backend applies them."""
        if self._simulated:
            return self._clamp_rows(self._config.kymera.bin_row_start,
                                    self._config.kymera.bin_row_end,
                                    self._sim_rows)
        return self._backend.bin_rows if self._backend is not None else (0, 0)

    @staticmethod
    def _clamp_rows(start: int, end: int, rows: int) -> tuple:
        if rows <= 0:
            return (int(start), int(end))
        end = rows if end <= 0 or end > rows else int(end)
        start = min(max(1, int(start)), rows)
        if start > end:
            start, end = end, start
        return (start, end)

    @property
    def focus_metric(self) -> str:
        """'ratio' or 'entropy': the spectrometer-autofocus metric for the
        connected hardware (Settings -> Spectrometer, 'auto' picks per device)."""
        return self._config.resolve_af_metric(self._backend_name)

    @property
    def wavelengths(self) -> np.ndarray:
        return self._wavelengths.copy()

    @property
    def wavelengths_nm(self) -> np.ndarray:
        """Native wavelength axis (nm) — the spectrometer's raw output format.

        If the device didn't provide a wavelength calibration, convert the
        Raman shift (cm⁻¹) axis using the configured excitation wavelength.
        """
        if self._wavelengths_nm is not None:
            return self._wavelengths_nm.copy()
        lam0 = self._config.excitation_wavelength_nm or 785.0
        return wavenumber_to_wavelength_nm(self._wavelengths, lam0)

    def metadata(self) -> dict:
        """Spectrometer identity for capture / acquisition metadata files."""
        if self._backend is None:
            return {"Spectrometer": "Simulated" if self._simulated else ""}
        try:
            return self._backend.metadata()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Spectrometer metadata failed: {e}")
            return {"Spectrometer": self._backend.model}

    def status(self) -> Optional[SpectrometerStatus]:
        if self._backend is None:
            return None
        try:
            return self._backend.status()
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def run(self):
        """Thread main loop."""
        logger.info("Spectrometer thread starting")
        self._running = True

        # Connect to spectrometer
        if not self._connect_spectrometer():
            self.error.emit("Failed to connect to spectrometer")
            self._running = False
            return

        self.connected.emit()
        logger.info(f"Spectrometer connected "
                    f"({'simulated' if self._simulated else self._backend_name})")

        # Do NOT auto-start acquisition. The spectrum is acquired only after
        # the user clicks the play button (which calls start_acquisition()).

        # Minimum interval between emissions to avoid overwhelming GUI
        min_emit_interval = 0.05  # 20 Hz max update rate

        while self._running:
            # Apply any queued setting changes (integration time / laser) on
            # THIS thread, before any read, so the device is never accessed
            # concurrently from a caller thread and a get_line() here.
            self._apply_pending_settings()
            self._poll_status()

            # Service an on-demand averaged-acquisition request first. This is
            # the path the auto-acquisition worker uses so its "Averaging"
            # really means N independent exposures (not N reads of one cached
            # free-running frame). Hardware is only ever touched here.
            with QMutexLocker(self._mutex):
                req = self._frames_request
            if req is not None:
                result = self._capture_frames(req)
                with QMutexLocker(self._mutex):
                    self._frames_result = result
                    self._frames_request = None
                    if result is not None and len(result):
                        self._latest_spectrum = result.mean(
                            axis=0).astype(np.float32)
                # Emit before releasing the waiting worker so the plotted
                # spectrum / frame are already out when acquire_frames()
                # returns.
                if result is not None and len(result):
                    self.spectrum_ready.emit(
                        self._wavelengths, self._latest_spectrum, time.time())
                self._frames_done.set()
                continue

            with QMutexLocker(self._mutex):
                live = self._acquiring and self._live_holds == 0
                image_mode = self._image_mode
            if live:
                acq_start = time.time()

                if image_mode:
                    # Image display: read the whole sensor, no spectrum.
                    frame = self._acquire_image()
                    if frame is not None:
                        self.image_ready.emit(np.asarray(frame), acq_start)
                else:
                    # Acquire a single spectrum for display (no averaging).
                    spectrum = self._acquire_spectrum()
                    if spectrum is not None:
                        spectrum = spectrum.astype(np.float32)
                        with QMutexLocker(self._mutex):
                            self._latest_spectrum = spectrum.copy()
                        self.spectrum_ready.emit(
                            self._wavelengths,
                            spectrum,
                            acq_start
                        )

                # For simulated mode, sleep to simulate integration time
                # For real hardware, acquire() already blocks for integration_time
                elapsed = time.time() - acq_start
                if self._simulated:
                    wait_time = max(min_emit_interval,
                                   (self._config.integration_time / 1000.0) - elapsed)
                else:
                    wait_time = max(min_emit_interval - elapsed, 0)
                if wait_time > 0:
                    time.sleep(wait_time)
            else:
                self.msleep(50)

        # Cleanup
        self._disconnect_spectrometer()
        self.disconnected.emit()
        logger.info("Spectrometer thread stopped")

    def stop(self):
        """Stop the spectrometer thread."""
        self._running = False
        self._acquiring = False
        self.set_laser(False)
        # Interrupt a long CCD exposure so stop() does not wait for it.
        backend = self._backend
        if backend is not None:
            try:
                backend.abort()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Spectrometer abort failed: {e}")
        self.wait(5000)

    # ------------------------------------------------------------------
    # Acquisition control
    # ------------------------------------------------------------------

    def start_acquisition(self):
        """Start continuous spectrum acquisition."""
        with QMutexLocker(self._mutex):
            self._acquiring = True
        logger.info("Spectrum acquisition started")

    def stop_acquisition(self):
        """Stop spectrum acquisition."""
        with QMutexLocker(self._mutex):
            self._acquiring = False
        logger.info("Spectrum acquisition stopped")

    def hold_live(self):
        """Pause the free-running live-view frames (see ``_live_holds``).
        On-demand acquire_frames() requests still run and are still plotted.
        A live frame already in progress finishes first. Pair every call with
        :meth:`release_live`."""
        with QMutexLocker(self._mutex):
            self._live_holds += 1
            n = self._live_holds
        logger.debug(f"Live spectrum paused (holds={n})")

    def release_live(self):
        """Undo one :meth:`hold_live`; live view resumes when none remain
        (and only if live acquisition is switched on)."""
        with QMutexLocker(self._mutex):
            self._live_holds = max(0, self._live_holds - 1)
            n = self._live_holds
        logger.debug(f"Live spectrum hold released (holds={n})")

    # ------------------------------------------------------------------
    # Settings (any thread; applied on the spectrum thread)
    # ------------------------------------------------------------------

    def set_integration_time(self, time_ms: float):
        """Set integration time in milliseconds (applied on the spectrum
        thread between reads — never touches the device from the caller)."""
        self._config.integration_time = time_ms
        with QMutexLocker(self._mutex):
            self._pending_settings['integration_time_ms'] = time_ms
        logger.info(f"Integration time set to {time_ms} ms")

    def set_bin_rows(self, start: int, end: int):
        """Rows (1-based, inclusive; end 0 = last) summed into the spectrum
        on a 2D detector. Stored in the config and applied on the spectrum
        thread before the next exposure."""
        self._config.kymera.bin_row_start = int(start)
        self._config.kymera.bin_row_end = int(end)
        with QMutexLocker(self._mutex):
            self._pending_settings['bin_rows'] = (int(start), int(end))
        logger.info(f"Spectrum bin rows set to {start}-{end if end > 0 else 'last'}")

    def apply_readout_config(self):
        """Re-push the bin rows from the config (after the Settings dialog)."""
        k = self._config.kymera
        self.set_bin_rows(k.bin_row_start, k.bin_row_end)

    def set_image_mode(self, on: bool):
        """Select what the live loop reads from a 2D detector: the binned
        spectrum (off, :attr:`spectrum_ready`) or the whole sensor as an image
        (on, :attr:`image_ready`). The two are separate exposures and are
        never mixed. On-demand :meth:`acquire_frames` (autofocus, acquisition)
        always takes spectra, whatever is displayed."""
        on = bool(on)
        with QMutexLocker(self._mutex):
            self._image_mode = on
        logger.info(f"Spectrometer live readout: {'image' if on else 'spectrum'}")

    def _acquire_image(self) -> Optional[np.ndarray]:
        """One full-sensor frame (rows, pixels) on this (owner) thread."""
        try:
            if self._simulated:
                return self._generate_simulated_frame()
            if self._backend is None:
                return None
            return self._backend.acquire_image()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Detector image acquisition error: {e}")
            return None

    def set_laser_power(self, power_percent: float):
        """Set laser power (0-100%)."""
        self._config.laser_power = max(0, min(100, power_percent))
        logger.info(f"Laser power set to {power_percent}%")

    def set_laser_power_mw(self, power_mw: float):
        """Set laser power in milliwatts (applied on the laser's owner thread)."""
        self._config.laser_power_mw = power_mw
        if self._use_laser_thread and self._laser_thread is not None:
            self._laser_thread.request_power_mw(power_mw)
        else:
            with QMutexLocker(self._mutex):
                self._pending_settings['laser_power_mw'] = power_mw
        logger.info(f"Laser power set to {power_mw} mW")

    def set_laser(self, on: bool):
        """Turn laser on or off (applied on the laser's owner thread)."""
        on = bool(on)
        if on and self._connected and not self._simulated and not self.laser_available:
            logger.error("Laser ON requested but the excitation laser is not available")
            self.error.emit("Excitation laser (Wasatch unit) not available")
            with QMutexLocker(self._mutex):
                self._laser_on = False
            self.laser_state_changed.emit(False)
            return
        with QMutexLocker(self._mutex):
            self._laser_on = on
        if self._use_laser_thread and self._laser_thread is not None:
            self._laser_thread.request_enable(on)
        else:
            with QMutexLocker(self._mutex):
                self._pending_settings['laser_enable'] = on
        self.laser_state_changed.emit(on)
        logger.info(f"Laser {'ON' if on else 'OFF'}")

    def _apply_pending_settings(self):
        """Apply queued setting changes to the device. Runs ONLY on the spectrum
        thread (the run loop), so the USB device is never accessed concurrently
        with an exposure."""
        with QMutexLocker(self._mutex):
            if not self._pending_settings:
                return
            pending = self._pending_settings
            self._pending_settings = {}
        if self._simulated or self._backend is None:
            return
        try:
            if 'integration_time_ms' in pending:
                self._backend.set_integration_time_ms(pending['integration_time_ms'])
            if 'bin_rows' in pending:
                self._backend.set_bin_rows(*pending['bin_rows'])
            if self._use_laser_thread and self._laser_thread is not None:
                # Laser settings queued before connect: forward them now.
                if 'laser_power_mw' in pending:
                    self._laser_thread.request_power_mw(pending['laser_power_mw'])
                if 'laser_enable' in pending:
                    self._laser_thread.request_enable(pending['laser_enable'])
            elif self._backend.has_laser:
                if 'laser_power_mw' in pending:
                    self._backend.set_laser_power_mw(pending['laser_power_mw'])
                if 'laser_enable' in pending:
                    self._backend.set_laser_enable(pending['laser_enable'])
            elif pending.get('laser_enable'):
                self.error.emit("No excitation laser available for this spectrometer")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to apply spectrometer settings: {e}")

    def _poll_status(self):
        """Backend housekeeping (CCD temperature...) about once a second."""
        if self._simulated or self._backend is None:
            return
        now = time.time()
        if now - self._last_status_emit < self._STATUS_INTERVAL_S:
            return
        self._last_status_emit = now
        try:
            self._backend.poll()
            self.status_changed.emit(self._backend.status())
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Spectrometer status poll failed: {e}")

    def _on_laser_thread_available(self, available: bool):
        self._laser_available = bool(available)
        if self._use_laser_thread:
            self.laser_available_changed.emit(self._laser_available)

    def _on_laser_thread_state(self, on: bool):
        with QMutexLocker(self._mutex):
            changed = self._laser_on != bool(on)
            self._laser_on = bool(on)
        if changed:
            self.laser_state_changed.emit(bool(on))

    # ------------------------------------------------------------------
    # On-demand capture
    # ------------------------------------------------------------------

    def get_latest_spectrum(self) -> Optional[np.ndarray]:
        """Get the most recently acquired spectrum (thread-safe)."""
        with QMutexLocker(self._mutex):
            if self._latest_spectrum is not None:
                return self._latest_spectrum.copy()
            return None

    def acquire_frames(self, n: int, timeout: Optional[float] = None
                       ) -> Optional[np.ndarray]:
        """Acquire ``n`` genuinely fresh exposures and return them stacked as an
        (m, pixels) array (m <= n, m = successful exposures), or None.

        For use from a worker thread: the acquisition runs on the spectrometer
        thread (the hardware owner), so the frames are independent exposures
        rather than repeated reads of the free-running cache. Blocks until the
        frames are captured or ``timeout`` (seconds) elapses.
        """
        n = max(1, int(n))
        with QMutexLocker(self._mutex):
            if not self._connected:
                return None
        if timeout is None:
            # Generous: n integration times plus headroom.
            timeout = max(10.0, (self._config.integration_time / 1000.0) * n * 3 + 10)

        self._frames_done.clear()
        with QMutexLocker(self._mutex):
            self._frames_result = None
            self._frames_request = n

        if not self._frames_done.wait(timeout):
            with QMutexLocker(self._mutex):
                self._frames_request = None
            logger.warning(f"acquire_frames({n}) timed out")
            return None
        with QMutexLocker(self._mutex):
            return None if self._frames_result is None else self._frames_result.copy()

    def acquire_average(self, n: int, timeout: Optional[float] = None
                        ) -> Optional[np.ndarray]:
        """Mean of ``n`` fresh exposures (see :meth:`acquire_frames`)."""
        frames = self.acquire_frames(n, timeout)
        if frames is None or not len(frames):
            return None
        return frames.mean(axis=0).astype(np.float32)

    def _capture_frames(self, n: int) -> Optional[np.ndarray]:
        """Acquire ``n`` exposures back-to-back on this (owner) thread."""
        frames = []
        for _ in range(max(1, n)):
            if not self._running:
                break
            s = self._acquire_spectrum()
            if s is not None:
                frames.append(np.asarray(s, dtype=np.float32))
        if not frames:
            return None
        return np.array(frames, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal: connect / disconnect / acquire
    # ------------------------------------------------------------------

    def _candidate_backends(self):
        """Backends to try, in order, for the configured selection."""
        # Imported here so the SDKs are only loaded on the spectrum thread and
        # a missing one does not break the application import.
        from raman_control.hardware.wasatch_spectrometer import WasatchSpectrometer
        from raman_control.hardware.kymera_spectrometer import KymeraSpectrometer

        choice = (self._config.backend or BACKEND_AUTO).lower()
        if choice == BACKEND_WASATCH:
            return [WasatchSpectrometer(self._config)]
        if choice == BACKEND_KYMERA:
            return [KymeraSpectrometer(self._config)]
        if choice != BACKEND_AUTO:
            logger.warning(f"Unknown spectrometer backend '{choice}', using auto")
        return [WasatchSpectrometer(self._config), KymeraSpectrometer(self._config)]

    def _connect_spectrometer(self) -> bool:
        """Connect to the configured spectrometer (with 'auto' fallback)."""
        candidates = self._candidate_backends()
        backend = None
        for cand in candidates:
            logger.info(f"Trying spectrometer backend: {cand.name}")
            try:
                if cand.connect():
                    backend = cand
                    break
            except Exception as e:  # noqa: BLE001
                logger.error(f"Spectrometer backend {cand.name} failed: {e}")
            if len(candidates) > 1:
                logger.warning(f"Spectrometer '{cand.name}' not available, "
                               f"trying the next one")
        if backend is None:
            logger.error("No spectrometer could be connected "
                         f"(tried: {', '.join(c.name for c in candidates)})")
            return False

        self._backend = backend
        self._backend_name = backend.name
        self._setup_axes(backend)

        # Laser routing.
        if backend.has_laser:
            self._use_laser_thread = False
            self._laser_available = backend.laser_available
        elif self._laser_thread is not None:
            self._use_laser_thread = True
            if not self._laser_thread.isRunning():
                self._laser_thread.start()
            self._laser_available = self._laser_thread.is_available
        else:
            self._use_laser_thread = False
            self._laser_available = False
            logger.warning(f"Spectrometer '{backend.name}' has no laser and no "
                           f"laser thread was provided: laser control disabled")
        self.laser_available_changed.emit(self.laser_available)

        self._simulated = False
        with QMutexLocker(self._mutex):
            self._connected = True
        self._last_status_emit = 0.0
        return True

    def _setup_axes(self, backend: SpectrometerBackend):
        """Fill the cm⁻¹ and nm axes from the backend's calibration."""
        lam0 = self._config.excitation_wavelength_nm or 785.0
        wl_nm = backend.wavelengths_nm
        device_wn = getattr(backend, 'device_wavenumbers', None)
        if wl_nm is not None and len(wl_nm):
            self._wavelengths_nm = np.asarray(wl_nm, dtype=np.float64)
        else:
            self._wavelengths_nm = None
        if device_wn is not None and len(device_wn):
            # Wasatch units ship their own Raman-shift calibration; keep using it.
            self._wavelengths = np.asarray(device_wn, dtype=np.float64)
        elif self._wavelengths_nm is not None:
            self._wavelengths = wavelength_nm_to_wavenumber(self._wavelengths_nm, lam0)
        else:
            logger.warning("No wavelength calibration from the spectrometer, "
                           "using pixel indices")
            self._wavelengths = np.arange(max(1, backend.pixels), dtype=np.float64)
        logger.info(f"Spectrometer axis: {len(self._wavelengths)} px, "
                    f"{self._wavelengths[0]:.1f} - {self._wavelengths[-1]:.1f} cm^-1")

    def _disconnect_spectrometer(self):
        """Disconnect from the spectrometer (and the laser-only thread)."""
        try:
            if self._laser_thread is not None and self._laser_thread.isRunning():
                self._laser_thread.stop()
            if not self._simulated and self._backend is not None:
                try:
                    if self._backend.has_laser:
                        self._backend.set_laser_enable(False)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to disable laser during disconnect: {e}")
                try:
                    self._backend.disconnect()
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Spectrometer disconnect error: {e}")
            self._backend = None
            self._use_laser_thread = False
            with QMutexLocker(self._mutex):
                self._connected = False
        except Exception as e:  # noqa: BLE001
            logger.error(f"Spectrometer disconnect error: {e}")

    def _acquire_spectrum(self) -> Optional[np.ndarray]:
        """Acquire a single spectrum."""
        try:
            if self._simulated:
                return self._generate_simulated_spectrum()
            if self._backend is None:
                return None
            # Blocks for the integration time (on this, the owner, thread).
            spectrum = self._backend.acquire()
            if spectrum is None:
                return None
            return np.asarray(spectrum, dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Spectrum acquisition error: {e}")
            return None

    def _generate_simulated_spectrum(self) -> np.ndarray:
        """Generate a simulated Raman spectrum."""
        # Base spectrum (background)
        baseline = 500 + 100 * np.exp(-((self._wavelengths - 1500) ** 2) / 500000)

        # Add characteristic Raman peaks
        peaks = [
            (520, 2000, 20),   # Silicon reference
            (1000, 800, 30),  # Phenylalanine
            (1450, 600, 40),  # CH2 deformation
            (1650, 400, 50),  # Amide I
            (2930, 1000, 60), # CH stretch
        ]

        spectrum = baseline.copy()
        for center, height, width in peaks:
            peak = height * np.exp(-((self._wavelengths - center) ** 2) / (2 * width ** 2))
            spectrum += peak

        # Add noise proportional to integration time
        noise_level = 50 / np.sqrt(self._config.integration_time / 100)
        noise = np.random.normal(0, noise_level, len(spectrum))
        spectrum += noise

        # Scale by laser power if on
        if self._laser_on:
            spectrum *= (self._config.laser_power / 100)
        else:
            spectrum = baseline + noise

        # Add time variation
        t = time.time()
        spectrum *= (1 + 0.05 * np.sin(t * 0.5))
        return np.maximum(spectrum, 0).astype(np.float32)

    def _generate_simulated_frame(self) -> np.ndarray:
        """Fake 2D sensor frame: the simulated spectrum spread over a
        Gaussian stripe of rows around the chip centre (slightly tilted, as a
        misaligned fibre would give), over a dark floor with noise."""
        spectrum = self._generate_simulated_spectrum()
        rows = self._sim_rows
        r = np.arange(rows, dtype=np.float32)[:, None]
        tilt = 4.0 * np.linspace(-1, 1, len(spectrum), dtype=np.float32)[None, :]
        profile = np.exp(-((r - rows / 2 - tilt) ** 2) / (2 * 12.0 ** 2))
        profile /= profile.sum(axis=0, keepdims=True)
        frame = profile * spectrum[None, :]
        frame += 100 + np.random.normal(0, 5, frame.shape)
        return np.maximum(frame, 0).astype(np.float32)
