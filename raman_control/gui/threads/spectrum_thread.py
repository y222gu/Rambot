"""
PyQt6 thread for spectrometer acquisition using Wasatch Photonics SDK.

Provides non-blocking spectrum acquisition with:
- Real Wasatch SDK integration (WasatchBus/WasatchDevice)
- Debug mode fallback to simulated spectra
- Integration time control
- Laser power control (mW)
- Thread-safe spectrum delivery via Qt signals
"""

import threading
import time
from typing import Optional
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

from raman_control.config.hardware_config import SpectrometerConfig
from raman_control.config.logging_config import get_logger

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

    Usage:
        spectrometer = SpectrumThread(config)
        spectrometer.spectrum_ready.connect(self.on_spectrum)
        spectrometer.start()
        # Acquisition auto-starts on connect
        spectrometer.stop()
    """

    spectrum_ready = pyqtSignal(np.ndarray, np.ndarray, float)  # wavelengths, intensities, timestamp
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)
    laser_state_changed = pyqtSignal(bool)

    def __init__(self, config: Optional[SpectrometerConfig] = None,
                 parent=None):
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

        # Wasatch SDK objects
        self._device = None
        self._fid = None  # FID = Feature Identification Device (hardware interface)

        # Simulated mode flag
        self._simulated = False

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
        wn0 = 1e7 / lam0                                   # laser wavenumber
        scattered = wn0 - self._wavelengths               # Stokes side
        scattered = np.where(scattered == 0, np.nan, scattered)
        return 1e7 / scattered

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
        logger.info(f"Spectrometer connected ({'simulated' if self._simulated else 'hardware'})")

        # Do NOT auto-start acquisition. The spectrum is acquired only after
        # the user clicks the play button (which calls start_acquisition()).

        # Minimum interval between emissions to avoid overwhelming GUI
        min_emit_interval = 0.05  # 20 Hz max update rate

        while self._running:
            # Apply any queued setting changes (integration time / laser) on
            # THIS thread, before any read, so the device is never accessed
            # concurrently from a caller thread and a get_line() here.
            self._apply_pending_settings()

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
                self._frames_done.set()
                if result is not None and len(result):
                    self.spectrum_ready.emit(
                        self._wavelengths, self._latest_spectrum, time.time())
                continue

            if self._acquiring:
                acq_start = time.time()

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
                # For real hardware, get_line() already blocks for integration_time
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
        self.wait(5000)

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

    def set_integration_time(self, time_ms: float):
        """Set integration time in milliseconds (applied on the spectrum
        thread between reads — never touches the device from the caller)."""
        self._config.integration_time = time_ms
        with QMutexLocker(self._mutex):
            self._pending_settings['integration_time_ms'] = time_ms
        logger.info(f"Integration time set to {time_ms} ms")

    def set_laser_power(self, power_percent: float):
        """Set laser power (0-100%)."""
        self._config.laser_power = max(0, min(100, power_percent))
        logger.info(f"Laser power set to {power_percent}%")

    def set_laser_power_mw(self, power_mw: float):
        """Set laser power in milliwatts (applied on the spectrum thread)."""
        self._config.laser_power_mw = power_mw
        with QMutexLocker(self._mutex):
            self._pending_settings['laser_power_mw'] = power_mw
        logger.info(f"Laser power set to {power_mw} mW")

    def set_laser(self, on: bool):
        """Turn laser on or off (applied on the spectrum thread)."""
        with QMutexLocker(self._mutex):
            self._laser_on = on
            self._pending_settings['laser_enable'] = on
        self.laser_state_changed.emit(on)
        logger.info(f"Laser {'ON' if on else 'OFF'}")

    def _apply_pending_settings(self):
        """Apply queued setting changes to the device. Runs ONLY on the spectrum
        thread (the run loop), so the USB device is never accessed concurrently
        with a get_line() read."""
        with QMutexLocker(self._mutex):
            if not self._pending_settings:
                return
            pending = self._pending_settings
            self._pending_settings = {}
        if self._simulated or self._fid is None:
            return
        try:
            if 'integration_time_ms' in pending:
                self._fid.set_integration_time_ms(pending['integration_time_ms'])
            if 'laser_power_mw' in pending:
                self._fid.set_laser_power_mW(pending['laser_power_mw'])
            if 'laser_enable' in pending:
                self._fid.set_laser_enable(pending['laser_enable'])
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to apply spectrometer settings: {e}")

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

    def _connect_spectrometer(self) -> bool:
        """Connect to the Wasatch spectrometer via SDK."""
        try:
            from wasatch.WasatchBus import WasatchBus
            from wasatch.WasatchDevice import WasatchDevice
            from wasatch.RealUSBDevice import RealUSBDevice

            logger.info("Searching for Wasatch spectrometer...")
            bus = WasatchBus(use_sim=False)

            if not bus.device_ids:
                logger.error("No Wasatch spectrometer found")
                return False

            device_id = bus.device_ids[0]
            logger.info(f"Found spectrometer: {device_id}")

            device_id.device_type = RealUSBDevice(device_id)
            self._device = WasatchDevice(device_id)
            ok = self._device.connect()

            if not ok:
                logger.error("Wasatch connect() failed")
                self._device = None
                return False

            settings = self._device.settings

            # Use wavenumbers (cm^-1) for Raman
            if settings.wavenumbers is not None and len(settings.wavenumbers) > 0:
                self._wavelengths = np.array(settings.wavenumbers, dtype=np.float64)
                logger.info(f"Wavenumber range: {self._wavelengths[0]:.1f} - "
                            f"{self._wavelengths[-1]:.1f} cm^-1 ({len(self._wavelengths)} pixels)")
            elif settings.wavelengths is not None and len(settings.wavelengths) > 0:
                self._wavelengths = np.array(settings.wavelengths, dtype=np.float64)
                logger.info(f"Wavelength range: {self._wavelengths[0]:.1f} - "
                            f"{self._wavelengths[-1]:.1f} nm ({len(self._wavelengths)} pixels)")
            else:
                logger.warning("No wavelength/wavenumber calibration from device, using pixel indices")
                self._wavelengths = np.arange(2048, dtype=np.float64)

            # Native wavelength (nm) axis for capture/export.
            if settings.wavelengths is not None and len(settings.wavelengths) > 0:
                self._wavelengths_nm = np.array(settings.wavelengths, dtype=np.float64)
            else:
                self._wavelengths_nm = None  # wavelengths_nm property will convert

            self._fid = self._device.hardware

            # Apply initial settings
            self._fid.set_integration_time_ms(self._config.integration_time)
            self._fid.set_laser_power_high_resolution(True)
            self._fid.set_laser_power_mW(self._config.laser_power_mw)

            logger.info(f"Integration time: {self._config.integration_time} ms, "
                        f"Laser power: {self._config.laser_power_mw} mW")

            self._simulated = False
            self._connected = True
            return True

        except ImportError:
            logger.error("Wasatch SDK (wasatch package) not installed")
            return False

        except Exception as e:
            logger.error(f"Spectrometer connection error: {e}")
            return False

    def _disconnect_spectrometer(self):
        """Disconnect from the spectrometer."""
        try:
            if not self._simulated and self._fid is not None:
                try:
                    self._fid.set_laser_enable(False)
                except Exception as e:
                    logger.warning(f"Failed to disable laser during disconnect: {e}")

            if not self._simulated and self._device is not None:
                try:
                    self._device.disconnect()
                except Exception as e:
                    logger.warning(f"Wasatch disconnect error: {e}")

            self._fid = None
            self._device = None
            self._connected = False

        except Exception as e:
            logger.error(f"Spectrometer disconnect error: {e}")

    def _acquire_spectrum(self) -> Optional[np.ndarray]:
        """Acquire a single spectrum."""
        try:
            if self._simulated:
                return self._generate_simulated_spectrum()

            if self._fid is None:
                return None

            # get_line() blocks for integration_time duration
            response = self._fid.get_line()
            if response is None:
                logger.warning("get_line() returned None")
                return None

            # Extract spectrum data from response
            if hasattr(response, 'data') and hasattr(response.data, 'spectrum'):
                spectrum = np.array(response.data.spectrum, dtype=np.float32)
            elif hasattr(response, 'data') and isinstance(response.data, (list, np.ndarray)):
                spectrum = np.array(response.data, dtype=np.float32)
            elif isinstance(response, (list, np.ndarray)):
                spectrum = np.array(response, dtype=np.float32)
            else:
                logger.warning(f"Unexpected get_line() response type: {type(response)}")
                return None

            return spectrum

        except Exception as e:
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
