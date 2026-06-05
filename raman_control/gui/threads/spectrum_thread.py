"""
PyQt6 thread for spectrometer acquisition using Wasatch Photonics SDK.

Provides non-blocking spectrum acquisition with:
- Real Wasatch SDK integration (WasatchBus/WasatchDevice)
- Debug mode fallback to simulated spectra
- Integration time control
- Laser power control (mW)
- Thread-safe spectrum delivery via Qt signals
"""

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
        """Set integration time in milliseconds."""
        self._config.integration_time = time_ms
        if not self._simulated and self._fid is not None:
            try:
                self._fid.set_integration_time_ms(time_ms)
            except Exception as e:
                logger.error(f"Failed to set integration time on hardware: {e}")
        logger.info(f"Integration time set to {time_ms} ms")

    def set_laser_power(self, power_percent: float):
        """Set laser power (0-100%)."""
        self._config.laser_power = max(0, min(100, power_percent))
        logger.info(f"Laser power set to {power_percent}%")

    def set_laser_power_mw(self, power_mw: float):
        """Set laser power in milliwatts."""
        self._config.laser_power_mw = power_mw
        if not self._simulated and self._fid is not None:
            try:
                self._fid.set_laser_power_mW(power_mw)
            except Exception as e:
                logger.error(f"Failed to set laser power on hardware: {e}")
        logger.info(f"Laser power set to {power_mw} mW")

    def set_laser(self, on: bool):
        """Turn laser on or off."""
        with QMutexLocker(self._mutex):
            self._laser_on = on
        if not self._simulated and self._fid is not None:
            try:
                self._fid.set_laser_enable(on)
            except Exception as e:
                logger.error(f"Failed to set laser state on hardware: {e}")
        self.laser_state_changed.emit(on)
        logger.info(f"Laser {'ON' if on else 'OFF'}")

    def get_latest_spectrum(self) -> Optional[np.ndarray]:
        """Get the most recently acquired spectrum (thread-safe)."""
        with QMutexLocker(self._mutex):
            if self._latest_spectrum is not None:
                return self._latest_spectrum.copy()
            return None

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
