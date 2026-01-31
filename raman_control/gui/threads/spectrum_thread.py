"""
PyQt6 thread for spectrometer acquisition.

Provides non-blocking spectrum acquisition with:
- Continuous acquisition mode
- Integration time control
- Laser power control
- Thread-safe spectrum delivery via Qt signals
"""

from typing import Optional, Tuple
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

        spectrometer.start_acquisition()
        # ... receive spectra via signal
        spectrometer.stop_acquisition()

        spectrometer.stop()
    """

    spectrum_ready = pyqtSignal(np.ndarray, np.ndarray, float)  # wavelengths, intensities, timestamp
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)
    laser_state_changed = pyqtSignal(bool)

    def __init__(self, config: Optional[SpectrometerConfig] = None, parent=None):
        super().__init__(parent)

        self._config = config or SpectrometerConfig()
        self._mutex = QMutex()

        # State
        self._running = False
        self._acquiring = False
        self._connected = False
        self._laser_on = False

        # Wavelength axis (typical Raman range)
        self._wavelengths = np.linspace(200, 3500, 2048)  # cm^-1

        # Latest spectrum storage for on-demand access
        self._latest_spectrum: Optional[np.ndarray] = None

        # Simulated spectrometer (replace with actual SDK)
        self._simulated = True

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
    def wavelengths(self) -> np.ndarray:
        return self._wavelengths.copy()

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
        logger.info("Spectrometer connected")

        # Acquisition loop
        import time

        # Minimum interval between emissions to avoid overwhelming GUI
        min_emit_interval = 0.05  # 20 Hz max update rate

        while self._running:
            if self._acquiring:
                acq_start = time.time()

                # Acquire spectrum
                spectrum = self._acquire_spectrum()

                if spectrum is not None:
                    with QMutexLocker(self._mutex):
                        self._latest_spectrum = spectrum.copy()
                    self.spectrum_ready.emit(
                        self._wavelengths,
                        spectrum,
                        acq_start
                    )

                # Wait for next acquisition (based on integration time)
                # But ensure minimum interval to not overwhelm GUI
                elapsed = time.time() - acq_start
                wait_time = max(min_emit_interval, (self._config.integration_time / 1000.0) - elapsed)
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
        # TODO: Apply to actual spectrometer SDK
        logger.info(f"Integration time set to {time_ms} ms")

    def set_laser_power(self, power_percent: float):
        """Set laser power (0-100%)."""
        self._config.laser_power = max(0, min(100, power_percent))
        # TODO: Apply to actual spectrometer SDK
        logger.info(f"Laser power set to {power_percent}%")

    def set_laser_power_mw(self, power_mw: float):
        """Set laser power in milliwatts."""
        # TODO: Apply to actual spectrometer SDK
        # The actual conversion depends on your hardware
        logger.info(f"Laser power set to {power_mw} mW")

    def set_laser(self, on: bool):
        """Turn laser on or off."""
        with QMutexLocker(self._mutex):
            self._laser_on = on
        # TODO: Apply to actual spectrometer SDK
        self.laser_state_changed.emit(on)
        logger.info(f"Laser {'ON' if on else 'OFF'}")

    def set_averaging(self, num_averages: int):
        """Set number of spectra to average."""
        self._config.averaging = max(1, num_averages)
        logger.info(f"Averaging set to {num_averages}")

    def get_latest_spectrum(self) -> Optional[np.ndarray]:
        """Get the most recently acquired spectrum (thread-safe)."""
        with QMutexLocker(self._mutex):
            if self._latest_spectrum is not None:
                return self._latest_spectrum.copy()
            return None

    def _connect_spectrometer(self) -> bool:
        """Connect to the spectrometer hardware."""
        try:
            if self._simulated:
                self._connected = True
                return True

            # TODO: Replace with actual Wasatch SDK initialization
            # from wasatch import WasatchDevice
            # self._device = WasatchDevice.connect()
            # self._wavelengths = self._device.get_wavelengths()

            self._connected = True
            return True

        except Exception as e:
            logger.error(f"Spectrometer connection error: {e}")
            return False

    def _disconnect_spectrometer(self):
        """Disconnect from the spectrometer."""
        try:
            self.set_laser(False)

            if not self._simulated:
                # TODO: Cleanup actual spectrometer SDK
                # self._device.disconnect()
                pass

            self._connected = False

        except Exception as e:
            logger.error(f"Spectrometer disconnect error: {e}")

    def _acquire_spectrum(self) -> Optional[np.ndarray]:
        """Acquire a single spectrum."""
        try:
            if self._simulated:
                return self._generate_simulated_spectrum()

            # TODO: Replace with actual spectrometer SDK acquisition
            # return self._device.get_spectrum()

            return None

        except Exception as e:
            logger.error(f"Spectrum acquisition error: {e}")
            return None

    def _generate_simulated_spectrum(self) -> np.ndarray:
        """Generate a simulated Raman spectrum."""
        import time

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
