"""
Autofocus module for coarse and fine focusing.

Features:
- Coarse autofocus using PRUSA stepper Z-axis
- Fine autofocus using NanoDrive piezo
- Multiple focus metrics (contrast, spectrometer intensity)
- Configurable search range and step sizes
"""

from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List
from enum import Enum, auto
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from raman_control.config.logging_config import get_logger

logger = get_logger('autofocus')


class FocusMetric(Enum):
    """Available focus quality metrics."""
    CONTRAST = auto()      # Image contrast (Laplacian variance)
    INTENSITY = auto()     # Spectrometer peak intensity
    GRADIENT = auto()      # Image gradient magnitude
    COMBINED = auto()      # Combined image + spectrometer


@dataclass
class AutofocusConfig:
    """Configuration for autofocus routines."""
    # Coarse focus (PRUSA stepper)
    coarse_step_um: float = 50.0      # Step size for coarse search
    coarse_range_um: float = 2000.0   # Total range for coarse search

    # Fine focus (NanoDrive piezo)
    fine_step_um: float = 1.0         # Step size for fine search
    fine_range_um: float = 50.0       # Total range for fine search

    # Z limits (set from coordinate manager)
    z_upper_mm: float = 40.0          # Upper limit
    z_lower_mm: float = 30.0          # Lower limit

    # Focus metric
    metric: FocusMetric = FocusMetric.CONTRAST

    # Algorithm parameters
    timeout_seconds: float = 60.0     # Maximum time for autofocus
    min_improvement: float = 0.01     # Minimum improvement to continue
    settling_time_ms: float = 50.0    # Wait time after each move


class AutofocusEngine(QObject):
    """
    Autofocus engine for finding optimal focus position.

    Signals:
        progress: (current_step, total_steps, current_z, metric_value)
        completed: (success, best_z, best_metric)
        error: (error_message)

    Usage:
        af = AutofocusEngine(config)
        af.completed.connect(self.on_focus_found)

        # For coarse focus with stepper
        af.start_coarse_focus(
            move_z_callback=self.motion_thread.request_move_to,
            get_metric_callback=self.get_focus_metric
        )

        # For fine focus with piezo
        af.start_fine_focus(
            move_z_callback=self.piezo.move_to_z,
            get_metric_callback=self.get_spectrometer_intensity
        )
    """

    # Signals
    progress = pyqtSignal(int, int, float, float)  # step, total, z, metric
    completed = pyqtSignal(bool, float, float)      # success, best_z, best_metric
    error = pyqtSignal(str)

    def __init__(self, config: Optional[AutofocusConfig] = None, parent=None):
        super().__init__(parent)
        self._config = config or AutofocusConfig()
        self._running = False
        self._abort = False

        # Results storage
        self._z_positions: List[float] = []
        self._metric_values: List[float] = []

    @property
    def config(self) -> AutofocusConfig:
        return self._config

    def set_z_limits(self, upper_mm: float, lower_mm: float):
        """Set Z search limits."""
        self._config.z_upper_mm = upper_mm
        self._config.z_lower_mm = lower_mm
        logger.info(f"Autofocus Z limits set: {lower_mm} to {upper_mm} mm")

    def abort(self):
        """Abort current autofocus operation."""
        self._abort = True
        logger.warning("Autofocus aborted by user")

    def start_coarse_focus(
        self,
        move_z_callback: Callable[[float], bool],
        get_metric_callback: Callable[[], float],
        start_z: Optional[float] = None
    ):
        """
        Start coarse autofocus using stepper.

        Args:
            move_z_callback: Function to move Z to absolute position (mm), returns success
            get_metric_callback: Function to get current focus metric value
            start_z: Starting Z position (if None, starts at current)
        """
        self._running = True
        self._abort = False
        self._z_positions = []
        self._metric_values = []

        try:
            # Calculate search positions
            z_lower = self._config.z_lower_mm
            z_upper = self._config.z_upper_mm
            step_mm = self._config.coarse_step_um / 1000.0

            positions = np.arange(z_lower, z_upper + step_mm, step_mm)
            total_steps = len(positions)

            logger.info(f"Starting coarse autofocus: {z_lower} to {z_upper} mm, {total_steps} steps")

            best_z = z_lower
            best_metric = float('-inf')

            for i, z in enumerate(positions):
                if self._abort:
                    self.completed.emit(False, best_z, best_metric)
                    return

                # Move to position
                if not move_z_callback(z):
                    self.error.emit(f"Failed to move to Z={z}")
                    continue

                # Wait for settling
                import time
                time.sleep(self._config.settling_time_ms / 1000.0)

                # Get metric
                metric = get_metric_callback()

                self._z_positions.append(z)
                self._metric_values.append(metric)

                # Update best
                if metric > best_metric:
                    best_metric = metric
                    best_z = z

                self.progress.emit(i + 1, total_steps, z, metric)

            # Move to best position
            move_z_callback(best_z)

            logger.info(f"Coarse autofocus complete: best Z = {best_z:.3f} mm, metric = {best_metric:.4f}")
            self.completed.emit(True, best_z, best_metric)

        except Exception as e:
            logger.error(f"Coarse autofocus error: {e}")
            self.error.emit(str(e))

        finally:
            self._running = False

    def start_fine_focus(
        self,
        move_z_callback: Callable[[float], bool],
        get_metric_callback: Callable[[], float],
        center_z: Optional[float] = None
    ):
        """
        Start fine autofocus using piezo.

        Args:
            move_z_callback: Function to move piezo Z (um), returns success
            get_metric_callback: Function to get current focus metric value
            center_z: Center position for search (um)
        """
        self._running = True
        self._abort = False
        self._z_positions = []
        self._metric_values = []

        try:
            # Calculate search range around center
            center = center_z if center_z is not None else 50.0  # Default center
            half_range = self._config.fine_range_um / 2.0
            step = self._config.fine_step_um

            z_lower = center - half_range
            z_upper = center + half_range

            positions = np.arange(z_lower, z_upper + step, step)
            total_steps = len(positions)

            logger.info(f"Starting fine autofocus: {z_lower} to {z_upper} um, {total_steps} steps")

            best_z = center
            best_metric = float('-inf')

            for i, z in enumerate(positions):
                if self._abort:
                    self.completed.emit(False, best_z, best_metric)
                    return

                # Move piezo
                if not move_z_callback(z):
                    self.error.emit(f"Failed to move piezo to Z={z}")
                    continue

                # Wait for settling (piezo is fast)
                import time
                time.sleep(self._config.settling_time_ms / 1000.0)

                # Get metric
                metric = get_metric_callback()

                self._z_positions.append(z)
                self._metric_values.append(metric)

                # Update best
                if metric > best_metric:
                    best_metric = metric
                    best_z = z

                self.progress.emit(i + 1, total_steps, z, metric)

            # Move to best position
            move_z_callback(best_z)

            logger.info(f"Fine autofocus complete: best Z = {best_z:.3f} um, metric = {best_metric:.4f}")
            self.completed.emit(True, best_z, best_metric)

        except Exception as e:
            logger.error(f"Fine autofocus error: {e}")
            self.error.emit(str(e))

        finally:
            self._running = False

    def get_results(self) -> Tuple[List[float], List[float]]:
        """Get recorded Z positions and metric values from last run."""
        return self._z_positions.copy(), self._metric_values.copy()

    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        """Calculate image contrast using Laplacian variance."""
        import cv2
        if image is None:
            return 0.0
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()

    @staticmethod
    def calculate_gradient(image: np.ndarray) -> float:
        """Calculate image sharpness using gradient magnitude."""
        import cv2
        if image is None:
            return 0.0
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(gx**2 + gy**2).mean()

    @staticmethod
    def calculate_intensity(spectrum: np.ndarray) -> float:
        """Calculate peak intensity from spectrum."""
        if spectrum is None or len(spectrum) == 0:
            return 0.0
        return np.max(spectrum)
