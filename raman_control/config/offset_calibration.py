"""
Wide-field to objective Y offset calibration curve.

The wide->objective Y offset is not constant across the stage: it varies
linearly with the WIDEFIELD X coordinate. ``offset_calibration.csv`` (kept
next to this module, in the ``config`` package) holds the measured
delta-Y at a handful of widefield X positions:

    x,delta y
    179.78,0.18
    142.16,0.84
    ...

This module fits a least-squares straight line through those points and
evaluates it at any X (interpolating between the measured points and
extrapolating beyond them).

The file is always resolved RELATIVE to this module's directory, so the
software keeps working no matter what the process working directory is,
as long as the CSV stays in ``raman_control/config/``.

Usage:
    from raman_control.config.offset_calibration import get_offset_calibration

    calib = get_offset_calibration()
    if calib.is_valid:
        dy = calib.offset_y_at(widefield_x)
"""

import csv
import os
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

from raman_control.config.logging_config import get_logger

logger = get_logger('offset_calibration')

# File name of the calibration table. It MUST stay inside the config package
# directory next to this module.
CALIBRATION_FILENAME = "offset_calibration.csv"


def default_calibration_path() -> str:
    """Absolute path to the calibration CSV inside the config package.

    Resolved from this module's own location, never from the current
    working directory, so it is independent of how the app is launched.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        CALIBRATION_FILENAME)


@dataclass(frozen=True)
class LinearFit:
    """A least-squares line ``y = slope * x + intercept``."""
    slope: float
    intercept: float
    n_points: int
    r_squared: float
    x_min: float
    x_max: float

    def value_at(self, x: float) -> float:
        """Evaluate the fitted line at ``x`` (interpolate or extrapolate)."""
        return self.slope * x + self.intercept

    def is_extrapolated(self, x: float) -> bool:
        """True if ``x`` lies outside the calibrated X range."""
        return x < self.x_min or x > self.x_max

    def describe(self) -> str:
        return (f"dy = {self.slope:+.6f} * x {self.intercept:+.6f} "
                f"(n={self.n_points}, R^2={self.r_squared:.4f}, "
                f"x range {self.x_min:.2f}..{self.x_max:.2f} mm)")


def fit_linear(points: List[Tuple[float, float]]) -> Optional[LinearFit]:
    """Least-squares fit of a straight line through ``(x, y)`` points.

    Returns None if there are fewer than two points or all X values are
    identical (the fit would be degenerate).
    """
    n = len(points)
    if n < 2:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return None  # all X identical -> vertical line, not fittable
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))

    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    syy = sum((y - mean_y) ** 2 for y in ys)
    if syy > 0.0:
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        r_squared = 1.0 - ss_res / syy
    else:
        r_squared = 1.0  # all Y identical: a horizontal line fits exactly

    return LinearFit(
        slope=slope,
        intercept=intercept,
        n_points=n,
        r_squared=r_squared,
        x_min=min(xs),
        x_max=max(xs),
    )


def load_calibration_points(filepath: str) -> List[Tuple[float, float]]:
    """Read ``(x, delta_y)`` pairs from a two-column CSV.

    A header row is detected and skipped automatically (any row whose first
    two cells are not both numbers is ignored), so the file can carry the
    ``x,delta y`` header. Blank lines and short rows are skipped too.
    """
    points: List[Tuple[float, float]] = []
    with open(filepath, 'r', newline='') as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                points.append((float(row[0].strip()), float(row[1].strip())))
            except ValueError:
                continue  # header or comment row
    return points


class OffsetCalibrationCurve:
    """Linear wide->objective Y offset as a function of widefield X.

    Loads ``offset_calibration.csv`` from the config package on first use.
    If the file is missing or unusable, ``is_valid`` is False and callers
    fall back to the fixed ``StageConfig.calib_wide_to_obj_y`` offset.

    Thread-safe: the fit is immutable once loaded and ``reload()`` swaps it
    under a lock, so worker threads can call ``offset_y_at()`` freely.
    """

    def __init__(self, filepath: Optional[str] = None):
        self._filepath = filepath or default_calibration_path()
        self._lock = threading.Lock()
        self._fit: Optional[LinearFit] = None
        self._error: str = ""
        self.reload()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def reload(self) -> bool:
        """(Re)read the CSV and refit. Returns True if a usable fit exists."""
        fit: Optional[LinearFit] = None
        error = ""
        try:
            if not os.path.isfile(self._filepath):
                error = f"Calibration file not found: {self._filepath}"
            else:
                points = load_calibration_points(self._filepath)
                fit = fit_linear(points)
                if fit is None:
                    error = (f"Need at least 2 points with distinct X in "
                             f"{self._filepath} (got {len(points)})")
        except Exception as e:  # unreadable / malformed file
            error = f"Failed to read {self._filepath}: {e}"

        with self._lock:
            self._fit = fit
            self._error = error

        if fit is not None:
            logger.info(f"Offset calibration curve loaded: {fit.describe()}")
        else:
            logger.warning(f"Offset calibration curve unavailable. {error} "
                           f"Falling back to the fixed Y offset.")
        return fit is not None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def filepath(self) -> str:
        return self._filepath

    @property
    def fit(self) -> Optional[LinearFit]:
        with self._lock:
            return self._fit

    @property
    def is_valid(self) -> bool:
        with self._lock:
            return self._fit is not None

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    def offset_y_at(self, widefield_x: float) -> Optional[float]:
        """Fitted wide->objective Y offset (mm) at a WIDEFIELD X position.

        Interpolates between the calibrated X points and extrapolates the
        same line outside them. Returns None when no fit is available.
        """
        with self._lock:
            fit = self._fit
        if fit is None:
            return None
        return fit.value_at(widefield_x)

    def describe(self) -> str:
        """One-line human readable summary for logs / tooltips."""
        fit = self.fit
        if fit is None:
            return f"unavailable ({self.error})"
        return fit.describe()


# ----------------------------------------------------------------------
# Process-wide singleton (the CSV is small and read once)
# ----------------------------------------------------------------------

_singleton: Optional[OffsetCalibrationCurve] = None
_singleton_lock = threading.Lock()


def get_offset_calibration() -> OffsetCalibrationCurve:
    """Return the shared calibration curve, loading it on first call."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = OffsetCalibrationCurve()
        return _singleton
