"""
Coordinate manager for handling camera/stage offsets and safe transitions.

CRITICAL SAFETY MODULE - Incorrect coordinate handling can damage hardware.

Features:
- Camera offset configuration (wide FOV vs objective)
- Safe Z position checking
- Coordinate conversion between camera views
- Movement validation
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
from enum import Enum, auto

from raman_control.config.logging_config import get_logger

logger = get_logger('coordinate_manager')


class CameraView(Enum):
    """Available camera views."""
    WIDE_FOV = auto()
    OBJECTIVE = auto()


@dataclass
class CameraOffset:
    """Offset between two camera positions.

    The offset represents: objective_position = wide_position + offset
    """
    x: float = 0.0  # mm
    y: float = 0.0  # mm
    z: float = 0.0  # mm - CRITICAL: Z offset between focal planes


@dataclass
class SafetyLimits:
    """Safety limits for stage movement."""
    z_min: float = 0.0      # Minimum Z (closest to sample)
    z_max: float = 200.0    # Maximum Z (furthest from sample)
    z_safe: float = 50.0    # Safe Z for camera switching

    # Autofocus limits
    af_z_upper: float = 40.0  # Upper limit for autofocus search
    af_z_lower: float = 30.0  # Lower limit for autofocus search

    # Stage limits
    x_min: float = 0.0
    x_max: float = 250.0
    y_min: float = 0.0
    y_max: float = 210.0


@dataclass
class CoordinateConfig:
    """Configuration for coordinate management."""
    # Offset from wide FOV to objective camera
    wide_to_objective: CameraOffset = field(default_factory=CameraOffset)

    # Safety limits
    safety: SafetyLimits = field(default_factory=SafetyLimits)

    # Current view
    current_view: CameraView = CameraView.WIDE_FOV


class CoordinateManager:
    """
    Manages coordinate conversions and safe stage transitions.

    SAFETY CRITICAL: This class prevents objective collisions by:
    1. Validating Z positions before camera switches
    2. Converting coordinates with proper offsets
    3. Enforcing safety limits

    Usage:
        coord_mgr = CoordinateManager()

        # Before switching to objective view
        if coord_mgr.can_switch_to_objective(current_z=35.0):
            safe_z = coord_mgr.get_safe_switch_z()
            # Move to safe_z first
            # Then switch camera view
    """

    def __init__(self, config: Optional[CoordinateConfig] = None):
        self._config = config or CoordinateConfig()
        self._current_view = self._config.current_view

        logger.info(f"CoordinateManager initialized. Wide→Obj offset: "
                   f"X={self._config.wide_to_objective.x}, "
                   f"Y={self._config.wide_to_objective.y}, "
                   f"Z={self._config.wide_to_objective.z}")

    @property
    def current_view(self) -> CameraView:
        return self._current_view

    @property
    def wide_to_obj_offset(self) -> CameraOffset:
        return self._config.wide_to_objective

    @property
    def safety_limits(self) -> SafetyLimits:
        return self._config.safety

    def set_offset(self, x: float = None, y: float = None, z: float = None):
        """Update camera offset values."""
        if x is not None:
            self._config.wide_to_objective.x = x
        if y is not None:
            self._config.wide_to_objective.y = y
        if z is not None:
            self._config.wide_to_objective.z = z
        logger.info(f"Offset updated: {self._config.wide_to_objective}")

    def set_autofocus_limits(self, upper: float = None, lower: float = None):
        """Set autofocus Z limits."""
        if upper is not None:
            self._config.safety.af_z_upper = upper
        if lower is not None:
            self._config.safety.af_z_lower = lower
        logger.info(f"Autofocus limits: {self._config.safety.af_z_lower} to {self._config.safety.af_z_upper}")

    def is_z_safe(self, z: float) -> bool:
        """Check if Z position is within safe limits."""
        return self._config.safety.z_min <= z <= self._config.safety.z_max

    def is_z_safe_for_objective(self, z: float) -> bool:
        """Check if Z is safe for objective view (accounts for offset)."""
        target_z = z + self._config.wide_to_objective.z
        return target_z >= self._config.safety.z_safe

    def can_switch_to_objective(self, current_x: float, current_y: float, current_z: float) -> Tuple[bool, str]:
        """
        Check if it's safe to switch from wide FOV to objective view.

        Returns:
            (is_safe, message)
        """
        # Calculate target position after offset
        target_x = current_x + self._config.wide_to_objective.x
        target_y = current_y + self._config.wide_to_objective.y
        target_z = current_z + self._config.wide_to_objective.z

        # Check X/Y limits
        if not (self._config.safety.x_min <= target_x <= self._config.safety.x_max):
            return False, f"Target X ({target_x:.2f}) out of limits"

        if not (self._config.safety.y_min <= target_y <= self._config.safety.y_max):
            return False, f"Target Y ({target_y:.2f}) out of limits"

        # Check Z safety - objective must not be too close to sample
        if target_z < self._config.safety.z_safe:
            return False, f"Target Z ({target_z:.2f}) below safe limit ({self._config.safety.z_safe})"

        return True, "Safe to switch"

    def can_switch_to_wide(self, current_x: float, current_y: float, current_z: float) -> Tuple[bool, str]:
        """
        Check if it's safe to switch from objective to wide FOV view.

        Returns:
            (is_safe, message)
        """
        # Calculate target position (reverse offset)
        target_x = current_x - self._config.wide_to_objective.x
        target_y = current_y - self._config.wide_to_objective.y
        target_z = current_z - self._config.wide_to_objective.z

        # Check limits
        if not (self._config.safety.x_min <= target_x <= self._config.safety.x_max):
            return False, f"Target X ({target_x:.2f}) out of limits"

        if not (self._config.safety.y_min <= target_y <= self._config.safety.y_max):
            return False, f"Target Y ({target_y:.2f}) out of limits"

        if not self.is_z_safe(target_z):
            return False, f"Target Z ({target_z:.2f}) out of safe range"

        return True, "Safe to switch"

    def get_objective_position(self, wide_x: float, wide_y: float, wide_z: float) -> Tuple[float, float, float]:
        """Convert wide FOV position to objective position."""
        return (
            wide_x + self._config.wide_to_objective.x,
            wide_y + self._config.wide_to_objective.y,
            wide_z + self._config.wide_to_objective.z
        )

    def get_wide_position(self, obj_x: float, obj_y: float, obj_z: float) -> Tuple[float, float, float]:
        """Convert objective position to wide FOV position."""
        return (
            obj_x - self._config.wide_to_objective.x,
            obj_y - self._config.wide_to_objective.y,
            obj_z - self._config.wide_to_objective.z
        )

    def get_safe_switch_z(self) -> float:
        """Get safe Z position for camera switching."""
        return self._config.safety.z_safe

    def switch_view(self, to_view: CameraView) -> bool:
        """
        Record view switch. Call this AFTER successful stage movement.

        Returns:
            True if view was changed
        """
        if self._current_view != to_view:
            old_view = self._current_view
            self._current_view = to_view
            logger.info(f"View switched: {old_view.name} → {to_view.name}")
            return True
        return False

    def validate_position(self, x: float, y: float, z: float) -> Tuple[bool, str]:
        """Validate a position against safety limits."""
        if not (self._config.safety.x_min <= x <= self._config.safety.x_max):
            return False, f"X ({x:.2f}) out of limits [{self._config.safety.x_min}, {self._config.safety.x_max}]"

        if not (self._config.safety.y_min <= y <= self._config.safety.y_max):
            return False, f"Y ({y:.2f}) out of limits [{self._config.safety.y_min}, {self._config.safety.y_max}]"

        if not (self._config.safety.z_min <= z <= self._config.safety.z_max):
            return False, f"Z ({z:.2f}) out of limits [{self._config.safety.z_min}, {self._config.safety.z_max}]"

        return True, "Position valid"

    def get_autofocus_range(self) -> Tuple[float, float]:
        """Get the autofocus Z search range."""
        return (self._config.safety.af_z_lower, self._config.safety.af_z_upper)

    def is_in_autofocus_range(self, z: float) -> bool:
        """Check if Z is within autofocus range."""
        return self._config.safety.af_z_lower <= z <= self._config.safety.af_z_upper
