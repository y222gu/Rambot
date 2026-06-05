"""
Coordinate manager for handling camera/stage offsets and safe transitions.

CRITICAL SAFETY MODULE - Incorrect coordinate handling can damage hardware.

All parameters (offsets, safety limits) live in StageConfig (hardware_config.py)
as the single source of truth. This module reads from that config directly.
"""

from typing import Tuple
from enum import Enum, auto

from raman_control.config.logging_config import get_logger
from raman_control.config.hardware_config import StageConfig

logger = get_logger('coordinate_manager')


class CameraView(Enum):
    """Available camera views."""
    WIDE_FOV = auto()
    OBJECTIVE = auto()


class CoordinateManager:
    """
    Manages coordinate conversions and safe stage transitions.

    Reads offsets and safety limits directly from StageConfig so there
    is only one copy of each parameter.

    Usage:
        coord_mgr = CoordinateManager(stage_config)

        # Before switching to objective view
        if coord_mgr.can_switch_to_objective(x, y, z):
            safe_z = coord_mgr.get_safe_switch_z()
    """

    def __init__(self, stage_config: StageConfig, initial_view: CameraView = CameraView.WIDE_FOV):
        self._stage = stage_config
        self._current_view = initial_view

        logger.info(f"CoordinateManager initialized. Wide->Obj offset: "
                   f"X={self._stage.calib_wide_to_obj_x}, "
                   f"Y={self._stage.calib_wide_to_obj_y}, "
                   f"Z={self._stage.calib_wide_to_obj_z}")

    @property
    def current_view(self) -> CameraView:
        return self._current_view

    @property
    def offset_x(self) -> float:
        return self._stage.calib_wide_to_obj_x

    @property
    def offset_y(self) -> float:
        return self._stage.calib_wide_to_obj_y

    @property
    def offset_z(self) -> float:
        return self._stage.calib_wide_to_obj_z

    @property
    def stage_config(self) -> StageConfig:
        """Direct access to the underlying StageConfig (single source of truth)."""
        return self._stage

    def set_offset(self, x: float = None, y: float = None, z: float = None):
        """Update camera offset values in the underlying StageConfig."""
        if x is not None:
            self._stage.calib_wide_to_obj_x = x
        if y is not None:
            self._stage.calib_wide_to_obj_y = y
        if z is not None:
            self._stage.calib_wide_to_obj_z = z
        logger.info(f"Offset updated: X={self._stage.calib_wide_to_obj_x}, "
                   f"Y={self._stage.calib_wide_to_obj_y}, "
                   f"Z={self._stage.calib_wide_to_obj_z}")

    def set_autofocus_limits(self, upper: float = None, lower: float = None):
        """Set autofocus Z limits in the underlying StageConfig."""
        if upper is not None:
            self._stage.af_z_upper = upper
        if lower is not None:
            self._stage.af_z_lower = lower
        logger.info(f"Autofocus limits: {self._stage.af_z_lower} to {self._stage.af_z_upper}")

    def is_z_safe(self, z: float) -> bool:
        """Check if Z position is within safe limits."""
        return self._stage.z_min <= z <= self._stage.z_max

    def is_z_safe_for_objective(self, z: float) -> bool:
        """Check if Z is safe for objective view (accounts for offset)."""
        target_z = z + self._stage.calib_wide_to_obj_z
        return target_z >= self._stage.z_safe

    def can_switch_to_objective(self, current_x: float, current_y: float, current_z: float) -> Tuple[bool, str]:
        """
        Check if it's safe to switch from wide FOV to objective view.

        Returns:
            (is_safe, message)
        """
        target_x = current_x + self._stage.calib_wide_to_obj_x
        target_y = current_y + self._stage.calib_wide_to_obj_y
        target_z = current_z + self._stage.calib_wide_to_obj_z

        if not (self._stage.x_min <= target_x <= self._stage.x_max):
            return False, f"Target X ({target_x:.2f}) out of limits"

        if not (self._stage.y_min <= target_y <= self._stage.y_max):
            return False, f"Target Y ({target_y:.2f}) out of limits"

        if target_z < self._stage.z_safe:
            return False, f"Target Z ({target_z:.2f}) below safe limit ({self._stage.z_safe})"

        return True, "Safe to switch"

    def can_switch_to_wide(self, current_x: float, current_y: float, current_z: float) -> Tuple[bool, str]:
        """
        Check if it's safe to switch from objective to wide FOV view.

        Returns:
            (is_safe, message)
        """
        target_x = current_x - self._stage.calib_wide_to_obj_x
        target_y = current_y - self._stage.calib_wide_to_obj_y
        target_z = current_z - self._stage.calib_wide_to_obj_z

        if not (self._stage.x_min <= target_x <= self._stage.x_max):
            return False, f"Target X ({target_x:.2f}) out of limits"

        if not (self._stage.y_min <= target_y <= self._stage.y_max):
            return False, f"Target Y ({target_y:.2f}) out of limits"

        if not self.is_z_safe(target_z):
            return False, f"Target Z ({target_z:.2f}) out of safe range"

        return True, "Safe to switch"

    def logical_to_physical(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Convert logical (sample-centric) coords to physical stage coords.

        Logical coordinates are always in the wide FOV frame. When in
        objective mode the camera offset is added so the stage moves to
        place the objective over the same sample point.
        """
        if self._current_view == CameraView.OBJECTIVE:
            return (
                x + self._stage.calib_wide_to_obj_x,
                y + self._stage.calib_wide_to_obj_y,
                z + self._stage.calib_wide_to_obj_z,
            )
        return (x, y, z)

    def physical_to_logical(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Convert physical stage coords to logical (sample-centric) coords.

        Removes the camera offset when in objective mode so the user
        always sees positions in the wide FOV reference frame.
        """
        if self._current_view == CameraView.OBJECTIVE:
            return (
                x - self._stage.calib_wide_to_obj_x,
                y - self._stage.calib_wide_to_obj_y,
                z - self._stage.calib_wide_to_obj_z,
            )
        return (x, y, z)

    def get_objective_position(self, wide_x: float, wide_y: float, wide_z: float) -> Tuple[float, float, float]:
        """Convert wide FOV position to objective position."""
        return (
            wide_x + self._stage.calib_wide_to_obj_x,
            wide_y + self._stage.calib_wide_to_obj_y,
            wide_z + self._stage.calib_wide_to_obj_z,
        )

    def get_wide_position(self, obj_x: float, obj_y: float, obj_z: float) -> Tuple[float, float, float]:
        """Convert objective position to wide FOV position."""
        return (
            obj_x - self._stage.calib_wide_to_obj_x,
            obj_y - self._stage.calib_wide_to_obj_y,
            obj_z - self._stage.calib_wide_to_obj_z,
        )

    def get_safe_switch_z(self) -> float:
        """Get safe Z position for camera switching."""
        return self._stage.z_safe

    def switch_view(self, to_view: CameraView) -> bool:
        """
        Record view switch. Call this AFTER successful stage movement.

        Returns:
            True if view was changed
        """
        if self._current_view != to_view:
            old_view = self._current_view
            self._current_view = to_view
            logger.info(f"View switched: {old_view.name} -> {to_view.name}")
            return True
        return False

    def validate_position(self, x: float, y: float, z: float) -> Tuple[bool, str]:
        """Validate a position against safety limits."""
        if not (self._stage.x_min <= x <= self._stage.x_max):
            return False, f"X ({x:.2f}) out of limits [{self._stage.x_min}, {self._stage.x_max}]"

        if not (self._stage.y_min <= y <= self._stage.y_max):
            return False, f"Y ({y:.2f}) out of limits [{self._stage.y_min}, {self._stage.y_max}]"

        if not (self._stage.z_min <= z <= self._stage.z_max):
            return False, f"Z ({z:.2f}) out of limits [{self._stage.z_min}, {self._stage.z_max}]"

        return True, "Position valid"

    def get_autofocus_range(self) -> Tuple[float, float]:
        """Get the autofocus Z search range."""
        return (self._stage.af_z_lower, self._stage.af_z_upper)

    def is_in_autofocus_range(self, z: float) -> bool:
        """Check if Z is within autofocus range."""
        return self._stage.af_z_lower <= z <= self._stage.af_z_upper
