"""
Sampling pattern generator from segmentation masks.

Converts segmented objects to physical stage coordinates for acquisition.
Uses center-relative coordinate model: the center of the captured image
corresponds to the stage position at capture time.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np

from raman_control.config.logging_config import get_logger

logger = get_logger('pattern_generator')


@dataclass
class SamplePosition:
    """A single sample position for acquisition."""
    id: int
    pixel_x: float  # pixel coordinate (subpixel)
    pixel_y: float
    stage_x: float  # widefield stage coordinate (mm)
    stage_y: float
    stage_z: float
    obj_stage_x: float = 0.0  # objective stage coordinate (mm, after offset)
    obj_stage_y: float = 0.0
    obj_stage_z: float = 0.0
    area: int = 0
    label: str = ""
    coordinates_order: Optional[Dict[str, int]] = None  # e.g. {'row': 1, 'col': 2}


@dataclass
class CalibrationParams:
    """Center-relative image-to-stage calibration.

    The image center corresponds to (capture_stage_x, capture_stage_y).
    Pixel offsets from center are scaled by pixel_size and optionally
    Y-inverted (image Y down, printer Y up).
    """
    pixel_size_x_um: float = 2.82   # micrometers per pixel
    pixel_size_y_um: float = 2.82
    image_width: int = 1440
    image_height: int = 1080
    capture_stage_x: float = 0.0    # stage position at capture time (mm)
    capture_stage_y: float = 0.0
    capture_stage_z: float = 0.0
    invert_y: bool = True            # image Y down, printer Y up


class PatternGenerator:
    """
    Generate sampling patterns from segmentation results.

    Converts pixel coordinates from segmentation to physical stage coordinates
    using a center-relative model matching the old noodlepy behavior.

    Usage:
        cal = CalibrationParams(
            pixel_size_x_um=2.82, pixel_size_y_um=2.82,
            image_width=1440, image_height=1080,
            capture_stage_x=100.0, capture_stage_y=100.0,
            capture_stage_z=35.0,
        )
        gen = PatternGenerator(cal)
        positions = gen.from_centroids(centroids, coordinates_orders=orders)
    """

    def __init__(self, calibration: Optional[CalibrationParams] = None):
        self._calibration = calibration or CalibrationParams()
        self._positions: List[SamplePosition] = []

    @property
    def calibration(self) -> CalibrationParams:
        return self._calibration

    def pixel_to_stage(self, pixel_x: float, pixel_y: float) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates to stage coordinates (center-relative).

        Steps (matching old noodlepy convert_pixel_position_to_relative_distance
        + calculate_sampling_point_position_in_absolute_coordinate):
          1. Center: px - image_width/2, py - image_height/2
          2. Scale: * pixel_size_um
          3. Flip Y: * -1 (if invert_y, because image Y down, printer Y up)
          4. um -> mm: / 1000
          5. Add capture stage position

        Returns:
            (stage_x, stage_y, stage_z) in mm
        """
        cal = self._calibration

        # 1. Offset from image center (pixels)
        dx_px = pixel_x - cal.image_width / 2.0
        dy_px = pixel_y - cal.image_height / 2.0

        # 2. Scale to micrometers
        dx_um = dx_px * cal.pixel_size_x_um
        dy_um = dy_px * cal.pixel_size_y_um

        # 3. Flip Y if needed (image Y down, printer Y up)
        if cal.invert_y:
            dy_um = -dy_um

        # 4. Convert to mm
        dx_mm = dx_um / 1000.0
        dy_mm = dy_um / 1000.0

        # 5. Add capture stage position
        stage_x = round(cal.capture_stage_x + dx_mm, 4)
        stage_y = round(cal.capture_stage_y + dy_mm, 4)
        stage_z = cal.capture_stage_z

        return stage_x, stage_y, stage_z

    def stage_to_pixel(self, stage_x: float, stage_y: float) -> Tuple[float, float]:
        """
        Convert stage coordinates back to pixel coordinates.

        Returns:
            (pixel_x, pixel_y) in image coordinates
        """
        cal = self._calibration

        dx_mm = stage_x - cal.capture_stage_x
        dy_mm = stage_y - cal.capture_stage_y

        dx_um = dx_mm * 1000.0
        dy_um = dy_mm * 1000.0

        if cal.invert_y:
            dy_um = -dy_um

        dx_px = dx_um / cal.pixel_size_x_um
        dy_px = dy_um / cal.pixel_size_y_um

        pixel_x = dx_px + cal.image_width / 2.0
        pixel_y = dy_px + cal.image_height / 2.0

        return pixel_x, pixel_y

    def from_centroids(
        self,
        centroids: List[Tuple[float, float]],
        areas: Optional[List[int]] = None,
        coordinates_orders: Optional[List[Dict[str, int]]] = None,
    ) -> List[SamplePosition]:
        """
        Generate sample positions from centroid list.

        Args:
            centroids: List of (x, y) pixel coordinates (may be float for subpixel)
            areas: Optional list of object areas
            coordinates_orders: Optional list of dicts (e.g. {'row':1,'col':2})

        Returns:
            List of SamplePosition objects
        """
        self._positions = []

        for i, (px, py) in enumerate(centroids):
            sx, sy, sz = self.pixel_to_stage(px, py)

            pos = SamplePosition(
                id=i + 1,
                pixel_x=px,
                pixel_y=py,
                stage_x=sx,
                stage_y=sy,
                stage_z=sz,
                area=areas[i] if areas and i < len(areas) else 0,
                label=f"Sample_{i+1}",
                coordinates_order=coordinates_orders[i] if coordinates_orders and i < len(coordinates_orders) else None,
            )
            self._positions.append(pos)

        logger.info(f"Generated {len(self._positions)} sample positions from centroids")
        return self._positions

    def from_masks(self, mask_dicts: List[Dict[str, Any]]) -> List[SamplePosition]:
        """
        Generate sample positions from mask dictionaries (SAM format).

        Args:
            mask_dicts: List of dictionaries with 'segmentation', 'area', etc.

        Returns:
            List of SamplePosition objects
        """
        centroids = []
        areas = []

        for mask_dict in mask_dicts:
            mask = mask_dict.get('segmentation')
            if mask is not None:
                y_indices, x_indices = np.where(mask)
                if len(x_indices) > 0:
                    cx = float(np.mean(x_indices))
                    cy = float(np.mean(y_indices))
                    centroids.append((cx, cy))
                    areas.append(mask_dict.get('area', int(np.sum(mask))))

        return self.from_centroids(centroids, areas)

    def add_position(
        self,
        pixel_x: float,
        pixel_y: float,
        label: str = ""
    ) -> SamplePosition:
        """Manually add a sample position."""
        sx, sy, sz = self.pixel_to_stage(pixel_x, pixel_y)

        pos = SamplePosition(
            id=len(self._positions) + 1,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            stage_x=sx,
            stage_y=sy,
            stage_z=sz,
            label=label or f"Sample_{len(self._positions)+1}"
        )
        self._positions.append(pos)
        return pos

    def remove_position(self, position_id: int) -> bool:
        """Remove a position by ID."""
        for i, pos in enumerate(self._positions):
            if pos.id == position_id:
                self._positions.pop(i)
                return True
        return False

    def clear_positions(self):
        """Clear all positions."""
        self._positions = []

    def get_positions(self) -> List[SamplePosition]:
        """Get all sample positions."""
        return self._positions.copy()

    def get_position_list(self) -> List[Tuple[float, float, float]]:
        """Get positions as list of (x, y, z) tuples for acquisition."""
        return [(p.stage_x, p.stage_y, p.stage_z) for p in self._positions]

    def get_position_dict_list(self) -> List[Dict[str, Any]]:
        """Get positions as list of dictionaries."""
        return [
            {
                'id': p.id,
                'pixel_x': p.pixel_x,
                'pixel_y': p.pixel_y,
                'stage_x': p.stage_x,
                'stage_y': p.stage_y,
                'stage_z': p.stage_z,
                'obj_stage_x': p.obj_stage_x,
                'obj_stage_y': p.obj_stage_y,
                'obj_stage_z': p.obj_stage_z,
                'area': p.area,
                'label': p.label,
                'coordinates_order': p.coordinates_order,
            }
            for p in self._positions
        ]

    def sort_by_distance(self, start_x: float = 0, start_y: float = 0):
        """Sort positions by distance from a starting point (nearest neighbor)."""
        if not self._positions:
            return

        sorted_positions = []
        remaining = self._positions.copy()
        current_x, current_y = start_x, start_y

        while remaining:
            min_dist = float('inf')
            nearest_idx = 0
            for i, pos in enumerate(remaining):
                dist = np.sqrt((pos.stage_x - current_x)**2 + (pos.stage_y - current_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i

            nearest = remaining.pop(nearest_idx)
            sorted_positions.append(nearest)
            current_x, current_y = nearest.stage_x, nearest.stage_y

        self._positions = sorted_positions
        logger.info("Positions sorted by distance")

    def filter_by_area(self, min_area: int = 0, max_area: int = float('inf')):
        """Filter positions by object area."""
        self._positions = [
            p for p in self._positions
            if min_area <= p.area <= max_area
        ]
        logger.info(f"{len(self._positions)} positions after area filter")
