"""
Sampling pattern generator from segmentation masks.

Converts segmented objects to physical stage coordinates for acquisition.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from raman_control.config.logging_config import get_logger

logger = get_logger('pattern_generator')


@dataclass
class SamplePosition:
    """A single sample position for acquisition."""
    id: int
    pixel_x: int
    pixel_y: int
    stage_x: float
    stage_y: float
    stage_z: float
    area: int = 0
    label: str = ""


@dataclass
class CalibrationParams:
    """Image to stage coordinate calibration parameters."""
    # Pixel to mm conversion
    pixels_per_mm_x: float = 100.0
    pixels_per_mm_y: float = 100.0

    # Image origin in stage coordinates
    origin_stage_x: float = 0.0
    origin_stage_y: float = 0.0

    # Z position for sampling (from autofocus)
    sample_z: float = 35.0

    # Axis inversion (if image Y increases downward but stage Y increases upward)
    invert_x: bool = False
    invert_y: bool = True

    # Image dimensions for coordinate calculation
    image_width: int = 1024
    image_height: int = 768


class PatternGenerator:
    """
    Generate sampling patterns from segmentation results.

    Converts pixel coordinates from segmentation to physical stage coordinates.

    Usage:
        gen = PatternGenerator()
        gen.set_calibration(pixels_per_mm=100, origin=(50, 50))

        # From centroids
        positions = gen.from_centroids(centroids)

        # From mask dictionaries
        positions = gen.from_masks(mask_dicts)

        # Export for acquisition
        position_list = gen.get_position_list()
    """

    def __init__(self, calibration: Optional[CalibrationParams] = None):
        self._calibration = calibration or CalibrationParams()
        self._positions: List[SamplePosition] = []

    @property
    def calibration(self) -> CalibrationParams:
        return self._calibration

    def set_calibration(
        self,
        pixels_per_mm_x: float = None,
        pixels_per_mm_y: float = None,
        origin_x: float = None,
        origin_y: float = None,
        sample_z: float = None,
        invert_x: bool = None,
        invert_y: bool = None,
        image_width: int = None,
        image_height: int = None
    ):
        """Update calibration parameters."""
        if pixels_per_mm_x is not None:
            self._calibration.pixels_per_mm_x = pixels_per_mm_x
        if pixels_per_mm_y is not None:
            self._calibration.pixels_per_mm_y = pixels_per_mm_y
        if origin_x is not None:
            self._calibration.origin_stage_x = origin_x
        if origin_y is not None:
            self._calibration.origin_stage_y = origin_y
        if sample_z is not None:
            self._calibration.sample_z = sample_z
        if invert_x is not None:
            self._calibration.invert_x = invert_x
        if invert_y is not None:
            self._calibration.invert_y = invert_y
        if image_width is not None:
            self._calibration.image_width = image_width
        if image_height is not None:
            self._calibration.image_height = image_height

        logger.info(f"Calibration updated: {self._calibration}")

    def pixel_to_stage(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to stage coordinates.

        Args:
            pixel_x: X coordinate in image pixels
            pixel_y: Y coordinate in image pixels

        Returns:
            (stage_x, stage_y) in mm
        """
        cal = self._calibration

        # Calculate offset from image center (or origin)
        dx_pixels = pixel_x
        dy_pixels = pixel_y

        # Apply inversion if needed
        if cal.invert_x:
            dx_pixels = cal.image_width - dx_pixels
        if cal.invert_y:
            dy_pixels = cal.image_height - dy_pixels

        # Convert to mm
        dx_mm = dx_pixels / cal.pixels_per_mm_x
        dy_mm = dy_pixels / cal.pixels_per_mm_y

        # Add to origin
        stage_x = cal.origin_stage_x + dx_mm
        stage_y = cal.origin_stage_y + dy_mm

        return stage_x, stage_y

    def stage_to_pixel(self, stage_x: float, stage_y: float) -> Tuple[int, int]:
        """
        Convert stage coordinates to pixel coordinates.

        Args:
            stage_x: X coordinate in mm
            stage_y: Y coordinate in mm

        Returns:
            (pixel_x, pixel_y) in image coordinates
        """
        cal = self._calibration

        # Calculate offset from origin
        dx_mm = stage_x - cal.origin_stage_x
        dy_mm = stage_y - cal.origin_stage_y

        # Convert to pixels
        dx_pixels = dx_mm * cal.pixels_per_mm_x
        dy_pixels = dy_mm * cal.pixels_per_mm_y

        # Apply inversion
        if cal.invert_x:
            pixel_x = int(cal.image_width - dx_pixels)
        else:
            pixel_x = int(dx_pixels)

        if cal.invert_y:
            pixel_y = int(cal.image_height - dy_pixels)
        else:
            pixel_y = int(dy_pixels)

        return pixel_x, pixel_y

    def from_centroids(
        self,
        centroids: List[Tuple[int, int]],
        areas: Optional[List[int]] = None
    ) -> List[SamplePosition]:
        """
        Generate sample positions from centroid list.

        Args:
            centroids: List of (x, y) pixel coordinates
            areas: Optional list of object areas

        Returns:
            List of SamplePosition objects
        """
        self._positions = []

        for i, (px, py) in enumerate(centroids):
            sx, sy = self.pixel_to_stage(px, py)

            pos = SamplePosition(
                id=i + 1,
                pixel_x=px,
                pixel_y=py,
                stage_x=sx,
                stage_y=sy,
                stage_z=self._calibration.sample_z,
                area=areas[i] if areas and i < len(areas) else 0,
                label=f"Sample_{i+1}"
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
                    cx = int(np.mean(x_indices))
                    cy = int(np.mean(y_indices))
                    centroids.append((cx, cy))
                    areas.append(mask_dict.get('area', np.sum(mask)))

        return self.from_centroids(centroids, areas)

    def add_position(
        self,
        pixel_x: int,
        pixel_y: int,
        label: str = ""
    ) -> SamplePosition:
        """Manually add a sample position."""
        sx, sy = self.pixel_to_stage(pixel_x, pixel_y)

        pos = SamplePosition(
            id=len(self._positions) + 1,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            stage_x=sx,
            stage_y=sy,
            stage_z=self._calibration.sample_z,
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
                'area': p.area,
                'label': p.label,
            }
            for p in self._positions
        ]

    def sort_by_distance(self, start_x: float = 0, start_y: float = 0):
        """Sort positions by distance from a starting point (traveling salesman approximation)."""
        if not self._positions:
            return

        # Simple nearest neighbor sorting
        sorted_positions = []
        remaining = self._positions.copy()
        current_x, current_y = start_x, start_y

        while remaining:
            # Find nearest
            min_dist = float('inf')
            nearest_idx = 0
            for i, pos in enumerate(remaining):
                dist = np.sqrt((pos.stage_x - current_x)**2 + (pos.stage_y - current_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i

            # Move to nearest
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
