"""
Coordinate utilities for image-to-physical coordinate conversion.

Provides functions for:
- Image pixel to stage coordinates conversion
- Sample drop grid coordinate calculations
- Calibration offset handling
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np


@dataclass
class Calibration:
    """Calibration data for coordinate transformations."""
    # Pixel to mm scaling factors
    scale_x: float = 1.0  # mm/pixel
    scale_y: float = 1.0  # mm/pixel

    # Offset from image origin to stage origin
    offset_x: float = 0.0  # mm
    offset_y: float = 0.0  # mm

    # Rotation (if camera is rotated relative to stage)
    rotation: float = 0.0  # degrees

    # Flip flags (camera may be mirrored)
    flip_x: bool = False
    flip_y: bool = True  # Y is typically flipped


def pixel_to_stage(
    pixel_x: float,
    pixel_y: float,
    calibration: Calibration,
    image_width: int = 640,
    image_height: int = 480
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to stage coordinates.

    Args:
        pixel_x: X coordinate in pixels (from image left)
        pixel_y: Y coordinate in pixels (from image top)
        calibration: Calibration data
        image_width: Width of image in pixels
        image_height: Height of image in pixels

    Returns:
        Tuple of (stage_x, stage_y) in mm
    """
    # Apply flips
    x = pixel_x if not calibration.flip_x else (image_width - pixel_x)
    y = pixel_y if not calibration.flip_y else (image_height - pixel_y)

    # Apply rotation if present
    if calibration.rotation != 0:
        angle_rad = np.radians(calibration.rotation)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Rotate around image center
        cx, cy = image_width / 2, image_height / 2
        x_centered = x - cx
        y_centered = y - cy

        x_rotated = x_centered * cos_a - y_centered * sin_a + cx
        y_rotated = x_centered * sin_a + y_centered * cos_a + cy

        x, y = x_rotated, y_rotated

    # Convert to mm and apply offsets
    stage_x = x * calibration.scale_x + calibration.offset_x
    stage_y = y * calibration.scale_y + calibration.offset_y

    return (round(stage_x, 3), round(stage_y, 3))


def stage_to_pixel(
    stage_x: float,
    stage_y: float,
    calibration: Calibration,
    image_width: int = 640,
    image_height: int = 480
) -> Tuple[int, int]:
    """
    Convert stage coordinates to pixel coordinates.

    Args:
        stage_x: X coordinate in mm
        stage_y: Y coordinate in mm
        calibration: Calibration data
        image_width: Width of image in pixels
        image_height: Height of image in pixels

    Returns:
        Tuple of (pixel_x, pixel_y) as integers
    """
    # Remove offsets and convert to pixels
    x = (stage_x - calibration.offset_x) / calibration.scale_x
    y = (stage_y - calibration.offset_y) / calibration.scale_y

    # Apply inverse rotation if present
    if calibration.rotation != 0:
        angle_rad = np.radians(-calibration.rotation)  # Negative for inverse
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        cx, cy = image_width / 2, image_height / 2
        x_centered = x - cx
        y_centered = y - cy

        x_rotated = x_centered * cos_a - y_centered * sin_a + cx
        y_rotated = x_centered * sin_a + y_centered * cos_a + cy

        x, y = x_rotated, y_rotated

    # Apply inverse flips
    pixel_x = x if not calibration.flip_x else (image_width - x)
    pixel_y = y if not calibration.flip_y else (image_height - y)

    return (int(round(pixel_x)), int(round(pixel_y)))


def calculate_sample_drop_coordinates(
    home_position: Tuple[float, float],
    grid_pattern: Tuple[int, ...],
    offset_to_p1: Tuple[float, float],
    interval_x: float,
    interval_y: float,
    view_offset: Tuple[float, float] = (0.0, 0.0)
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """
    Calculate physical coordinates for all sample drops in the grid.

    Args:
        home_position: (x, y) home position in mm
        grid_pattern: Number of drops per column
        offset_to_p1: Offset from home to first drop (x, y) in mm
        interval_x: X spacing between columns in mm
        interval_y: Y spacing between rows in mm
        view_offset: Additional offset for objective vs widefield

    Returns:
        Dictionary mapping (row, col) to (x, y) coordinates in mm
    """
    coordinates = {}
    home_x, home_y = home_position
    offset_x, offset_y = offset_to_p1
    view_x, view_y = view_offset

    for col_index, num_drops in enumerate(grid_pattern):
        for row_index in range(num_drops):
            x = round(
                home_x + offset_x + col_index * interval_x + view_x,
                2
            )
            y = round(
                home_y + offset_y - row_index * interval_y + view_y,
                2
            )
            coordinates[(row_index, col_index)] = (x, y)

    return coordinates


def calculate_sampling_points(
    center_position: Tuple[float, float],
    relative_positions: List[Tuple[float, float]],
    flip_y: bool = True
) -> List[Tuple[float, float]]:
    """
    Calculate absolute positions for sampling points relative to a center.

    Args:
        center_position: (x, y) center position in mm
        relative_positions: List of (dx, dy) offsets in um
        flip_y: Whether to flip Y axis (printer Y is inverted)

    Returns:
        List of (x, y) absolute positions in mm
    """
    center_x, center_y = center_position
    positions = []

    for dx_um, dy_um in relative_positions:
        # Convert um to mm
        dx_mm = dx_um / 1000.0
        dy_mm = dy_um / 1000.0

        # Apply Y flip if needed
        if flip_y:
            dy_mm = -dy_mm

        x = round(center_x + dx_mm, 3)
        y = round(center_y + dy_mm, 3)
        positions.append((x, y))

    return positions
