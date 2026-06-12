"""
Tests for the standalone pattern-layout algorithms (pattern_layouts.py) and the
segment->generate->stage-coord pipeline the Robotic Deposition worker uses.

No Qt / no hardware: pure functions on a synthetic droplet mask.
"""

import numpy as np
import pytest

from raman_control.segmentation import pattern_layouts as pl
from raman_control.segmentation.pattern_generator import (
    PatternGenerator, CalibrationParams)


@pytest.fixture
def disk_mask():
    """A filled disk (radius 60) centred in a 200x200 image."""
    h = w = 200
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - 100) ** 2 + (xx - 100) ** 2 < 60 ** 2).astype(np.uint8)


# --- counts: exact vs upper-bound ------------------------------------------

def test_line_count_is_exact(disk_mask):
    pts, orders = pl.generate_line(disk_mask, {'points': 7, 'buffer_px': 0})
    assert len(pts) == 7
    assert len(orders) == 7


def test_random_count_is_exact(disk_mask):
    pts, orders = pl.generate_random(disk_mask, {'points': 15})
    assert len(pts) == 15
    assert orders == []  # random has no meaningful order


def test_grid_count_is_bounded_by_rows_times_cols(disk_mask):
    pts, orders = pl.generate_grid(disk_mask, {'rows': 4, 'cols': 5, 'buffer_px': 0})
    assert 0 < len(pts) <= 4 * 5          # only points inside the disk are kept
    assert len(orders) == len(pts)


def test_ring_count_is_bounded_by_rays_times_rings(disk_mask):
    pts, orders = pl.generate_ring(
        disk_mask, {'rays': 8, 'rings': 3, 'interval_px': 15, 'buffer_px': 0})
    assert 0 < len(pts) <= 8 * 3
    assert len(orders) == len(pts)


# --- points stay inside the image; orders parallel points ------------------

@pytest.mark.parametrize("pattern,params", [
    ('grid', {'rows': 4, 'cols': 4, 'buffer_px': 5}),
    ('ring', {'rays': 12, 'rings': 2, 'interval_px': 12, 'buffer_px': 5}),
    ('line', {'points': 6, 'buffer_px': 5}),
    ('random', {'points': 10}),
])
def test_points_within_image_bounds(disk_mask, pattern, params):
    h, w = disk_mask.shape
    pts, orders = pl.generate(pattern, disk_mask, params)
    assert pts, "expected at least one point"
    for (x, y) in pts:
        assert 0 <= x < w and 0 <= y < h
    if pattern != 'random':
        assert len(orders) == len(pts)


# --- remove_outside drops points outside the RAW mask, any buffer ----------

@pytest.mark.parametrize("pattern,params", [
    ('grid', {'rows': 9, 'cols': 9}),
    ('ring', {'rays': 12, 'rings': 5, 'interval_px': 12}),
])
@pytest.mark.parametrize("buffer_px", [10, 0, -25])
def test_remove_outside_drops_points_outside_raw_mask(
        disk_mask, pattern, params, buffer_px):
    """'Remove Points Outside the Mask' must judge against the RAW segmented
    mask, independent of the placement buffer. A NEGATIVE buffer (place points
    outside) previously left points well outside the droplet because the test
    used the dilated mask — this is the regression guard for that bug."""
    p = dict(params, buffer_px=buffer_px, remove_outside=True)
    pts, _ = pl.generate(pattern, disk_mask, p)
    assert pts, "expected at least one point inside the disk"
    for (x, y) in pts:
        assert disk_mask[int(round(y)), int(round(x))] == 1, (
            f"{pattern} buf={buffer_px}: point ({x:.0f},{y:.0f}) is outside the mask")


def test_remove_outside_false_keeps_outside_points(disk_mask):
    """With the toggle OFF, a negative buffer is free to place points outside."""
    p = {'rows': 9, 'cols': 9, 'buffer_px': -25, 'remove_outside': False}
    pts, _ = pl.generate_grid(disk_mask, p)
    outside = sum(1 for (x, y) in pts
                  if disk_mask[int(round(y)), int(round(x))] == 0)
    assert outside > 0


def test_remove_outside_positive_buffer_keeps_margin(disk_mask):
    """A positive buffer with the toggle ON still enforces an inward margin
    (intersection of buffered + raw mask), not just 'inside the raw edge'."""
    import cv2
    p = {'rows': 11, 'cols': 11, 'buffer_px': 12, 'remove_outside': True}
    pts, _ = pl.generate_grid(disk_mask, p)
    inset = cv2.distanceTransform(disk_mask, cv2.DIST_L2, 5)
    assert pts
    assert min(inset[int(round(y)), int(round(x))] for (x, y) in pts) >= 11


# --- empty / no-area mask is handled gracefully ----------------------------

@pytest.mark.parametrize("pattern,params", [
    ('grid', {'rows': 3, 'cols': 3, 'buffer_px': 0}),
    ('ring', {'rays': 8, 'rings': 3, 'interval_px': 10, 'buffer_px': 0}),
    ('line', {'points': 5, 'buffer_px': 0}),
    ('random', {'points': 5}),
])
def test_empty_mask_returns_nothing(pattern, params):
    empty = np.zeros((50, 50), dtype=np.uint8)
    pts, orders = pl.generate(pattern, empty, params)
    assert pts == []
    assert orders == []


def test_unknown_pattern_raises(disk_mask):
    with pytest.raises(ValueError):
        pl.generate('spiral', disk_mask, {})


# --- the worker's segment->generate->stage pipeline (no Qt) -----------------

def test_generate_pipeline_maps_points_to_stage(disk_mask):
    """Mirror RoboticDepositionWorker._segment_and_generate: pattern_layouts ->
    PatternGenerator.from_centroids, asserting points become stage coords near
    the capture position (disk centre maps ~capture XY)."""
    points, orders = pl.generate_ring(
        disk_mask, {'rays': 8, 'rings': 2, 'interval_px': 15, 'buffer_px': 5})
    assert points

    cal = CalibrationParams(
        pixel_size_x_um=2.0, pixel_size_y_um=2.0,
        image_width=disk_mask.shape[1], image_height=disk_mask.shape[0],
        capture_stage_x=100.0, capture_stage_y=50.0, capture_stage_z=30.0,
        invert_y=True,
    )
    positions = PatternGenerator(cal).from_centroids(
        points, coordinates_orders=orders if orders else None)

    assert len(positions) == len(points)
    # Image is 200px wide at 2 µm/px => ±0.2 mm span from the centre (100, 50).
    for p in positions:
        assert 99.7 <= p.stage_x <= 100.3
        assert 49.7 <= p.stage_y <= 50.3
        assert p.stage_z == pytest.approx(30.0)
