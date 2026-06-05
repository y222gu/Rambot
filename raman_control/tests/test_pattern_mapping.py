"""
Dry-run tests for the pattern point -> stage coordinate mapping.

These verify, without hardware, that a generated pattern point maps to the
correct physical stage coordinate in BOTH camera views, using the real
production classes:

  * PatternGenerator.pixel_to_stage  — captured-image pixel -> widefield
    (logical) stage mm, center-relative with Y inversion.
  * CoordinateManager.get_objective_position / logical_to_physical — the
    wide->objective offset applied when in objective view.

The end-to-end test mirrors the live path exactly:
  pattern point (pixel) --PatternGenerator--> widefield stage coord
                         --CoordinateManager--> objective stage coord
and asserts the click-to-move conversion (logical_to_physical in objective
view) lands on the same objective coordinate.

NOTE: this checks the math/sign/scale logic, not whether the calibration
*values* (pixel_size, wide->obj offset) match the physical rig — that can
only be confirmed on hardware.
"""

import pytest

from raman_control.segmentation.pattern_generator import (
    PatternGenerator, CalibrationParams,
)
from raman_control.core.coordinate_manager import CoordinateManager, CameraView
from raman_control.config.hardware_config import StageConfig


# ---------------------------------------------------------------------------
# Fixtures — clean round numbers so the arithmetic is obvious.
# ---------------------------------------------------------------------------

# 2 um/pixel, 1000x800 image (center at 500,400), captured at stage (100,50,30).
CAL = CalibrationParams(
    pixel_size_x_um=2.0,
    pixel_size_y_um=2.0,
    image_width=1000,
    image_height=800,
    capture_stage_x=100.0,
    capture_stage_y=50.0,
    capture_stage_z=30.0,
    invert_y=True,
)

# Wide -> objective offset (realistic signs from hardware_config defaults).
OBJ_OFFSET = dict(calib_wide_to_obj_x=-62.48,
                  calib_wide_to_obj_y=1.32,
                  calib_wide_to_obj_z=-13.26)


@pytest.fixture
def gen():
    return PatternGenerator(CAL)


@pytest.fixture
def coord_wide():
    return CoordinateManager(StageConfig(**OBJ_OFFSET),
                             initial_view=CameraView.WIDE_FOV)


@pytest.fixture
def coord_obj():
    return CoordinateManager(StageConfig(**OBJ_OFFSET),
                             initial_view=CameraView.OBJECTIVE)


# ===================================================================
# A. pixel -> widefield stage
# ===================================================================

def test_center_pixel_maps_to_capture_position(gen):
    """The image center is exactly the stage position at capture time."""
    sx, sy, sz = gen.pixel_to_stage(500, 400)
    assert sx == pytest.approx(100.0)
    assert sy == pytest.approx(50.0)
    assert sz == pytest.approx(30.0)


def test_pixel_right_increases_stage_x(gen):
    """+100 px in X -> +100*2um = +0.2 mm in stage X; Y unchanged."""
    sx, sy, _ = gen.pixel_to_stage(600, 400)
    assert sx == pytest.approx(100.2)
    assert sy == pytest.approx(50.0)


def test_pixel_down_decreases_stage_y_when_inverted(gen):
    """+100 px DOWN in image -> stage Y DECREASES (image Y down, printer Y up)."""
    sx, sy, _ = gen.pixel_to_stage(500, 500)
    assert sx == pytest.approx(100.0)
    assert sy == pytest.approx(49.8)  # 50 - 0.2


def test_scale_at_image_edge(gen):
    """Edge pixel (width away from center) scales correctly: 500 px * 2um = 1.0 mm."""
    sx, _, _ = gen.pixel_to_stage(1000, 400)
    assert sx == pytest.approx(101.0)


def test_no_y_inversion_flips_sign():
    """With invert_y=False a downward pixel increases stage Y."""
    cal = CalibrationParams(
        pixel_size_x_um=2.0, pixel_size_y_um=2.0,
        image_width=1000, image_height=800,
        capture_stage_x=100.0, capture_stage_y=50.0, capture_stage_z=30.0,
        invert_y=False,
    )
    _, sy, _ = PatternGenerator(cal).pixel_to_stage(500, 500)
    assert sy == pytest.approx(50.2)


def test_round_trip_pixel_stage_pixel(gen):
    """pixel -> stage -> pixel returns the original pixel."""
    for px, py in [(123.0, 456.0), (0.0, 0.0), (999.0, 799.0), (500.0, 400.0)]:
        sx, sy, _ = gen.pixel_to_stage(px, py)
        rpx, rpy = gen.stage_to_pixel(sx, sy)
        assert rpx == pytest.approx(px, abs=1e-6)
        assert rpy == pytest.approx(py, abs=1e-6)


# ===================================================================
# B. widefield -> objective offset
# ===================================================================

def test_objective_position_adds_offset(coord_wide):
    wide = (100.0, 50.0, 30.0)
    obj = coord_wide.get_objective_position(*wide)
    assert obj == pytest.approx((100.0 - 62.48, 50.0 + 1.32, 30.0 - 13.26))


def test_logical_to_physical_identity_in_wide_view(coord_wide):
    """In wide FOV the logical coord is used as-is (no offset)."""
    assert coord_wide.logical_to_physical(100.0, 50.0, 30.0) == \
        pytest.approx((100.0, 50.0, 30.0))


def test_logical_to_physical_adds_offset_in_objective_view(coord_obj):
    """In objective view the wide->obj offset is applied."""
    assert coord_obj.logical_to_physical(100.0, 50.0, 30.0) == \
        pytest.approx((100.0 - 62.48, 50.0 + 1.32, 30.0 - 13.26))


# ===================================================================
# C. End-to-end: the exact live path, both views
# ===================================================================

def test_end_to_end_point_widefield_and_objective(gen, coord_wide, coord_obj):
    """
    A pattern point must resolve to the correct stage coordinate in both
    views, and the click-to-move conversion must agree with the objective
    coordinate computed when the positions were generated.
    """
    # 1. Pattern generation: two pixel points -> widefield stage coords.
    points = [(600.0, 400.0), (500.0, 500.0)]
    positions = gen.from_centroids(points)

    # Expected widefield coords (from section A).
    assert (positions[0].stage_x, positions[0].stage_y) == \
        pytest.approx((100.2, 50.0))
    assert (positions[1].stage_x, positions[1].stage_y) == \
        pytest.approx((100.0, 49.8))

    # 2. main_window._on_positions_generated fills objective coords.
    for p in positions:
        p.obj_stage_x, p.obj_stage_y, p.obj_stage_z = \
            coord_wide.get_objective_position(p.stage_x, p.stage_y, p.stage_z)

    # 3. Click-to-move in WIDE view -> physical == widefield coord.
    for p in positions:
        phys = coord_wide.logical_to_physical(p.stage_x, p.stage_y, p.stage_z)
        assert phys == pytest.approx((p.stage_x, p.stage_y, p.stage_z))

    # 4. Click-to-move in OBJECTIVE view -> physical == objective coord.
    #    This is the key cross-check the user asked for: the SAME point,
    #    viewed in objective mode, lands on its objective coordinate.
    for p in positions:
        phys = coord_obj.logical_to_physical(p.stage_x, p.stage_y, p.stage_z)
        assert phys == pytest.approx(
            (p.obj_stage_x, p.obj_stage_y, p.obj_stage_z)
        )


def test_objective_round_trip(coord_obj):
    """get_objective_position then get_wide_position returns the original."""
    wide = (137.9, 47.5, 21.3)
    obj = coord_obj.get_objective_position(*wide)
    back = coord_obj.get_wide_position(*obj)
    assert back == pytest.approx(wide)
