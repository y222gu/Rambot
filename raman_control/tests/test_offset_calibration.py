"""
Tests for the X-dependent wide->objective Y offset calibration curve.

The Y offset drifts linearly with the WIDEFIELD X position. These tests cover
the linear fit itself, the CSV loader, the enable/disable switch, and the
CoordinateManager conversions that consume it. No hardware required.
"""

import os

import pytest

from raman_control.config.hardware_config import StageConfig
from raman_control.config.offset_calibration import (
    CALIBRATION_FILENAME,
    OffsetCalibrationCurve,
    default_calibration_path,
    fit_linear,
    get_offset_calibration,
    load_calibration_points,
)
from raman_control.core.coordinate_manager import CameraView, CoordinateManager


# ---------------------------------------------------------------------------
# The shipped CSV
# ---------------------------------------------------------------------------

def test_calibration_file_lives_in_config_package():
    """The CSV must sit next to offset_calibration.py, in config/."""
    path = default_calibration_path()
    assert os.path.basename(path) == CALIBRATION_FILENAME
    assert os.path.basename(os.path.dirname(path)) == 'config'
    assert os.path.isfile(path), f"missing calibration file: {path}"


def test_shipped_csv_parses_and_fits():
    points = load_calibration_points(default_calibration_path())
    assert len(points) == 6, "expected 6 calibration points"
    fit = fit_linear(points)
    assert fit is not None
    # The offset decreases as X grows in the measured data.
    assert fit.slope < 0
    assert fit.r_squared > 0.99


def test_singleton_is_shared():
    assert get_offset_calibration() is get_offset_calibration()


# ---------------------------------------------------------------------------
# fit_linear
# ---------------------------------------------------------------------------

def test_fit_exact_line():
    fit = fit_linear([(0.0, 1.0), (10.0, 3.0)])
    assert fit.slope == pytest.approx(0.2)
    assert fit.intercept == pytest.approx(1.0)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.value_at(5.0) == pytest.approx(2.0)      # interpolated
    assert fit.value_at(20.0) == pytest.approx(5.0)     # extrapolated above
    assert fit.value_at(-10.0) == pytest.approx(-1.0)   # extrapolated below


def test_fit_least_squares_through_noisy_points():
    # y = 2x exactly, with symmetric noise that cancels out.
    fit = fit_linear([(0.0, 1.0), (0.0, -1.0), (1.0, 3.0), (1.0, 1.0)])
    assert fit.slope == pytest.approx(2.0)
    assert fit.intercept == pytest.approx(0.0)


def test_fit_rejects_degenerate_input():
    assert fit_linear([]) is None
    assert fit_linear([(1.0, 2.0)]) is None            # single point
    assert fit_linear([(5.0, 1.0), (5.0, 2.0)]) is None  # all X identical


def test_fit_flat_line_is_valid():
    fit = fit_linear([(0.0, 3.0), (10.0, 3.0)])
    assert fit.slope == pytest.approx(0.0)
    assert fit.value_at(1000.0) == pytest.approx(3.0)
    assert fit.r_squared == pytest.approx(1.0)


def test_is_extrapolated_flags_out_of_range_x():
    fit = fit_linear([(10.0, 0.0), (20.0, 1.0)])
    assert not fit.is_extrapolated(15.0)
    assert fit.is_extrapolated(9.9)
    assert fit.is_extrapolated(20.1)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _write_csv(tmp_path, text):
    p = tmp_path / CALIBRATION_FILENAME
    p.write_text(text, encoding='utf-8')
    return str(p)


def test_loader_skips_header_and_blank_rows(tmp_path):
    path = _write_csv(tmp_path, "x,delta y\n1,10\n\n2,20\n# note,here\n3,30\n")
    assert load_calibration_points(path) == [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)]


def test_curve_from_custom_file(tmp_path):
    path = _write_csv(tmp_path, "x,delta y\n100,1.0\n200,2.0\n")
    curve = OffsetCalibrationCurve(path)
    assert curve.is_valid
    assert curve.offset_y_at(150.0) == pytest.approx(1.5)
    assert curve.offset_y_at(300.0) == pytest.approx(3.0)  # extrapolates


def test_curve_missing_file_is_invalid(tmp_path):
    curve = OffsetCalibrationCurve(str(tmp_path / "does_not_exist.csv"))
    assert not curve.is_valid
    assert curve.offset_y_at(150.0) is None
    assert "not found" in curve.error


def test_curve_unusable_file_is_invalid(tmp_path):
    path = _write_csv(tmp_path, "x,delta y\n100,1.0\n")  # only one point
    curve = OffsetCalibrationCurve(path)
    assert not curve.is_valid
    assert curve.offset_y_at(150.0) is None


def test_curve_reload_picks_up_edits(tmp_path):
    path = _write_csv(tmp_path, "x,delta y\n100,1.0\n200,2.0\n")
    curve = OffsetCalibrationCurve(path)
    assert curve.offset_y_at(150.0) == pytest.approx(1.5)

    _write_csv(tmp_path, "x,delta y\n100,0.0\n200,0.0\n")
    assert curve.reload()
    assert curve.offset_y_at(150.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CoordinateManager integration
# ---------------------------------------------------------------------------

def _mgr(tmp_path, use_curve=True, view=CameraView.WIDE_FOV,
         csv_text="x,delta y\n100,1.0\n200,2.0\n"):
    """CoordinateManager wired to a known curve: dy = 0.01 * x."""
    path = _write_csv(tmp_path, csv_text)
    cfg = StageConfig(
        calib_wide_to_obj_x=-60.0,
        calib_wide_to_obj_y=0.5,      # the fixed fallback value
        calib_wide_to_obj_z=-13.0,
        use_offset_calibration_curve=use_curve,
    )
    mgr = CoordinateManager(cfg, initial_view=view)
    mgr._offset_calib = OffsetCalibrationCurve(path)
    return mgr


def test_offset_y_at_uses_curve_when_enabled(tmp_path):
    mgr = _mgr(tmp_path, use_curve=True)
    assert mgr.offset_curve_active
    assert mgr.offset_y_at(150.0) == pytest.approx(1.5)
    assert mgr.offset_y_at(400.0) == pytest.approx(4.0)  # extrapolated
    # The fixed value is still readable for display.
    assert mgr.offset_y == pytest.approx(0.5)


def test_offset_y_at_uses_fixed_value_when_disabled(tmp_path):
    mgr = _mgr(tmp_path, use_curve=False)
    assert not mgr.offset_curve_active
    for x in (0.0, 150.0, 400.0):
        assert mgr.offset_y_at(x) == pytest.approx(0.5)


def test_falls_back_to_fixed_value_when_curve_unusable(tmp_path):
    mgr = _mgr(tmp_path, use_curve=True, csv_text="x,delta y\n100,1.0\n")
    assert not mgr.offset_curve_active
    assert mgr.offset_y_at(150.0) == pytest.approx(0.5)


def test_toggling_the_setting_takes_effect_immediately(tmp_path):
    mgr = _mgr(tmp_path, use_curve=True)
    assert mgr.offset_y_at(150.0) == pytest.approx(1.5)
    mgr.stage_config.use_offset_calibration_curve = False
    assert mgr.offset_y_at(150.0) == pytest.approx(0.5)


def test_offset_y_for_objective_x_removes_x_offset(tmp_path):
    mgr = _mgr(tmp_path, use_curve=True)
    # Objective X 90 corresponds to widefield X 90 - (-60) = 150.
    assert mgr.offset_y_for_objective_x(90.0) == pytest.approx(1.5)


def test_get_objective_position_uses_curve_at_widefield_x(tmp_path):
    mgr = _mgr(tmp_path, use_curve=True)
    ox, oy, oz = mgr.get_objective_position(150.0, 20.0, 30.0)
    assert ox == pytest.approx(90.0)
    assert oy == pytest.approx(21.5)   # 20 + curve(150)
    assert oz == pytest.approx(17.0)


def test_objective_position_varies_with_x(tmp_path):
    """The whole point: two X positions get two different Y offsets."""
    mgr = _mgr(tmp_path, use_curve=True)
    _, y_a, _ = mgr.get_objective_position(120.0, 0.0, 0.0)
    _, y_b, _ = mgr.get_objective_position(180.0, 0.0, 0.0)
    assert y_a == pytest.approx(1.2)
    assert y_b == pytest.approx(1.8)


def test_get_wide_position_is_the_exact_inverse(tmp_path):
    mgr = _mgr(tmp_path, use_curve=True)
    wide = (150.0, 20.0, 30.0)
    assert mgr.get_wide_position(*mgr.get_objective_position(*wide)) == \
        pytest.approx(wide)


def test_logical_physical_round_trip_in_objective_view(tmp_path):
    mgr = _mgr(tmp_path, use_curve=True, view=CameraView.OBJECTIVE)
    logical = (150.0, 20.0, 30.0)
    phys = mgr.logical_to_physical(*logical)
    assert phys == pytest.approx((90.0, 21.5, 17.0))
    assert mgr.physical_to_logical(*phys) == pytest.approx(logical)


def test_logical_to_physical_unchanged_in_wide_view(tmp_path):
    mgr = _mgr(tmp_path, use_curve=True, view=CameraView.WIDE_FOV)
    assert mgr.logical_to_physical(150.0, 20.0, 30.0) == \
        pytest.approx((150.0, 20.0, 30.0))


def test_switch_checks_agree_with_the_moves_they_guard(tmp_path):
    """can_switch_* must validate the same targets get_*_position returns."""
    mgr = _mgr(tmp_path, use_curve=True)
    safe, _ = mgr.can_switch_to_objective(150.0, 20.0, 30.0)
    assert safe
    # Round trip from the objective side lands back on the widefield point.
    obj = mgr.get_objective_position(150.0, 20.0, 30.0)
    assert mgr.get_wide_position(*obj) == pytest.approx((150.0, 20.0, 30.0))
