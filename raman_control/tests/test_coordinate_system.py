"""
Dry-run tests for the sample-centric coordinate system.

All tests run without hardware — CoordinateManager is tested directly,
and GUI/motion integration uses mocks for MotionThread and StepperStage.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from raman_control.core.coordinate_manager import CoordinateManager, CameraView
from raman_control.config.hardware_config import StageConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_coord_mgr(offset_x=10.0, offset_y=5.0, offset_z=0.0,
                     view=CameraView.WIDE_FOV, use_curve=False):
    """Create a CoordinateManager with a known test offset.

    ``use_curve`` defaults to False so these tests pin the Y offset to
    ``offset_y`` instead of the X-dependent calibration curve (which has its
    own tests in test_offset_calibration.py).
    """
    config = StageConfig(
        calib_wide_to_obj_x=offset_x,
        calib_wide_to_obj_y=offset_y,
        calib_wide_to_obj_z=offset_z,
        use_offset_calibration_curve=use_curve,
    )
    return CoordinateManager(config, initial_view=view)


@pytest.fixture
def mgr():
    """CoordinateManager with offset (10, 5, 0) starting in wide FOV."""
    return _make_coord_mgr()


@pytest.fixture
def mgr_obj():
    """CoordinateManager with offset (10, 5, 0) starting in objective."""
    return _make_coord_mgr(view=CameraView.OBJECTIVE)


@pytest.fixture
def stage_config():
    """StageConfig with known home position."""
    return StageConfig(home_x=100.0, home_y=100.0, home_z=30.0)


@pytest.fixture
def mock_motion():
    """Mock MotionThread that records request_move_to / request_go_home calls."""
    m = MagicMock()
    m.is_connected = True
    m.all_referenced = True
    return m


# ===================================================================
# A. CoordinateManager Unit Tests
# ===================================================================

class TestLogicalToPhysical:
    def test_identity_in_wide_fov(self, mgr):
        assert mgr.logical_to_physical(100, 200, 30) == (100, 200, 30)

    def test_adds_offset_in_objective(self, mgr_obj):
        assert mgr_obj.logical_to_physical(100, 200, 30) == (110, 205, 30)

    def test_with_z_offset(self):
        m = _make_coord_mgr(offset_z=2.0, view=CameraView.OBJECTIVE)
        assert m.logical_to_physical(100, 200, 30) == (110, 205, 32)


class TestPhysicalToLogical:
    def test_identity_in_wide_fov(self, mgr):
        assert mgr.physical_to_logical(100, 200, 30) == (100, 200, 30)

    def test_subtracts_offset_in_objective(self, mgr_obj):
        assert mgr_obj.physical_to_logical(110, 205, 30) == (100, 200, 30)

    def test_with_z_offset(self):
        m = _make_coord_mgr(offset_z=2.0, view=CameraView.OBJECTIVE)
        assert m.physical_to_logical(110, 205, 32) == (100, 200, 30)


class TestRoundTrip:
    def test_roundtrip_wide(self, mgr):
        pos = (42.5, 99.1, 33.0)
        assert mgr.physical_to_logical(*mgr.logical_to_physical(*pos)) == pytest.approx(pos)

    def test_roundtrip_objective(self, mgr_obj):
        pos = (42.5, 99.1, 33.0)
        assert mgr_obj.physical_to_logical(*mgr_obj.logical_to_physical(*pos)) == pytest.approx(pos)

    def test_roundtrip_with_z_offset(self):
        m = _make_coord_mgr(offset_x=3.5, offset_y=-1.2, offset_z=7.0,
                            view=CameraView.OBJECTIVE)
        pos = (10.0, 20.0, 30.0)
        assert m.physical_to_logical(*m.logical_to_physical(*pos)) == pytest.approx(pos)


class TestSwitchView:
    def test_switch_updates_state(self, mgr):
        assert mgr.current_view == CameraView.WIDE_FOV
        mgr.switch_view(CameraView.OBJECTIVE)
        assert mgr.current_view == CameraView.OBJECTIVE

    def test_switch_same_view_returns_false(self, mgr):
        assert mgr.switch_view(CameraView.WIDE_FOV) is False

    def test_switch_different_view_returns_true(self, mgr):
        assert mgr.switch_view(CameraView.OBJECTIVE) is True


class TestSafetyChecks:
    def test_can_switch_to_objective_in_bounds(self):
        m = _make_coord_mgr(offset_x=10, offset_y=5, offset_z=0)
        safe, msg = m.can_switch_to_objective(100, 100, 50)
        assert safe

    def test_can_switch_to_objective_out_of_x(self):
        m = _make_coord_mgr(offset_x=10, offset_y=5, offset_z=0)
        safe, msg = m.can_switch_to_objective(245, 100, 50)
        assert not safe
        assert "X" in msg

    def test_can_switch_to_wide_in_bounds(self):
        m = _make_coord_mgr(offset_x=10, offset_y=5, offset_z=0)
        m.switch_view(CameraView.OBJECTIVE)
        safe, msg = m.can_switch_to_wide(110, 105, 50)
        assert safe

    def test_can_switch_to_wide_out_of_bounds(self):
        m = _make_coord_mgr(offset_x=10, offset_y=5, offset_z=0)
        m.switch_view(CameraView.OBJECTIVE)
        # current physical = (5, 3, 50) → wide target = (-5, -2, 50) → out of bounds
        safe, msg = m.can_switch_to_wide(5, 3, 50)
        assert not safe


# ===================================================================
# B. Camera Switch Direction Tests
# ===================================================================

class TestCameraSwitchDirection:
    """Verify the stage moves the right direction on camera switch.

    We test by checking what physical coords request_move_to receives.
    """

    def test_wide_to_objective_adds_offset(self, mgr, mock_motion):
        """Switching wide→obj should move stage to logical + offset."""
        logical_pos = (100.0, 100.0, 50.0)

        # Simulate what MainWindow._on_camera_switch_requested does:
        mgr.switch_view(CameraView.OBJECTIVE)
        phys = mgr.logical_to_physical(*logical_pos)

        mock_motion.request_move_to(x=phys[0], y=phys[1], z=phys[2])
        mock_motion.request_move_to.assert_called_once_with(
            x=110.0, y=105.0, z=50.0,
        )

    def test_objective_to_wide_removes_offset(self, mgr_obj, mock_motion):
        """Switching obj→wide should move stage to logical (no offset)."""
        logical_pos = (100.0, 100.0, 50.0)

        mgr_obj.switch_view(CameraView.WIDE_FOV)
        phys = mgr_obj.logical_to_physical(*logical_pos)

        mock_motion.request_move_to(x=phys[0], y=phys[1], z=phys[2])
        mock_motion.request_move_to.assert_called_once_with(
            x=100.0, y=100.0, z=50.0,
        )

    def test_same_view_no_movement(self, mgr, mock_motion):
        """Switching to the same view should not trigger any movement."""
        changed = mgr.switch_view(CameraView.WIDE_FOV)
        assert changed is False
        mock_motion.request_move_to.assert_not_called()

    def test_logical_position_unchanged_after_switch(self, mgr):
        """Logical position should be identical before and after switch."""
        logical_pos = (42.0, 84.0, 35.0)

        # In wide FOV
        assert mgr.physical_to_logical(*logical_pos) == logical_pos

        # Switch to objective — physical changes but logical stays the same
        mgr.switch_view(CameraView.OBJECTIVE)
        phys = mgr.logical_to_physical(*logical_pos)
        assert mgr.physical_to_logical(*phys) == pytest.approx(logical_pos)

    def test_both_directions_with_nonzero_offset(self):
        """Full round-trip: wide→obj→wide with non-zero offset."""
        m = _make_coord_mgr(offset_x=15, offset_y=-3, offset_z=2)
        logical = (80.0, 120.0, 40.0)

        # wide → obj
        m.switch_view(CameraView.OBJECTIVE)
        phys_obj = m.logical_to_physical(*logical)
        assert phys_obj == (95.0, 117.0, 42.0)

        # obj → wide
        m.switch_view(CameraView.WIDE_FOV)
        phys_wide = m.logical_to_physical(*logical)
        assert phys_wide == logical


# ===================================================================
# C. Go Home + Go to Target Tests
# ===================================================================

class TestGoHome:
    def test_go_home_from_wide_fov(self, mgr, mock_motion, stage_config):
        """In wide FOV, go home should use home coords directly."""
        phys = mgr.logical_to_physical(
            stage_config.home_x, stage_config.home_y, stage_config.home_z
        )
        assert phys == (100.0, 100.0, 30.0)

    def test_go_home_from_objective(self, mgr_obj, mock_motion, stage_config):
        """In objective view, go home should add offset to home coords."""
        phys = mgr_obj.logical_to_physical(
            stage_config.home_x, stage_config.home_y, stage_config.home_z
        )
        assert phys == (110.0, 105.0, 30.0)

    def test_go_home_from_objective_with_z_offset(self, stage_config):
        """Objective with Z offset should shift home Z too."""
        m = _make_coord_mgr(offset_x=10, offset_y=5, offset_z=3,
                            view=CameraView.OBJECTIVE)
        phys = m.logical_to_physical(
            stage_config.home_x, stage_config.home_y, stage_config.home_z
        )
        assert phys == (110.0, 105.0, 33.0)


class TestGoToTarget:
    def test_target_from_wide_fov(self, mgr):
        """Target in wide FOV passes through unchanged."""
        target = (150.0, 80.0, 35.0)
        assert mgr.logical_to_physical(*target) == target

    def test_target_from_objective(self, mgr_obj):
        """Target in objective adds offset."""
        target = (150.0, 80.0, 35.0)
        assert mgr_obj.logical_to_physical(*target) == (160.0, 85.0, 35.0)


class TestCalibrateHome:
    def test_calibrate_saves_logical_position(self, mgr_obj, stage_config):
        """Calibrate Home saves the logical position (not physical)."""
        # Simulate: physical stage at (110, 105, 30), logical = (100, 100, 30)
        physical = (110.0, 105.0, 30.0)
        logical = mgr_obj.physical_to_logical(*physical)

        # Simulate what StageControlWidget._on_calibrate_home_clicked does
        stage_config.home_x = logical[0]
        stage_config.home_y = logical[1]
        stage_config.home_z = logical[2]

        assert stage_config.home_x == 100.0
        assert stage_config.home_y == 100.0
        assert stage_config.home_z == 30.0


class TestStepperGoToHome:
    """Test that StepperStage.go_to_home accepts and uses override coords."""

    def test_go_to_home_uses_overrides(self):
        """When x/y/z are provided they should be used instead of config."""
        from raman_control.hardware.stepper_stage import StepperStage

        stage = StepperStage(StageConfig())
        stage._connected = True
        stage._referenced = {'X': True, 'Y': True, 'Z': True}

        # Mock the protocol so move_to doesn't actually send G-code
        stage._protocol = MagicMock()
        stage._protocol.is_running = True

        moves = []
        original_move_to = stage.move_to

        def tracking_move_to(**kwargs):
            moves.append(kwargs)
            # Immediately fire callback as success
            cb = kwargs.get('callback')
            if cb:
                cb(True)
            return True

        stage.move_to = tracking_move_to

        stage.go_to_home(x=110, y=105, z=30)

        # First move is safety height, then XY, then Z
        assert len(moves) == 3
        # XY move should use overridden values
        assert moves[1]['x'] == 110
        assert moves[1]['y'] == 105
        # Z lower should use overridden value
        assert moves[2]['z'] == 30

    def test_go_to_home_falls_back_to_config(self):
        """When no overrides, config values are used."""
        from raman_control.hardware.stepper_stage import StepperStage

        config = StageConfig(home_x=100, home_y=100, home_z=30)
        stage = StepperStage(config)
        stage._connected = True
        stage._referenced = {'X': True, 'Y': True, 'Z': True}
        stage._protocol = MagicMock()
        stage._protocol.is_running = True

        moves = []

        def tracking_move_to(**kwargs):
            moves.append(kwargs)
            cb = kwargs.get('callback')
            if cb:
                cb(True)
            return True

        stage.move_to = tracking_move_to
        stage.go_to_home()

        assert len(moves) == 3
        assert moves[1]['x'] == 100
        assert moves[1]['y'] == 100
        assert moves[2]['z'] == 30


# ===================================================================
# D. Position Display Tests
# ===================================================================

class TestPositionDisplay:
    def test_wide_fov_display_equals_physical(self, mgr):
        """In wide FOV physical and logical are the same."""
        physical = (120.0, 80.0, 35.0)
        assert mgr.physical_to_logical(*physical) == physical

    def test_objective_display_subtracts_offset(self, mgr_obj):
        """In objective view, display should subtract offset."""
        physical = (130.0, 85.0, 35.0)
        logical = mgr_obj.physical_to_logical(*physical)
        assert logical == (120.0, 80.0, 35.0)

    def test_display_unchanged_after_camera_switch(self):
        """Switching cameras should not change displayed logical position."""
        m = _make_coord_mgr(offset_x=10, offset_y=5, offset_z=0)
        logical = (100.0, 100.0, 50.0)

        # In wide: physical = logical
        phys_wide = m.logical_to_physical(*logical)
        display_wide = m.physical_to_logical(*phys_wide)

        # Switch to objective
        m.switch_view(CameraView.OBJECTIVE)
        phys_obj = m.logical_to_physical(*logical)
        display_obj = m.physical_to_logical(*phys_obj)

        assert display_wide == pytest.approx(display_obj)

    def test_zero_offset_no_change(self):
        """With zero offset, physical = logical in any view."""
        m = _make_coord_mgr(offset_x=0, offset_y=0, offset_z=0,
                            view=CameraView.OBJECTIVE)
        pos = (50.0, 60.0, 35.0)
        assert m.physical_to_logical(*pos) == pos
        assert m.logical_to_physical(*pos) == pos


# ===================================================================
# E. Edge Cases + Safety
# ===================================================================

class TestEdgeCases:
    def test_zero_offset_works(self):
        """Zero offset should be identity in both views."""
        m = _make_coord_mgr(offset_x=0, offset_y=0, offset_z=0)
        pos = (100.0, 100.0, 50.0)

        assert m.logical_to_physical(*pos) == pos
        m.switch_view(CameraView.OBJECTIVE)
        assert m.logical_to_physical(*pos) == pos
        assert m.physical_to_logical(*pos) == pos

    def test_negative_offsets_work(self):
        """Negative offsets should subtract when going to objective."""
        m = _make_coord_mgr(offset_x=-5, offset_y=-3, offset_z=-1,
                            view=CameraView.OBJECTIVE)
        assert m.logical_to_physical(100, 100, 50) == (95, 97, 49)
        assert m.physical_to_logical(95, 97, 49) == (100, 100, 50)

    def test_large_offset_blocked_by_safety(self):
        """Offset that pushes position out of bounds should block switch."""
        config = StageConfig(
            calib_wide_to_obj_x=200,
            x_max=250,
        )
        m = CoordinateManager(config)
        # Position 100 + offset 200 = 300 > x_max 250
        safe, msg = m.can_switch_to_objective(100, 100, 50)
        assert not safe
        assert "X" in msg

    def test_position_validation(self, mgr):
        """validate_position should reject out-of-bounds."""
        valid, _ = mgr.validate_position(100, 100, 50)
        assert valid

        invalid, msg = mgr.validate_position(-1, 100, 50)
        assert not invalid
        assert "X" in msg

    def test_objective_switch_blocked_when_target_z_below_safe(self):
        """Switch to objective blocked when target Z < safe height."""
        config = StageConfig(
            calib_wide_to_obj_z=-30,
            z_safe=50,
        )
        m = CoordinateManager(config)
        # Current Z=60, target Z=60-30=30 < safe(50) → blocked
        safe, msg = m.can_switch_to_objective(100, 100, 60)
        assert not safe
        assert "Z" in msg
