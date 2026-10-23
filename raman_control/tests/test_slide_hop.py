"""
Dry-run tests for the Robotic Deposition slide-change hop.

Checks, without hardware, that wells map to the same slide numbers the sample
map draws, and that moving the objective to a droplet on another slide lifts Z
to the objective "Highest" + clearance, travels XY at that height, then lowers
back to "Highest" (and does nothing when staying on the same slide).
"""

from types import SimpleNamespace

import pytest

from raman_control.config.hardware_config import SampleDropConfig, StageConfig
from raman_control.core.coordinate_manager import CameraView, CoordinateManager
from raman_control.core.slide_layout import (
    ROW_GROUPS, COL_GROUPS, plan_slide_hop, slide_at_position, slide_of_well)
from raman_control.gui.widgets.sample_grid import VALID_WELLS
from raman_control.gui.threads.robotic_thread import RoboticDepositionWorker


# --- slide_of_well ---------------------------------------------------------

@pytest.mark.parametrize("well, slide", [
    # Bottom row of slides (COL_GROUPS[0]) runs left -> right (A..P) ...
    ("A4", 1), ("E6", 1), ("G5", 2), ("J4", 2), ("L4", 3), ("P6", 3),
    # ... the next row runs right -> left (snake order).
    ("L10", 4), ("G10", 5), ("A12", 6),
    ("A16", 7), ("H17", 8), ("P18", 9),
    ("N22", 10), ("H23", 11), ("B24", 12),
])
def test_slide_of_well_known_wells(well, slide):
    assert slide_of_well(well) == slide


def test_slide_of_well_is_case_and_space_tolerant():
    assert slide_of_well(" h10 ") == slide_of_well("H10") == 5


@pytest.mark.parametrize("bad", ["", "F4", "A8", "Z1", "A", "4A", None])
def test_slide_of_well_off_slide_or_invalid(bad):
    assert slide_of_well(bad) is None


def test_every_valid_well_is_on_a_slide_with_16_spots():
    counts = {}
    for w in VALID_WELLS:
        s = slide_of_well(w)
        assert s is not None, w
        counts[s] = counts.get(s, 0) + 1
    assert sorted(counts) == list(range(1, len(ROW_GROUPS) * len(COL_GROUPS) + 1))
    assert all(n == 16 for n in counts.values()), counts


# --- _slide_hop_if_needed --------------------------------------------------

class FakeMotion:
    def __init__(self, fail_on=None):
        self.calls = []
        self._fail_on = fail_on

    def move_z_blocking(self, z, speed=None, timeout=30.0):
        self.calls.append(('z', z))
        return self._fail_on != len(self.calls)

    def move_xyz_blocking(self, x=None, y=None, z=None, speed=None,
                          timeout=60.0):
        self.calls.append(('xyz', x, y, z))
        return self._fail_on != len(self.calls)


def make_worker(obj_range=(30.0, 40.0), clearance=2.0, z_max=200.0,
                fail_on=None, obj_slide=None, cur_z=35.0):
    motion = FakeMotion(fail_on)
    ctx = SimpleNamespace(motion=motion, obj_prusa_range=obj_range,
                          slide_change_z_clearance=clearance,
                          stage_z_max=z_max,
                          position_provider=lambda: (0.0, 0.0, cur_z))
    return SimpleNamespace(
        _ctx=ctx, _abort=False, _obj_slide=obj_slide,
        status=SimpleNamespace(emit=lambda *_: None),
        _objective_xy=lambda p: (p.x, p.y),
    ), motion


def hop(worker, name, point):
    return RoboticDepositionWorker._slide_hop_if_needed(worker, name, point)


def test_first_droplet_after_switch_always_hops():
    w, m = make_worker(obj_slide=None)
    assert hop(w, "A4", SimpleNamespace(x=10.0, y=20.0))
    assert m.calls == [('z', 42.0), ('xyz', 10.0, 20.0, 42.0), ('z', 40.0)]
    assert w._obj_slide == 1


def test_same_slide_does_not_move():
    w, m = make_worker(obj_slide=1)
    assert hop(w, "B5", SimpleNamespace(x=1.0, y=2.0))
    assert m.calls == []


def test_different_slide_hops():
    w, m = make_worker(obj_slide=1)
    assert hop(w, "G5", SimpleNamespace(x=5.0, y=6.0))
    assert m.calls == [('z', 42.0), ('xyz', 5.0, 6.0, 42.0), ('z', 40.0)]
    assert w._obj_slide == 2


def test_lift_is_clamped_to_stage_z_max():
    w, m = make_worker(obj_range=(190.0, 199.0), z_max=200.0, cur_z=195.0)
    assert hop(w, "A4", SimpleNamespace(x=0.0, y=0.0))
    assert m.calls[0] == ('z', 200.0)
    assert m.calls[-1] == ('z', 199.0)


@pytest.mark.parametrize("fail_on", [1, 2, 3])
def test_failed_move_reports_failure_and_forgets_slide(fail_on):
    w, m = make_worker(obj_slide=1, fail_on=fail_on)
    assert not hop(w, "G5", SimpleNamespace(x=5.0, y=6.0))
    assert len(m.calls) == fail_on  # stops at the failing move
    assert w._obj_slide is None     # next droplet hops again


def test_lift_never_lowers_an_objective_already_above_the_lift_height():
    w, m = make_worker(cur_z=50.0)  # already above Highest (40) + 2
    assert hop(w, "A4", SimpleNamespace(x=1.0, y=2.0))
    assert m.calls == [('z', 50.0), ('xyz', 1.0, 2.0, 50.0), ('z', 40.0)]


def test_plan_slide_hop_order():
    assert plan_slide_hop(10.0, 20.0, 35.0, 40.0, 2.0, 200.0) == [
        (None, None, 42.0), (10.0, 20.0, 42.0), (None, None, 40.0)]


# --- slide_at_position (logical stage mm -> slide) -------------------------

STAGE = StageConfig()
DROP = SampleDropConfig()


def well_xy(name):
    row = ord(name[0]) - ord('A')
    col = int(name[1:]) - 1
    return (STAGE.home_x + DROP.offset_home_to_a1_x + row * DROP.well_pitch_mm,
            STAGE.home_y + DROP.offset_home_to_a1_y + col * DROP.well_pitch_mm)


@pytest.mark.parametrize("well", sorted(VALID_WELLS))
def test_slide_at_position_matches_well_slide(well):
    x, y = well_xy(well)
    # Anywhere within the well's own cell maps to the same slide.
    q = DROP.well_pitch_mm * 0.4
    for dx, dy in [(0, 0), (q, q), (-q, -q)]:
        assert slide_at_position(x + dx, y + dy, STAGE, DROP) ==             slide_of_well(well)


def test_slide_at_position_gap_is_none():
    x, y = well_xy("F4")  # row F is the gap between slide row-groups
    assert slide_at_position(x, y, STAGE, DROP) is None


# --- MainWindow._manual_move (objective-view manual moves) -----------------

from raman_control.gui.main_window import MainWindow  # noqa: E402


class FakeMotionThread:
    def __init__(self):
        self.calls = []

    def request_move_to(self, x=None, y=None, z=None, speed=None):
        self.calls.append((x, y, z))


def make_window(view, cur_well, cur_z=35.0, target_hint=None,
                obj_limits=(40.0, 30.0)):
    cm = CoordinateManager(STAGE)
    if view == CameraView.OBJECTIVE:
        cm.switch_view(CameraView.OBJECTIVE)
    lx, ly = well_xy(cur_well)
    px, py, _ = cm.logical_to_physical(lx, ly, cur_z)
    return SimpleNamespace(
        _coordinate_manager=cm,
        _config=SimpleNamespace(stage=STAGE, sample_drop=DROP),
        _current_position=(px, py, cur_z),
        _obj_target_slide=target_hint,
        _motion_thread=FakeMotionThread(),
        _stage_control=SimpleNamespace(
            get_objective_z_limits=lambda: obj_limits),
        _status_widget=SimpleNamespace(show_message=lambda *a, **k: None),
    )


def move_to_well(win, well, phys_z=None):
    lx, ly = well_xy(well)
    px, py, _ = win._coordinate_manager.logical_to_physical(
        lx, ly, win._current_position[2])
    MainWindow._manual_move(win, px, py, phys_z)
    return px, py


def test_manual_widefield_move_never_hops():
    win = make_window(CameraView.WIDE_FOV, "A4")
    px, py = move_to_well(win, "P24")
    assert win._motion_thread.calls == [(px, py, None)]


def test_manual_objective_same_slide_no_hop():
    win = make_window(CameraView.OBJECTIVE, "A4", target_hint=1)
    px, py = move_to_well(win, "B5")
    assert win._motion_thread.calls == [(px, py, None)]


def test_manual_objective_other_slide_hops():
    win = make_window(CameraView.OBJECTIVE, "A4", target_hint=1)
    px, py = move_to_well(win, "H10")
    assert win._motion_thread.calls == [
        (None, None, 42.0), (px, py, 42.0), (None, None, 40.0)]
    assert win._obj_target_slide == 5


def test_manual_objective_unknown_hint_hops_even_on_same_slide():
    # e.g. first move after switching to the objective view
    win = make_window(CameraView.OBJECTIVE, "A4", target_hint=None)
    move_to_well(win, "B5")
    assert win._motion_thread.calls[0] == (None, None, 42.0)
    assert win._obj_target_slide == 1


def test_manual_objective_go_with_z_ends_at_requested_z():
    win = make_window(CameraView.OBJECTIVE, "A4", target_hint=1)
    px, py = move_to_well(win, "G5", phys_z=33.0)
    assert win._motion_thread.calls == [
        (None, None, 42.0), (px, py, 42.0), (None, None, 40.0),
        (None, None, 33.0)]


def test_manual_objective_hop_blocked_without_valid_limits():
    win = make_window(CameraView.OBJECTIVE, "A4", target_hint=1,
                      obj_limits=(30.0, 30.0))
    move_to_well(win, "H10")
    assert win._motion_thread.calls == []
