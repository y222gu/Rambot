"""
Dry-run tests: with photobleaching on, the spectrometer Prusa autofocus runs
only at each sample's first point and the other points reuse that Z.

Drives the real AcquisitionWorker._measure_point over one 3-point sample with
the hardware-facing helpers stubbed, and counts Prusa scans / bleaches / the Z
each point is moved to.
"""

from types import SimpleNamespace

import pytest

from raman_control.gui.threads.acquisition_thread import (
    AcquisitionWorker, PrusaAfPlan, prusa_af_plan)
from raman_control.gui.widgets.protocol_panel import ProtocolPanelWidget as ProtocolPanel

PP = {'autofocus_wasatch_prusa': True}
PS = {'autofocus_wasatch_prusa_per_sample': True}
BL = {'wait_before_measure': True}


@pytest.mark.parametrize("params, first, expected", [
    ({}, True, PrusaAfPlan(False, False, False, False)),
    (BL, True, PrusaAfPlan(False, False, False, False)),
    # Every point, no bleach: one scan at every point.
    (PP, True, PrusaAfPlan(False, True, False, False)),
    (PP, False, PrusaAfPlan(False, True, False, False)),
    # Every point + bleach: pre-bleach scan only at the first point (its Z is
    # reused for bleaching), post-bleach scan at EVERY point, not stored.
    ({**PP, **BL}, True, PrusaAfPlan(True, True, True, False)),
    ({**PP, **BL}, False, PrusaAfPlan(True, False, True, False)),
    # Per sample: first point only (post-bleach refreshes the stored Z).
    (PS, True, PrusaAfPlan(True, True, False, True)),
    (PS, False, PrusaAfPlan(True, False, False, True)),
    ({**PS, **BL}, True, PrusaAfPlan(True, True, True, True)),
    ({**PS, **BL}, False, PrusaAfPlan(True, False, False, True)),
])
def test_prusa_af_plan(params, first, expected):
    assert prusa_af_plan(params, first) == expected


@pytest.mark.parametrize("params, first_scans, later_scans", [
    (PP, 1, 1),
    ({**PP, **BL}, 2, 1),
    (PS, 1, 0),
    ({**PS, **BL}, 2, 0),
])
def test_time_estimate_scan_counts(params, first_scans, later_scans):
    assert ProtocolPanel._prusa_scans(params, True) == first_scans
    assert ProtocolPanel._prusa_scans(params, False) == later_scans


BASE_Z = 10.0
AF_Z = [11.5 + 0.1 * i for i in range(20)]  # successive scan results


def make_worker(params):
    log = {'prusa': 0, 'bleach': 0, 'move_z': []}
    af_results = iter(AF_Z)

    def prusa_af():
        log['prusa'] += 1
        return next(af_results)

    def bleach(_s):
        log['bleach'] += 1

    def move(x, y, z, timeout=60.0):
        log['move_z'].append(z)
        return True

    p = dict(params)
    sig = SimpleNamespace(emit=lambda *a, **k: None)
    w = SimpleNamespace(
        _ctx=SimpleNamespace(motion=SimpleNamespace(move_xyz_blocking=move)),
        _p=p,
        _p_int=lambda k, d=0: int(p.get(k, d)),
        _p_float=lambda k, d=0.0: float(p.get(k, d)),
        _abort=False,
        _sample_prusa_z=None,
        measuring_point=sig, progress=sig, status=sig, error=sig,
        _objective_xy=lambda pt: (0.0, 0.0),
        _ensure_laser_on=lambda: None,
        _wasatch_prusa_af=prusa_af,
        _wasatch_nano_af=lambda: None,
        _wait_with_laser=bleach,
        _wait_if_paused=lambda: None,
        _acquire_one=lambda: [1.0],
        _save_group=lambda *a: None,
        # One repeat condition; acquisition itself is not under test here.
        _cond_idx=0,
        _point_conditions=lambda rep: [0],
        _condition_note=lambda c: "",
        _acquire_condition=lambda pt, i, c, done, *a: done + 1,
    )
    return w, log


def measure_sample(w, n_points=3):
    w._sample_prusa_z = None  # new sample, as the run loops do
    done = 0
    for i in range(n_points):
        done = AcquisitionWorker._measure_point(
            w, object(), i, 0, BASE_Z, done, 100, f"pt {i}",
            first_in_sample=(i == 0))


def test_bleach_with_per_point_af():
    w, log = make_worker({**PP, **BL, 'wait_time_s': 5})
    measure_sample(w)
    # First point: scan before + after its bleach; other points: after only.
    assert log['prusa'] == 4
    assert log['bleach'] == 3
    # Points 2 and 3 move to (and bleach at) the first point's PRE-bleach Z.
    assert log['move_z'] == [BASE_Z, AF_Z[0], AF_Z[0]]


def test_bleach_with_per_point_af_next_sample_refocuses():
    w, log = make_worker({**PP, **BL, 'wait_time_s': 5})
    measure_sample(w)
    measure_sample(w)
    assert log['prusa'] == 8
    # Second sample starts from the nominal Z, then reuses its own first scan.
    assert log['move_z'][3:] == [BASE_Z, AF_Z[4], AF_Z[4]]


def test_bleach_with_per_sample_af_is_unchanged():
    w, log = make_worker({**PS, **BL, 'wait_time_s': 5})
    measure_sample(w)
    assert log['prusa'] == 2
    assert log['bleach'] == 3
    # The post-bleach re-focus at the first point sets the reused Z.
    assert log['move_z'] == [BASE_Z, AF_Z[1], AF_Z[1]]


def test_per_point_af_without_bleach_is_unchanged():
    w, log = make_worker(PP)
    measure_sample(w)
    assert log['prusa'] == 3
    assert log['bleach'] == 0
    assert log['move_z'] == [BASE_Z, BASE_Z, BASE_Z]
