"""
Dry-run tests for per-repeat spectrum acquisition settings ("repeat
conditions") and the 'complete all repeat conditions before moving to the next
point' option.

Uses the real AcquisitionWorker with fake stage / spectrometer / NanoDrive and
records which exposure / power each spectrum was taken at, how often autofocus
and photobleaching ran, and which repeat number each saved file got.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from raman_control.gui.threads.acquisition_thread import (
    AcquisitionContext, AcquisitionWorker, all_conditions_per_point,
    repeat_conditions)

BASE = {'acq_exposure': 1000, 'acq_power': 450, 'reps_per_point': 2}
EXTRA = [{'exposure_ms': 5000, 'power_mw': 400, 'reps_per_point': 1},
         {'exposure_ms': 10000, 'power_mw': 300, 'reps_per_point': 3},
         {'exposure_ms': 15000, 'power_mw': 200, 'reps_per_point': 1}]
FOUR = {**BASE, 'repeat_each_sample': 4, 'repeat_conditions_enabled': True,
        'repeat_conditions': EXTRA}


# --- repeat_conditions / all_conditions_per_point --------------------------

def test_conditions_default_to_main_settings():
    conds = repeat_conditions({**BASE, 'repeat_each_sample': 3})
    assert conds == [{'exposure_ms': 1000.0, 'power_mw': 450.0,
                      'reps_per_point': 2}] * 3


def test_conditions_ignored_when_disabled():
    conds = repeat_conditions({**FOUR, 'repeat_conditions_enabled': False})
    assert [c['exposure_ms'] for c in conds] == [1000.0] * 4


def test_per_repeat_conditions():
    conds = repeat_conditions(FOUR)
    assert [c['exposure_ms'] for c in conds] == [1000, 5000, 10000, 15000]
    assert [c['power_mw'] for c in conds] == [450, 400, 300, 200]
    assert [c['reps_per_point'] for c in conds] == [2, 1, 3, 1]


def test_missing_extra_conditions_fall_back_to_main():
    conds = repeat_conditions({**FOUR, 'repeat_each_sample': 5})
    assert conds[4]['exposure_ms'] == 1000.0


def test_all_conditions_needs_more_than_one_repeat():
    assert all_conditions_per_point(
        {'repeat_all_conditions_per_point': True, 'repeat_each_sample': 2})
    assert not all_conditions_per_point(
        {'repeat_all_conditions_per_point': True, 'repeat_each_sample': 1})
    assert not all_conditions_per_point({'repeat_each_sample': 4})


# --- worker ----------------------------------------------------------------

class FakeSpec:
    is_connected = True

    def __init__(self):
        self.expo = None
        self.power = None
        self.shots = []  # (exposure, power) of every acquired spectrum

    def set_integration_time(self, ms):
        self.expo = ms

    def set_laser_power_mw(self, mw):
        self.power = mw

    def set_laser(self, on):
        pass

    def start_acquisition(self):
        pass

    def acquire_average(self, n):
        self.shots.append((self.expo, self.power))
        return np.ones(4, dtype=np.float32)


class FakeNano:
    is_connected = True
    initial_position = 50.0

    def __init__(self):
        self.z = []

    def move_to(self, z):
        self.z.append(z)
        return True


def make_worker(params, nano=None):
    spec = FakeSpec()
    motion = SimpleNamespace(move_xyz_blocking=lambda *a, **k: True)
    ctx = AcquisitionContext(
        motion=motion, spectrometer=spec, nanodrive=nano,
        frame_provider=lambda: None,
        position_provider=lambda: (0.0, 0.0, 10.0),
        wavelengths_nm_provider=lambda: None,
        obj_prusa_range=(10.0, 12.0), nano_range=(40.0, 60.0))
    w = AcquisitionWorker(ctx, [], {'spec_af_fine': 3, **params})
    log = {'prusa': 0, 'bleach': [], 'saved': []}
    w._laser_warmed = True  # skip the one-time warm-up sleep

    def prusa_af():
        log['prusa'] += 1
        return 11.0

    def bleach(_s):
        # The real photobleach sets the laser to the current condition's power.
        log['bleach'].append(w._acq_power())

    def save(point, idx, rep, spectra, z_index=None, z_um=None):
        log['saved'].append((idx, rep, len(spectra), z_index))

    w._wasatch_prusa_af = prusa_af
    w._wait_with_laser = bleach
    w._save_group = save
    return w, spec, log


def point(i):
    return SimpleNamespace(stage_x=float(i), stage_y=0.0)


def run_points(w, n_points, rep=0):
    done = 0
    for i in range(n_points):
        done = w._measure_point(point(i), i, rep, 10.0, done, 100, f"pt {i}",
                                first_in_sample=(i == 0))
    return done


ALL = {**FOUR, 'repeat_all_conditions_per_point': True,
       'autofocus_wasatch_prusa': True, 'wait_before_measure': True,
       'wait_time_s': 1}


def test_all_conditions_at_each_point_focus_and_bleach_once():
    w, spec, log = make_worker(ALL)
    assert w._outer_repeats() == 1
    run_points(w, 2)
    # Per point: Prusa before + after bleach at the first point, after-bleach
    # only at the second (photobleach rule); ONE bleach per point.
    assert log['prusa'] == 3
    assert len(log['bleach']) == 2
    # Each point: every condition in turn, each at its own exposure / power /
    # reps, saved as its own repeat number.
    one_point = ([(1000, 450)] * 2 + [(5000, 400)] + [(10000, 300)] * 3
                 + [(15000, 200)])
    assert spec.shots == one_point * 2
    assert log['saved'] == [(0, 0, 2, None), (0, 1, 1, None), (0, 2, 3, None),
                            (0, 3, 1, None),
                            (1, 0, 2, None), (1, 1, 1, None), (1, 2, 3, None),
                            (1, 3, 1, None)]
    # Photobleach uses the first condition's power.
    assert log['bleach'] == [450, 450]


def test_progress_total_counts_every_condition():
    w, _, _ = make_worker(ALL)
    assert w._run_spectra_per_point() == 2 + 1 + 3 + 1


def test_per_repeat_conditions_without_all_per_point():
    w, spec, log = make_worker({**FOUR, 'autofocus_wasatch_prusa': True})
    assert w._outer_repeats() == 4
    run_points(w, 1, rep=2)  # third repeat -> third condition only
    assert spec.shots == [(10000, 300)] * 3
    assert log['saved'] == [(0, 2, 3, None)]
    assert log['prusa'] == 1


def test_all_conditions_with_z_stack_at_each_z_step():
    nano = FakeNano()
    w, spec, log = make_worker({**ALL, 'wait_before_measure': False,
                                'autofocus_wasatch_prusa': False,
                                'z_stack': True}, nano=nano)
    run_points(w, 1)
    assert nano.z == [40.0, 50.0, 60.0]  # 3 Z steps, one move each
    # Every Z step acquires every condition, one file per (z-step, condition).
    assert [(r, z) for (_, r, _, z) in log['saved']] == [
        (c, zi) for zi in range(3) for c in range(4)]
    assert len(spec.shots) == 3 * (2 + 1 + 3 + 1)


def test_single_repeat_unchanged():
    w, spec, log = make_worker({**BASE, 'repeat_each_sample': 1,
                                'repeat_all_conditions_per_point': True})
    assert w._outer_repeats() == 1
    run_points(w, 1)
    assert spec.shots == [(1000, 450)] * 2
    assert log['saved'] == [(0, 0, 2, None)]
