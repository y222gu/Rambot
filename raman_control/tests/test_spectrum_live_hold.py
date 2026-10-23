"""
Dry-run tests for pausing the free-running live spectrum (hold_live).

Runs the real SpectrumThread loop (no hardware: connect and the exposure are
stubbed) and checks that while a hold is active the loop takes NO live frames
of its own, that on-demand acquire_frames() requests still work, and that the
live view resumes once every hold is released.
"""

import threading
import time

import numpy as np
import pytest

from raman_control.gui.threads.spectrum_thread import SpectrumThread


@pytest.fixture
def spec(monkeypatch):
    s = SpectrumThread()
    counts = {'frames': 0}

    def fake_connect():
        s._simulated = True
        s._connected = True
        return True

    def fake_acquire():
        counts['frames'] += 1
        time.sleep(0.005)
        return np.ones(8, dtype=np.float32)

    monkeypatch.setattr(s, '_connect_spectrometer', fake_connect)
    monkeypatch.setattr(s, '_disconnect_spectrometer', lambda: None)
    monkeypatch.setattr(s, '_acquire_spectrum', fake_acquire)
    s._config.integration_time = 1  # ms; keeps the simulated loop fast
    s.counts = counts
    t = threading.Thread(target=s.run, daemon=True)
    t.start()
    deadline = time.time() + 2
    while not s.is_connected and time.time() < deadline:
        time.sleep(0.01)
    yield s
    s._running = False
    t.join(2)


def frames_during(spec, seconds):
    start = spec.counts['frames']
    time.sleep(seconds)
    return spec.counts['frames'] - start


def test_live_runs_without_hold(spec):
    spec.start_acquisition()
    assert frames_during(spec, 0.3) > 0


def test_hold_stops_live_frames(spec):
    spec.start_acquisition()
    spec.hold_live()
    time.sleep(0.1)  # let any in-flight live frame finish
    assert frames_during(spec, 0.3) == 0
    spec.release_live()
    assert frames_during(spec, 0.3) > 0


def test_requests_still_served_while_held(spec):
    spec.start_acquisition()
    spec.hold_live()
    time.sleep(0.1)
    before = spec.counts['frames']
    frames = spec.acquire_frames(3, timeout=2.0)
    assert frames is not None and frames.shape == (3, 8)
    assert spec.counts['frames'] - before == 3  # exactly the requested frames
    spec.release_live()


def test_nested_holds_need_every_release(spec):
    spec.start_acquisition()
    spec.hold_live()   # e.g. acquisition run
    spec.hold_live()   # e.g. autofocus inside it
    spec.release_live()
    time.sleep(0.1)
    assert frames_during(spec, 0.3) == 0  # still held once
    spec.release_live()
    assert frames_during(spec, 0.3) > 0


def test_extra_release_does_not_go_negative(spec):
    spec.release_live()
    spec.hold_live()
    spec.start_acquisition()
    time.sleep(0.1)
    assert frames_during(spec, 0.3) == 0
    spec.release_live()
