"""
Tests for the Calibration tab (unit 1 motion test).

Covers:
- the encoded tests match the checklist workbook row by row (G-code + names)
- G-code safety rules (Z below focus, travel limits, microsteps)
- atomic relative blocks in SerialProtocol (no interleaving, no retry of
  moves, G90 always restored) and the out-of-band M112
- StepperStage.run_relative_block bookkeeping (Live-tab jog base stays right)
- session folder / TIFF / checklist workbook output
- the worker end to end with a fake stage and camera
No real hardware is touched.
"""

import os
import threading
import time

import numpy as np
import pytest

from raman_control.core import motion_test_protocol as mtp
from raman_control.core.calibration_session import (
    CalibrationSession, default_template_path, write_tiff, STATUS_DONE,
    STATUS_PROBLEM)
from raman_control.core.serial_protocol import SerialProtocol
from raman_control.config.hardware_config import StageConfig

openpyxl = pytest.importorskip('openpyxl')

PRUSA_M92 = {'X': 100.0, 'Y': 100.0, 'Z': 400.0, 'E': 280.0}
MID = (120.0, 100.0, 35.0)


# ---------------------------------------------------------------------------
# Protocol definition vs. checklist
# ---------------------------------------------------------------------------

def _template_rows(sheet):
    wb = openpyxl.load_workbook(default_template_path())
    ws = wb[sheet]
    return {r[0].value: (r[2].value, r[3].value)
            for r in ws.iter_rows(min_row=5) if r[0].value is not None}


@pytest.mark.parametrize('key', list(mtp.all_tests()))
def test_steps_match_checklist(key):
    """Every step's G-code and file name equal the checklist's columns C/D.

    Two documented additions: Test 4 ends each run with M400 (so the
    software knows when motion stopped and records 5 s more), and the 6B
    reference row waits M400 + G4 S2 (no motion) before its image.
    """
    test = mtp.all_tests()[key]
    rows = _template_rows(test.sheet)
    for s in test.steps:
        assert s.row in rows, f"{key} row {s.row} not on the sheet"
        gcode, image = rows[s.row]
        mine = list(s.gcode)
        if key == 'T4' and s.kind == 'record':
            assert mine[-1] == 'M400'
            mine = mine[:-1]
        if key == 'T6B' and s.row == 35:
            assert mine == ['M400', 'G4 S2']
            mine = []
        assert (', '.join(mine) if mine else None) == gcode, (key, s.row)
        assert s.image == image, (key, s.row)


def test_every_checklist_row_is_covered():
    tests = mtp.all_tests()
    covered = {}
    for t in tests.values():
        covered.setdefault(t.sheet, set()).update(s.row for s in t.steps)
    for sheet, rows in covered.items():
        assert rows == set(_template_rows(sheet)), sheet


def test_image_counts_match_handoff_list():
    counts = {}
    for t in mtp.all_tests().values():
        for s in t.steps:
            if s.image and s.kind != 'record':
                counts[t.folder] = counts.get(t.folder, 0) + 1
    for folder, n in mtp.EXPECTED_IMAGE_COUNTS.items():
        assert counts[folder] == n, folder


def test_timed_captures():
    t0, t5 = mtp.build_t0(), mtp.build_t5()
    assert [s.at_s for s in t0.steps] == [i * 5.0 for i in range(10)]
    timed = [s.at_s for s in t5.steps if s.image]
    assert timed == [m * 60.0 for m in range(31)]


# ---------------------------------------------------------------------------
# Safety rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('key', list(mtp.all_tests()))
def test_all_tests_pass_safety_mid_travel(key):
    issues = mtp.check_test(mtp.all_tests()[key], start_xyz=MID,
                            limits=mtp.stage_limits(StageConfig()),
                            steps_per_mm=PRUSA_M92)
    assert not [i for i in issues if i.level in ('error', 'warning')], issues


def test_z_never_below_focus_more_than_50um():
    # T6A goes exactly 20 um below focus; T3/T6B lift 1 mm and come back.
    blocks = [s.gcode for s in mtp.build_t6a().steps if s.gcode]
    z = lo = 0.0
    for b in blocks:
        for line in b:
            z += mtp.parse_g1(line)[0].get('Z', 0.0)
            lo = min(lo, z)
    assert lo == pytest.approx(-0.020)
    assert z == pytest.approx(0.020)  # ends 20 um above focus, sweep start


def test_unpaired_negative_z_is_an_error():
    issues = mtp.check_blocks([('G1 Z-1 F300', 'M400')])
    assert any(i.level == 'error' for i in issues)
    issues = mtp.check_blocks([('G1 Z1 F300', 'G1 Z-1 F300')])
    assert not any(i.level == 'error' for i in issues)


def test_travel_limit_violation_blocks_t3():
    near_x_min = (3.0, 100.0, 35.0)
    issues = mtp.check_test(mtp.build_t3('X'), start_xyz=near_x_min,
                            limits=mtp.stage_limits(StageConfig()))
    assert any(i.level == 'error' and 'X' in i.message for i in issues)


def test_disallowed_line_in_block():
    issues = mtp.check_blocks([('G90',), ('G28 X',)])
    assert sum(i.level == 'error' for i in issues) == 2


def test_microstep_warning():
    issues = mtp.check_test(mtp.build_t2('X'), steps_per_mm={'X': 80.0})
    assert any(i.level == 'warning' and 'microstep' in i.message for i in issues)


def test_parse_m92_prusa_reply():
    text = ("echo:Steps per unit:\n"
            "echo:  M92 X100.00 Y100.00 Z400.00 E280.00\n"
            "echo:Maximum feedrates (mm/s):\nok")
    assert mtp.parse_m92(text) == PRUSA_M92
    assert mtp.m92_line(text) == 'M92 X100.00 Y100.00 Z400.00 E280.00'


def test_move_formatting():
    assert mtp._move('X', 0.010, 60) == 'G1 X0.010 F60'
    assert mtp._move('Z', -0.0025, 60) == 'G1 Z-0.0025 F60'
    assert mtp._move('Y', -5, 1500) == 'G1 Y-5 F1500'
    assert mtp.jog_block('Z', 2.5) == ('G1 Z0.0025 F60', 'M400')


# ---------------------------------------------------------------------------
# SerialProtocol: atomic sequences
# ---------------------------------------------------------------------------

class AckSerial:
    """Mock port that ACKs every line except those in ``silent``."""

    def __init__(self, protocol_ref, silent=()):
        self.written = []
        self.is_open = True
        self._p = protocol_ref
        self.silent = set(silent)
        self.lock = threading.Lock()

    def write(self, data):
        line = data.decode().strip()
        with self.lock:
            self.written.append(line)
        if line not in self.silent:
            self._p[0]._response_queue.put('ok')
        return len(data)

    def flush(self):
        pass


@pytest.fixture
def running_protocol():
    ref = [None]
    p = SerialProtocol(port='COMX', command_delay=0.0, max_retries=3)
    ser = AckSerial(ref)
    ref[0] = p
    p._serial = ser
    p._connected = True
    p._running = True
    t = threading.Thread(target=p._send_worker, daemon=True)
    t.start()
    yield p, ser
    p._running = False
    t.join(timeout=2)


def _wait(event, timeout=5.0):
    assert event.wait(timeout), "timed out"


def test_sequence_is_not_interleaved(running_protocol):
    p, ser = running_protocol
    done = threading.Event()
    p.send_sequence(['G91', 'G1 X0.010 F60', 'M400'], finally_commands=['G90'],
                    callback=lambda ok, res: done.set())
    # Commands queued by "another thread" right after must run after G90.
    p.send_command('G0 X10 F3000')
    p.send_command('M105')
    _wait(done)
    time.sleep(0.2)
    assert ser.written[:4] == ['G91', 'G1 X0.010 F60', 'M400', 'G90']
    assert ser.written[4:6] == ['G0 X10 F3000', 'M105']


def test_sequence_stops_on_failure_and_restores_g90(running_protocol):
    p, ser = running_protocol
    ser.silent = {'G1 Z1 F300'}
    out = {}
    done = threading.Event()

    def cb(ok, res):
        out['ok'], out['res'] = ok, res
        done.set()

    p.send_sequence(['G91', 'G1 Z1 F300', 'G1 Z-1 F300'],
                    finally_commands=['G90'], callback=cb, timeout_ms=300)
    _wait(done)
    assert out['ok'] is False
    # The lift was sent ONCE (no retry of a relative move), the return move
    # was never sent, and G90 was still restored.
    assert ser.written.count('G1 Z1 F300') == 1
    assert 'G1 Z-1 F300' not in ser.written
    assert ser.written[-1] == 'G90'


def test_send_immediate_bypasses_queue(running_protocol):
    p, ser = running_protocol
    ser.silent = {'M400'}
    p.send_command('M400', timeout_ms=3000, retries=0)  # blocks the queue
    time.sleep(0.1)
    assert p.send_immediate('M112')
    assert 'M112' in ser.written
    assert ser.written.index('M112') == ser.written.index('M400') + 1


# ---------------------------------------------------------------------------
# StepperStage.run_relative_block
# ---------------------------------------------------------------------------

class FakeProtocol:
    is_running = True

    def __init__(self, fail_line=None):
        self.sequences = []
        self.fail_line = fail_line

    def send_sequence(self, commands, finally_commands=None, callback=None,
                      timeout_ms=0):
        self.sequences.append((list(commands), list(finally_commands or [])))
        res, ok = [], True
        for c in commands:
            acked = c != self.fail_line
            res.append((c, acked, 'ok' if acked else ''))
            if not acked:
                ok = False
                break
        for c in finally_commands or []:
            res.append((c, True, 'ok'))
        callback(ok, res)

    def send_command(self, command, callback=None, timeout_ms=0, retries=None):
        if command == 'M114' and callback:
            callback(True, 'X:120.05 Y:100.00 Z:35.00 E:0.00 Count X:1 Y:2 Z:3\nok')


def _stage(proto):
    from raman_control.hardware.stepper_stage import StepperStage, StagePosition
    st = StepperStage(StageConfig())
    st._protocol = proto
    st._connected = True
    st._referenced = {'X': True, 'Y': True, 'Z': True}
    st._target_position = StagePosition(*MID)
    return st


def test_relative_block_wraps_g91_g90_and_tracks_base():
    proto = FakeProtocol()
    st = _stage(proto)
    out = []
    assert st.run_relative_block(['G1 X0.010 F60', 'M400', 'G4 S2'],
                                 callback=lambda ok, r: out.append(ok))
    cmds, fin = proto.sequences[0]
    assert cmds == ['G91', 'G1 X0.010 F60', 'M400', 'G4 S2']
    assert fin == ['G90']
    assert out == [True]
    assert st._target_position.x == pytest.approx(MID[0] + 0.010)
    assert st._target_position.z == pytest.approx(MID[2])


def test_relative_block_failure_reanchors_from_m114():
    proto = FakeProtocol(fail_line='G1 X5 F1500')
    st = _stage(proto)
    out = []
    st.run_relative_block(['G1 X-5 F1500', 'G1 X5 F1500'],
                          callback=lambda ok, r: out.append(ok))
    assert out == [False]
    assert st._target_position.x == pytest.approx(120.05)


def test_relative_block_refusals():
    st = _stage(FakeProtocol())
    assert not st.run_relative_block(['G90'])
    assert not st.run_relative_block(['G28 Z'])
    st._referenced['Z'] = False
    assert not st.run_relative_block(['G1 X0.010 F60'])


# ---------------------------------------------------------------------------
# Session output
# ---------------------------------------------------------------------------

def test_write_tiff_uncompressed_16bit(tmp_path):
    from PIL import Image
    arr = (np.arange(64 * 48, dtype=np.uint16).reshape(48, 64) * 13) % 1024
    p = str(tmp_path / 'a.tif')
    write_tiff(p, arr)
    with Image.open(p) as im:
        assert im.tag_v2.get(259) == 1          # Compression = none
        back = np.array(im)
    assert back.dtype == np.uint16
    assert np.array_equal(back, arr)


def test_session_checklist_and_verify(tmp_path):
    import datetime
    s = CalibrationSession(str(tmp_path), date=datetime.date(2026, 10, 23))
    s.create()
    assert os.path.basename(s.root) == 'unit1_motion_2026-10-23'
    for f in mtp.FOLDERS:
        assert os.path.isdir(s.folder(f))
    s.set_row(mtp.SHEET_T0, 3, STATUS_DONE, 'note three', time='10:00:00')
    s.set_row(mtp.SHEET_SETUP, 11, STATUS_DONE, 'M92 X100 Y100 Z400 E280')
    s.set_row(mtp.SHEET_T6, 55, STATUS_PROBLEM, 'blurry')
    s.save()

    s2 = CalibrationSession(str(tmp_path), date=datetime.date(2026, 10, 23))
    s2.load()
    assert s2.get_row(mtp.SHEET_T0, 3)['notes'] == 'note three'

    out = s2.write_checklist()
    wb = openpyxl.load_workbook(out)
    ws = wb[mtp.SHEET_T0]
    row = next(r for r in ws.iter_rows(min_row=5) if r[0].value == 3)
    assert (row[4].value, row[5].value, row[6].value) == (
        STATUS_DONE, '10:00:00', 'note three')
    ws6 = wb[mtp.SHEET_T6]
    row = next(r for r in ws6.iter_rows(min_row=5) if r[0].value == 55)
    assert row[4].value == STATUS_PROBLEM
    # The dropdown validation and summary formulas survive.
    assert ws.data_validations.dataValidation
    assert str(wb['Summary']['C6'].value).startswith('=COUNTIF')

    res = s2.verify_files()
    assert not res['ok_counts']
    for name in [f'T0_{i:02d}.tif' for i in range(1, 11)]:
        write_tiff(os.path.join(s2.folder('T0_noise'), name),
                   np.zeros((4, 4), np.uint16))
    res = s2.verify_files()
    assert res['counts']['T0_noise'] == (10, 10)
    assert 'T0_noise/T0_01.tif' not in res['missing']
    assert s2.write_data_sheet().endswith('data_sheet.csv')


# ---------------------------------------------------------------------------
# Worker end to end (fake stage + camera)
# ---------------------------------------------------------------------------

class FakeMotion:
    is_connected = True
    all_referenced = True

    def __init__(self, grabber, fail_on=None):
        self.blocks = []
        self.grabber = grabber
        self.fail_on = fail_on
        self.pos = list(MID)

    def query_position_blocking(self, timeout=0):
        return tuple(self.pos)

    def run_relative_block_blocking(self, lines, timeout=0):
        self.blocks.append(list(lines))
        res = []
        for ln in lines:
            if ln == self.fail_on:
                res.append((ln, False, ''))
                return False, res
            d, _ = mtp.parse_g1(ln)
            for i, a in enumerate('XYZ'):
                self.pos[i] += d.get(a, 0.0)
            res.append((ln, True, 'ok'))
        self.grabber.offset_px = (self.pos[0] - MID[0]) * 1000 / 0.5  # 0.5 um/px
        return True, res


class FakeGrabber:
    """Serves a textured frame shifted by the stage offset; frames are
    'fresh' immediately. Streams frames to a sink when recording."""

    def __init__(self):
        rng = np.random.default_rng(0)
        base = rng.integers(100, 900, size=(64, 64)).astype(np.float32)
        import cv2
        self.base = cv2.GaussianBlur(base, (0, 0), 2)
        self.offset_px = 0.0
        self.count = 0
        self._sink = None
        self._stop = threading.Event()

    def _frame(self):
        import cv2
        m = np.float32([[1, 0, self.offset_px], [0, 1, 0]])
        img = cv2.warpAffine(self.base, m, (64, 64), borderMode=cv2.BORDER_WRAP)
        self.count += 1
        return img.astype(np.uint16)

    def wait_fresh(self, after_t, timeout):
        return self._frame(), max(after_t, time.time()), self.count

    def set_sink(self, sink):
        self._sink = sink
        if sink is not None:
            self._stop.clear()

            def feed():
                while not self._stop.is_set() and self._sink is not None:
                    s = self._sink
                    if s:
                        s(self._frame(), time.time(), self.count)
                    time.sleep(0.02)
            threading.Thread(target=feed, daemon=True).start()
        else:
            self._stop.set()


def _run_worker(qapp, test, tmp_path, grabber, motion, steps=None):
    from raman_control.gui.threads.calibration_thread import CalibrationWorker
    rows, finished = [], []
    w = CalibrationWorker(test, motion, grabber, out_dir=str(tmp_path),
                          exposure_s=0.0, bit_depth=10,
                          stage_limits=mtp.stage_limits(StageConfig()),
                          steps_per_mm=PRUSA_M92, steps=steps)
    w.row_result.connect(lambda *a: rows.append(a))
    w.finished_test.connect(lambda *a: finished.append(a))
    w.run()  # synchronously, on this thread
    return rows, finished


def test_worker_t2_staircase(qapp, tmp_path):
    g = FakeGrabber()
    m = FakeMotion(g)
    test = mtp.build_t2('X')
    rows, finished = _run_worker(qapp, test, tmp_path, g, m)
    assert finished[-1][1] is True
    files = sorted(f for f in os.listdir(tmp_path) if f.endswith('.tif'))
    assert files == [f'T2_X_step{i:02d}.tif' for i in range(16)]
    assert all(b[0].startswith('G1') or b[0] == 'M400' for b in m.blocks)
    by_row = {r[1]: r for r in rows}
    assert by_row[19][2] == STATUS_DONE        # image moved at most steps
    # 10 um step / 0.5 um per px = 20 px shift vs the previous image.
    dx = float(by_row[4][3].split('shift ')[1].split('/')[0])
    assert dx == pytest.approx(20.0, abs=0.5)
    assert os.path.exists(tmp_path / 'capture_log.csv')
    assert m.pos[0] == pytest.approx(MID[0] + 0.070)


def test_worker_stops_on_motion_failure(qapp, tmp_path):
    g = FakeGrabber()
    m = FakeMotion(g, fail_on='G1 X5 F1500')
    rows, finished = _run_worker(qapp, mtp.build_t3('X'), tmp_path, g, m)
    assert finished[-1][1] is False
    assert any(r[2] == STATUS_PROBLEM for r in rows)
    assert len(m.blocks) == 2   # ref + first return, then stopped


def test_worker_refuses_unsafe_start(qapp, tmp_path):
    g = FakeGrabber()
    m = FakeMotion(g)
    m.pos = [2.0, 100.0, 35.0]     # X-5 would leave travel
    rows, finished = _run_worker(qapp, mtp.build_t3('X'), tmp_path, g, m)
    assert finished[-1][1] is False
    assert m.blocks == []


def test_worker_t4_recording(qapp, tmp_path):
    g = FakeGrabber()
    m = FakeMotion(g)
    t4 = mtp.build_t4()
    step = next(s for s in t4.steps if s.kind == 'record')
    short = mtp.Step(step.row, step.action, gcode=step.gcode,
                     image=step.image, kind='record', record_s_after=0.3)
    rows, finished = _run_worker(qapp, t4, tmp_path, g, m, steps=[short])
    assert finished[-1][1] is True
    run_dir = tmp_path / 'T4_X_run1'
    frames = [f for f in os.listdir(run_dir) if f.endswith('.tif')]
    assert len(frames) > 10
    assert (run_dir / 'timestamps.csv').exists()
    assert m.blocks == [['G1 X5 F1500', 'G1 X-5 F1500', 'M400']]


class StickyMotion(FakeMotion):
    """Stage that moves the image once (stick-slip) and then not at all."""

    def run_relative_block_blocking(self, lines, timeout=0):
        ok, res = super().run_relative_block_blocking(lines, timeout)
        self.grabber.offset_px = 20.0 if len(self.blocks) >= 4 else 0.0
        return ok, res


def test_worker_t2_flags_single_jump_as_problem(qapp, tmp_path):
    """Regression: a run where the image jumps once and then never moves
    must fail the 'image shifted at most steps' check (shift vs PREVIOUS)."""
    import dataclasses
    g = FakeGrabber()
    m = StickyMotion(g)
    t2 = mtp.build_t2('X')
    # The panel runs a copy built like this; it must keep compare='previous'.
    sub = dataclasses.replace(t2, steps=tuple(t2.steps))
    assert sub.compare == 'previous'
    rows, finished = _run_worker(qapp, sub, tmp_path, g, m)
    check = next(r for r in rows if r[1] == 19)
    assert check[2] == STATUS_PROBLEM
    assert '1/15' in check[3]


def test_t2_extra_run_is_separate_from_checklist():
    base, extra = mtp.build_t2('X'), mtp.build_t2('X', 60)
    assert base == mtp.build_t2('X', 10.0)          # protocol run unchanged
    assert extra.folder == 'T2_steps_X_60um'
    assert extra.folder not in mtp.FOLDERS
    assert extra.sheet not in mtp.SHEET_T2.values()
    assert extra.steps[2].image == 'T2_X_60um_step00.tif'
    assert extra.steps[3].gcode[0] == 'G1 X0.060 F60'
    issues = mtp.check_test(extra, start_xyz=MID,
                            limits=mtp.stage_limits(StageConfig()),
                            steps_per_mm=PRUSA_M92)
    assert not [i for i in issues if i.level != 'info']
    assert mtp.build_t2('Y', 30).steps[0].gcode == ('G1 Y0.030 F60',)
