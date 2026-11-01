"""
Worker thread for the calibration (unit 1 motion test) tab.

Runs one test of motion_test_protocol: each step's G-code is executed as an
atomic relative block on the motion thread's queue (never on the GUI thread),
then a FRESH full-resolution raw frame is grabbed from the camera's raw-frame
tap and written as an uncompressed TIFF. Test 4 records every frame during
the move into a per-run folder.

The worker never touches the session object directly: checklist results are
emitted as signals and applied on the GUI thread.
"""

import csv
import os
import queue
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from raman_control.config.logging_config import get_logger
from raman_control.core import motion_test_protocol as mtp
from raman_control.core.calibration_session import (
    STATUS_DONE, STATUS_PROBLEM, write_tiff)

logger = get_logger('calibration_thread')


class FrameGrabber:
    """Keeps the newest raw frame of one camera (via its raw-frame tap) and
    can stream every frame to a recording sink."""

    def __init__(self):
        self._camera = None
        self._cond = threading.Condition()
        self._latest: Optional[Tuple[np.ndarray, float, int]] = None
        self._sink = None

    @property
    def camera(self):
        return self._camera

    def attach(self, camera) -> None:
        if camera is self._camera:
            return
        self.detach()
        self._camera = camera
        if camera is not None:
            camera.add_raw_frame_listener(self._on_raw)

    def detach(self) -> None:
        if self._camera is not None:
            self._camera.remove_raw_frame_listener(self._on_raw)
        self._camera = None
        with self._cond:
            self._latest = None

    def _on_raw(self, raw: np.ndarray, t: float, count: int) -> None:
        with self._cond:
            self._latest = (raw, t, count)
            sink = self._sink
            self._cond.notify_all()
        if sink is not None:
            sink(raw, t, count)

    def latest(self) -> Optional[Tuple[np.ndarray, float, int]]:
        with self._cond:
            return self._latest

    def wait_fresh(self, after_t: float, timeout: float):
        """First frame that arrived at or after ``after_t``, or None."""
        deadline = time.time() + timeout
        with self._cond:
            while True:
                if self._latest is not None and self._latest[1] >= after_t:
                    return self._latest
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)

    def set_sink(self, sink) -> None:
        with self._cond:
            self._sink = sink


class _SequenceWriter(threading.Thread):
    """Writes recorded frames to disk off the camera thread."""

    def __init__(self, run_dir: str, max_backlog: int = 400):
        super().__init__(name='T4Writer', daemon=True)
        self.run_dir = run_dir
        self._q: queue.Queue = queue.Queue(maxsize=max_backlog)
        self.rows: List[Tuple[int, int, float]] = []
        self.dropped = 0
        self.errors = 0
        self._idx = 0

    def put(self, raw: np.ndarray, t: float, count: int) -> None:
        try:
            self._q.put_nowait((raw, t, count))
        except queue.Full:
            self.dropped += 1

    def finish(self) -> None:
        self._q.put(None)
        self.join(timeout=120)

    def run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            raw, t, count = item
            name = f"frame_{self._idx:05d}.tif"
            try:
                write_tiff(os.path.join(self.run_dir, name), raw)
                self.rows.append((self._idx, count, t))
            except Exception as e:  # noqa: BLE001
                self.errors += 1
                logger.error(f"T4 frame write failed: {e}")
            self._idx += 1


def _shift(ref: np.ndarray, img: np.ndarray) -> Tuple[float, float]:
    """Sub-pixel translation of ``img`` relative to ``ref`` (phase correlation)."""
    import cv2
    a = ref.astype(np.float32)
    b = img.astype(np.float32)
    if a.ndim == 3:
        a, b = a.mean(axis=2), b.mean(axis=2)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), _resp = cv2.phaseCorrelate(a, b, win)
    return float(dx), float(dy)


def _sharpness(img: np.ndarray) -> float:
    """Normalized Laplacian variance (higher = sharper)."""
    import cv2
    a = img.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    m = float(a.mean()) or 1.0
    return float(cv2.Laplacian(a, cv2.CV_32F).var()) / (m * m)


class CalibrationWorker(QThread):
    """Runs the (non-manual) steps of one motion test."""

    progress = pyqtSignal(int, str)                 # percent, text
    log = pyqtSignal(str, str)                      # level, message
    row_result = pyqtSignal(str, int, str, str)     # sheet, row, status, notes
    image_captured = pyqtSignal(str)                # file path
    fps_measured = pyqtSignal(float)
    finished_test = pyqtSignal(str, bool, bool, str)  # key, ok, aborted, summary

    def __init__(self, test: mtp.TestDef, motion, grabber: FrameGrabber,
                 out_dir: str, exposure_s: float, bit_depth: int,
                 stage_limits: dict, steps_per_mm: Optional[dict] = None,
                 steps: Optional[List[mtp.Step]] = None, parent=None):
        super().__init__(parent)
        self._test = test
        self._motion = motion
        self._grabber = grabber
        self._out_dir = out_dir
        self._exposure_s = max(0.0, float(exposure_s))
        self._bit_depth = int(bit_depth or 0)
        self._limits = stage_limits
        self._steps_per_mm = steps_per_mm or None
        self._steps = [s for s in (steps if steps is not None else test.steps)
                       if s.kind != 'manual']
        self._abort = threading.Event()
        self._offset = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
        self.start_time: Optional[float] = None

    def abort(self) -> None:
        self._abort.set()

    @property
    def aborted(self) -> bool:
        return self._abort.is_set()

    # ------------------------------------------------------------------

    def _sleep_until(self, t_target: float) -> bool:
        """Abortable wait; returns False if aborted."""
        while not self._abort.is_set():
            remaining = t_target - time.time()
            if remaining <= 0:
                return True
            self._abort.wait(min(0.2, remaining))
        return False

    def _say(self, level: str, msg: str) -> None:
        getattr(logger, 'warning' if level == 'warning' else
                'error' if level == 'error' else 'info')(
            f"[{self._test.key}] {msg}")
        self.log.emit(level, msg)

    def _preflight(self) -> bool:
        blocks = [s.gcode for s in self._steps if s.gcode]
        if not blocks:
            return True
        if not (self._motion and self._motion.is_connected):
            self._say('error', "Stage not connected")
            return False
        if not self._motion.all_referenced:
            self._say('error', "Stage not referenced: home all axes first")
            return False
        pos = self._motion.query_position_blocking(5.0)
        if pos is None:
            self._say('error', "Could not read the stage position (M114)")
            return False
        self._say('info', f"Start position X{pos[0]:.3f} Y{pos[1]:.3f} "
                          f"Z{pos[2]:.3f} mm")
        issues = mtp.check_blocks(blocks, start_xyz=pos, limits=self._limits,
                                  steps_per_mm=self._steps_per_mm)
        ok = True
        for iss in issues:
            self._say(iss.level, iss.message)
            if iss.level == 'error':
                ok = False
        if not ok:
            self._say('error', "Safety check failed: test not started")
        return ok

    def _block_timeout(self, lines) -> float:
        return 30.0 + mtp.estimate_duration_s(
            mtp.TestDef('', '', '', '', (mtp.Step(0, '', gcode=tuple(lines)),)))

    def run(self) -> None:
        test = self._test
        self.start_time = time.time()
        os.makedirs(self._out_dir, exist_ok=True)
        self._say('info', f"{test.title} started")
        if not self._preflight():
            self.finished_test.emit(test.key, False, False, "Safety check failed")
            return

        n = len(self._steps)
        ref_img = None
        prev_img = None
        moved_flags: List[bool] = []
        n_images = 0
        failed = ''
        t0 = time.time()
        for i, s in enumerate(self._steps):
            if self._abort.is_set():
                break
            self.progress.emit(int(100 * i / max(1, n)),
                               f"{test.key}: {s.action}")

            if s.at_s is not None:
                if not self._sleep_until(t0 + s.at_s):
                    break

            if s.kind == 'check':
                self._emit_check(s, moved_flags)
                continue

            if s.kind == 'record':
                ok, note = self._record(s)
                self.row_result.emit(test.sheet, s.row,
                                     STATUS_DONE if ok else STATUS_PROBLEM, note)
                if not ok:
                    failed = note
                    break
                continue

            if s.gcode:
                ok, results = self._motion.run_relative_block_blocking(
                    list(s.gcode), timeout=self._block_timeout(s.gcode))
                if not ok:
                    bad = next((ln for ln, acked, _ in results if not acked),
                               'no response')
                    note = f"Motion block failed at '{bad}'; test stopped"
                    self.row_result.emit(test.sheet, s.row, STATUS_PROBLEM, note)
                    self._say('error', note)
                    failed = note
                    break
                for line in s.gcode:
                    deltas, _f = mtp.parse_g1(line)
                    for k, v in deltas.items():
                        self._offset[k] += v

            if not s.image:
                self.row_result.emit(test.sheet, s.row, STATUS_DONE,
                                     ', '.join(s.gcode))
                continue

            t_ready = time.time()
            after = t_ready + 2 * self._exposure_s + mtp.FRAME_MARGIN_S
            got = self._grabber.wait_fresh(after, timeout=5.0 + 3 * self._exposure_s)
            if got is None:
                note = ("No fresh camera frame (camera disconnected or live "
                        "acquisition paused?)")
                self.row_result.emit(test.sheet, s.row, STATUS_PROBLEM, note)
                self._say('error', note)
                failed = note
                break
            raw, t_frame, count = got
            path = os.path.join(self._out_dir, s.image)
            try:
                write_tiff(path, raw)
            except Exception as e:  # noqa: BLE001
                note = f"Could not save {s.image}: {e}"
                self.row_result.emit(test.sheet, s.row, STATUS_PROBLEM, note)
                self._say('error', note)
                failed = note
                break
            n_images += 1

            max_val = int(raw.max())
            sat_level = ((1 << self._bit_depth) - 1 if self._bit_depth
                         else (255 if raw.dtype == np.uint8 else 65535))
            saturated = int(np.count_nonzero(raw >= sat_level))

            ref = prev_img if test.compare == 'previous' else ref_img
            dx = dy = None
            if ref is not None and ref.shape == raw.shape:
                try:
                    dx, dy = _shift(ref, raw)
                    moved_flags.append(abs(dx) > 0.3 or abs(dy) > 0.3)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"shift failed: {e}")
            try:
                sharp = _sharpness(raw)
            except Exception:  # noqa: BLE001
                sharp = float('nan')
            if ref_img is None:
                ref_img = raw
            prev_img = raw

            pos = self._motion.query_position_blocking(3.0) if self._motion else None
            notes = []
            if dx is not None:
                notes.append(f"shift {dx:+.2f}/{dy:+.2f} px vs "
                             f"{'prev' if test.compare == 'previous' else 'ref'}")
            notes.append(f"sharp {sharp:.4g}")
            if saturated:
                notes.append(f"SATURATED {saturated} px")
                self._say('warning', f"{s.image}: {saturated} saturated pixels")
            if pos is None and s.gcode:
                notes.append("M114 n/a")

            record = {
                'file': s.image,
                'iso_time': time.strftime('%Y-%m-%dT%H:%M:%S',
                                          time.localtime(t_frame)),
                'elapsed_s': f"{t_frame - t0:.3f}",
                'm114_x': pos[0] if pos else '',
                'm114_y': pos[1] if pos else '',
                'm114_z': pos[2] if pos else '',
                'cmd_offset_x_um': f"{self._offset['X'] * 1000:.2f}",
                'cmd_offset_y_um': f"{self._offset['Y'] * 1000:.2f}",
                'cmd_offset_z_um': f"{self._offset['Z'] * 1000:.2f}",
                'sdk_frame_count': count,
                'max_value': max_val,
                'saturated_px': saturated,
                'shift_dx_px': '' if dx is None else f"{dx:.3f}",
                'shift_dy_px': '' if dy is None else f"{dy:.3f}",
                'shift_ref': test.compare,
                'gcode': ' | '.join(s.gcode),
            }
            try:
                _append_log(self._out_dir, record)
            except Exception as e:  # noqa: BLE001
                self._say('warning', f"capture_log.csv not written: {e}")

            self.image_captured.emit(path)
            self.row_result.emit(test.sheet, s.row, STATUS_DONE, '; '.join(notes))
            self._say('info', f"{s.image} saved ({'; '.join(notes)})")

        aborted = self._abort.is_set()
        ok = not failed and not aborted
        self.progress.emit(100 if ok else 0, "")
        summary = (f"{n_images} images saved" if ok else
                   ("Aborted" if aborted else f"Stopped: {failed}"))
        if aborted:
            self._say('warning', f"{test.title} aborted after {n_images} images")
        self.finished_test.emit(test.key, ok, aborted, summary)

    def _emit_check(self, s: mtp.Step, moved_flags: List[bool]) -> None:
        total = len(moved_flags)
        moved = sum(moved_flags)
        if total and moved >= max(1, int(0.6 * total)):
            self.row_result.emit(self._test.sheet, s.row, STATUS_DONE,
                                 f"Image shifted at {moved}/{total} steps")
        else:
            note = (f"Image shifted at only {moved}/{total} steps. Check setup: "
                    f"motors idle-disabled, frozen camera frame, or steps lost "
                    f"in backlash")
            self.row_result.emit(self._test.sheet, s.row, STATUS_PROBLEM, note)
            self._say('warning', note)

    def _record(self, s: mtp.Step) -> Tuple[bool, str]:
        """Test 4: record every frame while the moves run, plus 5 s after."""
        run_dir = os.path.join(self._out_dir, s.image)
        os.makedirs(run_dir, exist_ok=True)
        for f in os.listdir(run_dir):
            if f.lower().endswith('.tif') or f == 'timestamps.csv':
                os.remove(os.path.join(run_dir, f))
        writer = _SequenceWriter(run_dir)
        writer.start()
        self._grabber.set_sink(writer.put)
        ok, note = True, ''
        try:
            if not self._sleep_until(time.time() + 1.0):  # pre-roll
                return False, "Aborted"
            t_cmd = time.time()
            ok, results = self._motion.run_relative_block_blocking(
                list(s.gcode), timeout=self._block_timeout(s.gcode))
            t_done = time.time()
            if not ok:
                bad = next((ln for ln, acked, _ in results if not acked),
                           'no response')
                note = f"Motion block failed at '{bad}'"
            # Keep recording after the stage stopped (not cut short by abort,
            # so the run that is in progress is complete).
            time.sleep(s.record_s_after)
        finally:
            self._grabber.set_sink(None)
            writer.finish()

        rows = writer.rows
        with open(os.path.join(run_dir, 'timestamps.csv'), 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['# move_sent_rel_s', 0.0])
            if ok:
                w.writerow(['# motion_done_rel_s (M400 ack, approx)',
                            f"{t_done - t_cmd:.3f}"])
            w.writerow(['# gcode', ' | '.join(s.gcode)])
            w.writerow(['frame_index', 'sdk_frame_count', 't_rel_s'])
            for idx, count, t in rows:
                w.writerow([idx, count, f"{t - t_cmd:.4f}"])
        fps = 0.0
        if len(rows) > 1 and rows[-1][2] > rows[0][2]:
            fps = (len(rows) - 1) / (rows[-1][2] - rows[0][2])
            self.fps_measured.emit(fps)
        if not ok:
            self._say('error', note)
            return False, note
        note = (f"{len(rows)} frames at {fps:.1f} fps; motion done at "
                f"+{t_done - t_cmd:.2f} s")
        if writer.dropped or writer.errors:
            note += f"; {writer.dropped} dropped, {writer.errors} write errors"
            self._say('warning', note)
        self._say('info', f"{s.image}: {note}")
        return True, note


def _append_log(folder: str, record: dict) -> None:
    cols = ['file', 'iso_time', 'elapsed_s', 'm114_x', 'm114_y', 'm114_z',
            'cmd_offset_x_um', 'cmd_offset_y_um', 'cmd_offset_z_um',
            'sdk_frame_count', 'max_value', 'saturated_px',
            'shift_dx_px', 'shift_dy_px', 'shift_ref', 'gcode']
    path = os.path.join(folder, 'capture_log.csv')
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(record)
