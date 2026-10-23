"""
Autofocus scan worker (runs off the GUI thread).

Steps a Z axis across a set of positions, scores the focus at each step via a
caller-supplied callback, then moves to the position with the best score. Used
for both camera autofocus (Prusa Z + image contrast) and spectrometer
autofocus (Prusa Z or NanoDrive + spectrum intensity).

The move and score callbacks run inside this worker thread, so they may block.
They must be thread-safe with respect to the hardware they touch (the motion
thread, camera thread and spectrum thread all expose thread-safe accessors).
"""

import time
from typing import Callable, Optional, Sequence

from PyQt6.QtCore import QThread, pyqtSignal

from raman_control.config.logging_config import get_logger

logger = get_logger('autofocus_worker')


class AutofocusWorker(QThread):
    """Scan Z positions, score each, and move to the best.

    Signals:
        point: (z, score) emitted after each scored step
        progress: (percent, message) for the status-bar progress bar
        finished_focus: (success, best_z) when the scan ends
    """

    point = pyqtSignal(float, float)
    progress = pyqtSignal(int, str)
    finished_focus = pyqtSignal(bool, float)

    def __init__(
        self,
        positions: Sequence[float],
        move_fn: Callable[[float], bool],
        score_fn: Callable[[], float],
        settle_s: float = 0.1,
        label: str = "Autofocus",
        maximize: bool = True,
        prepare_fn: Optional[Callable[[], None]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._positions = [float(z) for z in positions]
        self._move_fn = move_fn
        self._score_fn = score_fn
        self._settle_s = settle_s
        self._label = label
        self._maximize = maximize  # True: best = highest score; False: lowest
        # Optional one-time setup run on THIS thread before the scan starts
        # (e.g. parking the NanoDrive at its initial position), so the caller
        # never has to do blocking hardware I/O on the GUI thread.
        self._prepare_fn = prepare_fn
        self._abort = False

    def _is_better(self, score: float, best: float) -> bool:
        return score > best if self._maximize else score < best

    def abort(self):
        self._abort = True

    def run(self):
        positions = self._positions
        if not positions:
            self.finished_focus.emit(False, 0.0)
            return

        # Bring the system to a known starting state before scoring anything.
        if self._prepare_fn is not None and not self._abort:
            try:
                self._prepare_fn()
            except Exception as e:  # noqa: BLE001 - never abort the scan for this
                logger.error(f"{self._label}: prepare step failed: {e}")

        best_z = positions[0]
        best_score = float('-inf') if self._maximize else float('inf')
        found = False
        n = len(positions)
        logger.info(f"{self._label}: scanning {n} positions "
                    f"[{positions[0]:.3f} .. {positions[-1]:.3f}]")

        for i, z in enumerate(positions):
            if self._abort:
                break
            try:
                moved = self._move_fn(z)
            except Exception as e:  # noqa: BLE001 - report and skip the step
                logger.error(f"{self._label}: move to {z} failed: {e}")
                moved = False
            if not moved:
                continue

            if self._settle_s > 0:
                time.sleep(self._settle_s)

            try:
                score = float(self._score_fn())
            except Exception as e:  # noqa: BLE001
                logger.error(f"{self._label}: scoring at {z} failed: {e}")
                continue

            self.point.emit(float(z), score)
            if self._is_better(score, best_score):
                best_score = score
                best_z = z
                found = True
            self.progress.emit(int((i + 1) / n * 100), f"{self._label} {i+1}/{n}")

        success = found and not self._abort
        if success:
            try:
                self._move_fn(best_z)  # park at the best-focus position
            except Exception as e:  # noqa: BLE001
                logger.error(f"{self._label}: move to best {best_z} failed: {e}")
            logger.info(f"{self._label}: best Z = {best_z:.3f} (score {best_score:.4g})")
        else:
            logger.info(f"{self._label}: {'aborted' if self._abort else 'no valid steps'}")
        self.finished_focus.emit(success, best_z)
