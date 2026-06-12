"""
Acquisition worker thread for the Current-FoV acquisition mode.

Runs the full per-point acquisition loop OFF the GUI thread:

    1. (once, optional) Autofocus the widefield camera with the Prusa stage.
    2. Switch to the objective (the Raman signal is collected through it).
    3. For each generated sample point, in order:
        a. Move the objective over the point.
        b. (optional) Coarse-focus the Wasatch signal with the Prusa stage.
        c. (optional) Fine-focus the Wasatch signal with the NanoDrive, OR
           run a NanoDrive Z-stack if that option is enabled.
        d. (optional) Wait with the laser on before measuring.
        e. Acquire the spectra (repetitions per point).
        f. Save each spectrum to CSV with a metadata sidecar.

Assumes the system starts in the widefield view (the natural state right after
capture / segment / generate). The one-time widefield autofocus runs in that
view; the worker then moves to the objective coordinates for each point and the
run ends in objective view.

The worker only ever touches hardware through thread-safe accessors
(``MotionThread.move_*_blocking``, the spectrum thread's locked getters, and
the NanoPiezo's locked ``move_to``). Everything the GUI must do — switching the
displayed camera, updating the coordinate manager, refreshing buttons — is
emitted as a signal and handled by ``MainWindow`` on the GUI thread.

NEVER construct or run this on the GUI thread's behalf with blocking calls; it
is a QThread and its work happens in :meth:`run`.
"""

import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from raman_control.core.autofocus import AutofocusEngine
from raman_control.config.logging_config import get_logger

logger = get_logger('acquisition_thread')


@dataclass
class AcquisitionContext:
    """Everything the worker needs from the host window, passed explicitly so
    the worker has no hidden dependency on ``MainWindow``.

    Hardware objects are the real threads/devices; the providers are small
    callables returning cached/thread-safe values. Z ranges are ``(lower,
    upper)`` in their native units (mm for Prusa, um for NanoDrive)."""

    motion: object                       # MotionThread
    spectrometer: object                 # SpectrumThread
    nanodrive: object                    # NanoPiezo (may be None)

    frame_provider: Callable[[], Optional[np.ndarray]]   # widefield frame
    position_provider: Callable[[], Tuple[float, float, float]]  # cached phys
    wavelengths_nm_provider: Callable[[], Optional[np.ndarray]]

    # Wide -> objective calibration offset (mm)
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    excitation_nm: float = 785.0

    # Z scan ranges (lower, upper)
    wf_prusa_range: Tuple[float, float] = (0.0, 0.0)   # mm, widefield view
    obj_prusa_range: Tuple[float, float] = (0.0, 0.0)  # mm, objective view
    nano_range: Tuple[float, float] = (0.0, 0.0)       # um

    # Camera-autofocus stabilization (seconds): per-step settle during the
    # widefield AF scan so vibration dies down before each frame is scored, and
    # a longer one-time settle before the segmentation capture (see CameraConfig).
    af_settle_s: float = 1.0
    af_capture_stabilization_s: float = 1.5

    # Robotic Deposition only (None for Current FoV): a loaded UNetSegmenter and
    # a snapshot of the Live-tab pattern settings, used to segment + generate
    # sample points on each droplet at run time.
    segmenter: object = None             # UNetSegmenter (loaded)
    pattern_settings: Optional[dict] = None

    # Current FoV only: the captured image + segmentation mask from the Live tab,
    # saved once at the start of the run (Robotic captures these per droplet).
    capture_image: Optional[np.ndarray] = None
    capture_mask: Optional[np.ndarray] = None

    # Sample/droplet identity for structured file names. droplet_position (well
    # name, e.g. 'B5') is used whenever known; sample_name/droplet_number come
    # from a loaded config. droplet_map (well -> info) is used by Robotic.
    droplet_map: Optional[dict] = None
    droplet_position: Optional[str] = None  # Current FoV: captured well name
    sample_name: Optional[str] = None
    droplet_number: object = None


class AcquisitionWorker(QThread):
    """Run a Current-FoV acquisition over a list of pre-generated points."""

    # done_steps, total_steps, message
    progress = pyqtSignal(int, int, str)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    # 'objective' or 'wide' — the GUI updates the coordinate manager / display
    view_changed = pyqtSignal(str)
    # aborted?, number of spectra files written
    finished_acq = pyqtSignal(bool, int)
    # Spectrometer autofocus -> Live-tab focus-score plot (mirrors the
    # interactive autofocus). focus_started(stage) clears that stage's curve;
    # focus_point(stage, z, score) adds a scored point. stage = 'prusa'/'nano'.
    focus_started = pyqtSignal(str)
    focus_point = pyqtSignal(str, float, float)
    # Drive the illumination LED on/off from the worker. Handled on the GUI
    # thread (which owns the LED control widget); True = on, False = off. Used
    # to light the sample for the widefield camera autofocus and turn it back
    # off afterwards.
    led_set = pyqtSignal(bool)

    def __init__(self, ctx: AcquisitionContext, points: Sequence,
                 params: dict, parent=None):
        super().__init__(parent)
        self._ctx = ctx
        self._points = list(points)
        self._p = dict(params)
        self._abort = False
        # Paused state: the run loop blocks while this Event is clear.
        self._resume = threading.Event()
        self._resume.set()
        self._n_saved = 0
        # Whether this run has already turned the laser on (and warmed it up).
        self._laser_warmed = False
        # Sample/droplet identity for file names. Current FoV takes it from the
        # context; Robotic Deposition overrides it per droplet.
        self._cur_droplet_position = ctx.droplet_position
        self._cur_sample_name = ctx.sample_name
        self._cur_droplet_number = ctx.droplet_number

    # Seconds to wait after the FIRST laser-on so a cold laser has time to
    # actually fire before the first spectrometer autofocus / measurement.
    _LASER_WARMUP_S = 5.0

    # --- external controls (call from the GUI thread) -----------------------
    def abort(self):
        self._abort = True
        self._resume.set()  # unblock a paused loop so it can exit

    def pause(self):
        self._resume.clear()

    def resume(self):
        self._resume.set()

    def _wait_if_paused(self):
        # Block here while paused, but wake promptly on abort.
        while not self._resume.wait(0.1):
            if self._abort:
                return

    def _interruptible_sleep(self, seconds: float):
        """Sleep, but wake promptly on abort and honour pause."""
        end = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < end:
            if self._abort:
                return
            self._wait_if_paused()
            if self._abort:
                return
            time.sleep(0.1)

    def _ensure_laser_on(self):
        """Turn the laser on before the first spectrometer measurement of the
        run. The first power-up can take a few seconds to actually fire, so wait
        once after enabling it (only the first time)."""
        if self._laser_warmed:
            return
        spec = self._ctx.spectrometer
        spec.set_laser_power_mw(self._p_float('acq_power', 50.0))
        spec.set_laser(True)
        self._laser_warmed = True
        self.status.emit(
            f"Laser on, warming up {int(self._LASER_WARMUP_S)} s…")
        self._interruptible_sleep(self._LASER_WARMUP_S)

    def _laser_off(self):
        """Turn the laser off (used at the end of the run)."""
        if not self._laser_warmed:
            return
        try:
            self._ctx.spectrometer.set_laser(False)
        except Exception:  # noqa: BLE001
            pass
        self._laser_warmed = False

    # --- illumination LED ---------------------------------------------------
    def _led_on(self):
        """Turn the illumination LED on for the widefield camera autofocus and
        let it settle before the first frame is scored. The GUI owns the LED
        widget, so we ask it via a queued signal rather than touching the serial
        port from this worker thread."""
        self.status.emit("Illumination LED on (widefield autofocus)")
        self.led_set.emit(True)
        self._interruptible_sleep(0.4)  # let illumination stabilise

    def _led_off(self):
        """Turn the illumination LED back off after the widefield autofocus."""
        self.led_set.emit(False)
        self.status.emit("Illumination LED off")

    def _p_int(self, key, default):
        try:
            return int(self._p.get(key, default))
        except (TypeError, ValueError):
            return default

    def _p_float(self, key, default):
        try:
            return float(self._p.get(key, default))
        except (TypeError, ValueError):
            return default

    # --- main loop ----------------------------------------------------------
    def run(self):
        try:
            self._run()
        except Exception as e:  # noqa: BLE001 - never let the thread crash silently
            logger.exception("Acquisition worker crashed")
            self.error.emit(f"Acquisition failed: {e}")
            self.finished_acq.emit(True, self._n_saved)

    def _run(self):
        ctx = self._ctx
        motion = ctx.motion

        if not self._points:
            self.error.emit("No sample points to acquire")
            self.finished_acq.emit(True, 0)
            return
        if not (motion and motion.is_connected and motion.all_referenced):
            self.error.emit("Stage not connected / referenced")
            self.finished_acq.emit(True, 0)
            return
        if not (ctx.spectrometer and ctx.spectrometer.is_connected):
            self.error.emit("Spectrometer not connected")
            self.finished_acq.emit(True, 0)
            return

        repeats = max(1, self._p_int('repeat_each_sample', 1))
        do_wf_af = bool(self._p.get('autofocus_widefield_prusa'))

        # Spectra saved per point already accounts for the z-stack (reps at every
        # Z step), so the progress total and the bar stay correct in both modes.
        total = repeats * len(self._points) * self._per_point_spectra()
        done = 0

        # --- Phase 1: one-time widefield camera autofocus (Prusa) -----------
        # Status emit so the step-preview cursor lands on the "Switch to
        # widefield view" step (no physical move; Phase 1 runs in widefield).
        self.status.emit("Switching to widefield view")
        cur = ctx.position_provider()
        wf_focus_z = cur[2]
        if do_wf_af:
            self.status.emit("Autofocusing widefield camera (Prusa)…")
            steps = max(2, self._p_int('cam_af_steps', 20))
            lo, hi = ctx.wf_prusa_range
            if hi > lo:
                # Illuminate the sample for the camera autofocus, then turn the
                # LED back off (it isn't wanted during the Raman measurements).
                # The finally guarantees the light goes off even on abort.
                self._led_on()
                try:
                    best = self._scan(
                        np.linspace(lo, hi, steps),
                        move_fn=lambda z: motion.move_z_blocking(z, timeout=30.0),
                        score_fn=self._camera_score,
                        maximize=True, settle=ctx.af_settle_s)
                    if best is not None:
                        wf_focus_z = best
                finally:
                    self._led_off()
            else:
                self.status.emit("Skipping widefield AF (no valid Z range)")
            if self._abort:
                return self._finish()

        # --- Switch to the objective for the actual Raman measurements ------
        # The per-point XY targets are the precomputed objective coords; the
        # base objective Z is the widefield focus plane plus the calibration
        # offset, then refined per point by the Wasatch autofocus.
        base_obj_z = wf_focus_z + ctx.offset[2]

        # Save the captured FoV + segmentation/points overlay once for this
        # sample. self._points is the Live tab's full point list, so manually
        # added points are included in the saved overlay.
        self._save_sample_images(ctx.capture_image, ctx.capture_mask, self._points)

        self.status.emit("Switching to objective view")
        self.view_changed.emit('objective')

        # --- Phase 2: per-point loop ----------------------------------------
        for rep in range(repeats):
            for idx, point in enumerate(self._points):
                if self._abort:
                    return self._finish()
                self._wait_if_paused()
                if self._abort:
                    return self._finish()

                label = self._point_label(point, idx)
                move_note = (f" (repeat {rep + 1}/{repeats})"
                             if repeats > 1 else "")
                done = self._measure_point(point, idx, rep, base_obj_z,
                                           done, total, label, move_note)

        self._finish()

    def _measure_point(self, point, idx, rep, base_obj_z, done, total,
                       label, move_note="") -> int:
        """Measure one sample point and return the updated ``done`` counter.

        Moves the objective over the point, (optionally) Wasatch-autofocuses,
        then acquires + saves ``reps_per_point`` spectra (or a Z-stack). Shared
        by Current FoV (`_run`) and Robotic Deposition (per generated point).
        Objective XY come from the point's precomputed obj coords, or fall back
        to its widefield stage coords + the wide->obj calibration offset (this is
        the path the runtime-generated robotic points take)."""
        ctx = self._ctx
        reps_per_point = max(1, self._p_int('reps_per_point', 1))
        z_stack = bool(self._p.get('z_stack'))
        do_wasatch_prusa = bool(self._p.get('autofocus_wasatch_prusa'))
        do_wasatch_nano = bool(self._p.get('autofocus_wasatch_nanodrive'))

        self.progress.emit(done, total, f"Moving to {label}{move_note}")

        ox = getattr(point, 'obj_stage_x', None)
        oy = getattr(point, 'obj_stage_y', None)
        if ox is None or oy is None:
            base = ctx.position_provider()
            ox = getattr(point, 'stage_x', base[0]) + ctx.offset[0]
            oy = getattr(point, 'stage_y', base[1]) + ctx.offset[1]
        if not ctx.motion.move_xyz_blocking(ox, oy, base_obj_z, timeout=60.0):
            self.error.emit(f"Move to {label} failed, skipping")
            return done

        # The laser must be on for any spectrometer signal (autofocus or
        # measurement). Turn it on once, before the first point's spectrometer
        # work, with a one-time warm-up wait.
        if not self._abort:
            self._ensure_laser_on()

        # Per-point coarse Wasatch autofocus (Prusa)
        if do_wasatch_prusa and not self._abort:
            self.status.emit(f"Wasatch autofocus (Prusa) at {label}…")
            self._wasatch_prusa_af()

        # Per-point fine focus: NanoDrive AF, OR a Z-stack if enabled. The
        # z-stack acquires reps_per_point spectra at EACH Z step and reports its
        # own per-step progress, so it owns the rest of this point's work.
        if z_stack and not self._abort:
            return self._z_stack(point, idx, rep, done, total, label, move_note)
        if do_wasatch_nano and not self._abort:
            self.status.emit(f"Wasatch autofocus (NanoDrive) at {label}…")
            self._wasatch_nano_af()

        # Optional wait with the laser on before measuring.
        if self._p.get('wait_before_measure') and not self._abort:
            self._wait_with_laser(self._p_float('wait_time_s', 0.0))

        # Acquire + save the requested repetitions at this point.
        for k in range(reps_per_point):
            if self._abort:
                return done
            self._wait_if_paused()
            self.progress.emit(done, total,
                               f"Acquiring {label}  {k + 1}/{reps_per_point}")
            spectrum = self._acquire_one()
            if spectrum is not None:
                self._save(point, idx, rep, k, spectrum)
            done += 1
            self.progress.emit(min(done, total), total,
                               f"Saved {label}  {k + 1}/{reps_per_point}")
        return done

    def _finish(self):
        # Always turn the laser off at the end of the run (completed OR stopped).
        # Status emit so the cursor lands on the "Turn laser off" preview step.
        self.status.emit("Laser off (end of run)")
        self._laser_off()
        # The run leaves the system in whatever view it reached: if it got to
        # the objective, the stage is physically there and the coordinate
        # manager already reflects objective view, so they stay consistent.
        # Switching the bookkeeping back to widefield WITHOUT a physical move
        # would desync it from the real position — the user switches back with
        # the normal (safe) camera-switch button when ready.
        #
        # Emit the status first and finished_acq LAST: finished_acq triggers the
        # host's cleanup (which waits for this thread to exit), so nothing should
        # run in run() after it returns.
        if self._abort:
            self.status.emit("Acquisition stopped")
        else:
            self.status.emit(f"Acquisition complete: {self._n_saved} spectra saved")
        self.finished_acq.emit(self._abort, self._n_saved)

    # --- focusing helpers ---------------------------------------------------
    def _scan(self, positions, move_fn, score_fn, maximize, settle, stage=None):
        """Generic focus scan: move to each Z, score, return to the best.

        If ``stage`` is given ('prusa'/'nano'), each scored point is streamed to
        the Live-tab focus-score plot (clearing it first), exactly like the
        interactive autofocus does.

        Returns the best Z (and leaves the hardware there), or None if the
        scan was aborted before any point was scored."""
        if stage:
            self.focus_started.emit(stage)
        best_z = None
        best_s = None
        for z in positions:
            if self._abort:
                break
            self._wait_if_paused()
            if self._abort:
                break
            move_fn(float(z))
            time.sleep(settle)
            s = score_fn()
            if s is None:
                continue
            if stage:
                self.focus_point.emit(stage, float(z), float(s))
            if best_s is None or (s > best_s if maximize else s < best_s):
                best_s, best_z = s, float(z)
        if best_z is not None and not self._abort:
            move_fn(best_z)
        return best_z

    def _camera_score(self) -> float:
        frame = self._ctx.frame_provider()
        if frame is None:
            return 0.0
        return float(AutofocusEngine.calculate_tenengrad(frame))

    def _spectrum_score(self, reps: int, expo_ms: float) -> float:
        """Median of `reps` FRESH spectra → Raman ratio score (lower is
        better). The median rejects cosmic rays; using independent exposures
        (acquire_frames) makes the rejection meaningful."""
        spec = self._ctx.spectrometer
        frames = spec.acquire_frames(max(1, reps))
        if frames is None or not len(frames):
            return float('inf')
        median = np.median(frames, axis=0)
        wl_nm = self._ctx.wavelengths_nm_provider()
        try:
            return float(AutofocusEngine.calculate_ratio_score(wl_nm, median))
        except Exception:  # noqa: BLE001
            return float('inf')

    def _wasatch_prusa_af(self):
        ctx = self._ctx
        lo, hi = ctx.obj_prusa_range
        if hi <= lo:
            return
        steps = max(2, self._p_int('spec_af_coarse', 20))
        expo = self._p_float('spec_af_exposure', 100.0)
        power = self._p_float('spec_af_power',
                              self._p_float('acq_power', 50.0))
        reps = self._p_int('spec_af_reps', 3)
        restore = self._apply_spec_settings(expo, power)
        try:
            self._scan(np.linspace(lo, hi, steps),
                       move_fn=lambda z: ctx.motion.move_z_blocking(z, timeout=30.0),
                       score_fn=lambda: self._spectrum_score(reps, expo),
                       maximize=False, settle=0.05, stage='prusa')
        finally:
            restore()

    def _wasatch_nano_af(self):
        ctx = self._ctx
        nano = ctx.nanodrive
        if not (nano and getattr(nano, 'is_connected', False)):
            return
        lo, hi = ctx.nano_range
        if hi <= lo:
            return
        steps = max(2, self._p_int('spec_af_fine', 10))
        expo = self._p_float('spec_af_exposure', 100.0)
        power = self._p_float('spec_af_power',
                              self._p_float('acq_power', 50.0))
        reps = self._p_int('spec_af_reps', 3)
        restore = self._apply_spec_settings(expo, power)
        try:
            self._scan(np.linspace(lo, hi, steps),
                       move_fn=lambda z: (nano.move_to(z=float(z)) or True),
                       score_fn=lambda: self._spectrum_score(reps, expo),
                       maximize=False, settle=0.03, stage='nano')
        finally:
            restore()

    def _apply_spec_settings(self, expo_ms: float, power_mw: float):
        """Set spectrometer exposure/power; return a callable restoring the
        acquisition exposure/power afterwards."""
        spec = self._ctx.spectrometer
        prev_expo = self._p_float('acq_exposure', expo_ms)
        prev_power = self._p_float('acq_power', power_mw)
        spec.set_integration_time(expo_ms)
        spec.set_laser_power_mw(power_mw)

        def _restore():
            spec.set_integration_time(prev_expo)
            spec.set_laser_power_mw(prev_power)
        return _restore

    # --- acquisition + saving ----------------------------------------------
    def _wait_with_laser(self, seconds: float):
        if seconds <= 0:
            return
        spec = self._ctx.spectrometer
        spec.set_laser_power_mw(self._p_float('acq_power', 50.0))
        spec.start_acquisition()
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self._abort:
                return
            self._wait_if_paused()
            time.sleep(0.1)

    def _acquire_one(self) -> Optional[np.ndarray]:
        """Acquire one fresh spectrum at the acquisition exposure/power settings.

        Uses the spectrometer thread's coordinated single-shot capture so each
        spectrum is a genuine fresh exposure (not a read of the cached
        free-running frame). Multiple spectra per point come from 'Repetitions
        per point', not from in-spectrum averaging."""
        spec = self._ctx.spectrometer
        expo = self._p_float('acq_exposure', 100.0)
        power = self._p_float('acq_power', 50.0)
        spec.set_integration_time(expo)
        spec.set_laser_power_mw(power)
        return spec.acquire_average(1)

    def _z_steps(self) -> int:
        """Number of NanoDrive Z positions in a z-stack (matches the NanoDrive
        Steps setting; at least 2)."""
        return max(2, self._p_int('spec_af_fine', 10))

    def _per_point_spectra(self) -> int:
        """Spectra saved per point: reps_per_point, multiplied by the z-stack
        step count when z-stack is enabled (reps at every Z step)."""
        reps = max(1, self._p_int('reps_per_point', 1))
        return reps * (self._z_steps() if self._p.get('z_stack') else 1)

    def _z_stack(self, point, idx: int, rep: int, done: int, total: int,
                 label: str, note: str = "") -> int:
        """Acquire a NanoDrive Z-stack at the current point: at EACH Z step,
        acquire and save ``reps_per_point`` spectra (each file tagged with its
        z-step index and spectrum index). Reports per-step progress and returns
        the updated ``done`` counter."""
        reps = max(1, self._p_int('reps_per_point', 1))
        # Clear the NanoDrive focus-score curve so a stale autofocus curve from a
        # previous point/run isn't left on the plot during the z-stack.
        self.focus_started.emit('nano')
        nano = self._ctx.nanodrive
        if not (nano and getattr(nano, 'is_connected', False)):
            # No piezo -> acquire reps spectra at the current Z (single step).
            for k in range(reps):
                if self._abort:
                    return done
                self._wait_if_paused()
                self.progress.emit(done, total,
                                   f"Acquiring {label}  {k + 1}/{reps}")
                spectrum = self._acquire_one()
                if spectrum is not None:
                    self._save(point, idx, rep, k, spectrum)
                done += 1
            return done
        lo, hi = self._ctx.nano_range
        steps = self._z_steps()
        for zi, z in enumerate(np.linspace(lo, hi, steps)):
            if self._abort:
                return done
            self._wait_if_paused()
            nano.move_to(z=float(z))
            time.sleep(0.03)
            for k in range(reps):
                if self._abort:
                    return done
                self._wait_if_paused()
                self.progress.emit(
                    done, total,
                    f"Z-stack {label}: z-step {zi + 1}/{steps}, "
                    f"spectrum {k + 1}/{reps}{note}")
                spectrum = self._acquire_one()
                if spectrum is not None:
                    self._save(point, idx, rep, k, spectrum,
                               z_index=zi, z_um=float(z))
                done += 1
        return done

    def _point_label(self, point, idx: int) -> str:
        # Positional "point N/M" (1-based, N total). The GUI step-preview cursor
        # matches this against its positional "[point N/M]" headers, so the label
        # must be positional too (a custom/id-based label would not line up with
        # the right point block and the cursor would stick on the first point).
        n = len(self._points)
        return f"point {idx + 1}/{n}" if n else f"point {idx + 1}"

    # --- structured file naming ---------------------------------------------
    @staticmethod
    def _sanitize(value) -> str:
        """Make a value safe + tidy for a file name: keep alphanumerics, turn
        everything else (incl. spaces) into '_', and collapse repeats."""
        out = ''.join(c if c.isalnum() else '_'
                      for c in str(value).strip()).strip('_')
        while '__' in out:
            out = out.replace('__', '_')
        return out or 'NA'

    def _point_position_token(self, point, idx: int, pattern: str) -> str:
        """Pattern-specific point-position as ``name_value`` pairs from the
        point's coordinates_order: grid -> row_R_col_C, ring -> ring_R_ray_N,
        line -> point_N, else point_<id>."""
        order = getattr(point, 'coordinates_order', None) or {}
        if pattern == 'grid' and 'row' in order:
            return f"row_{order['row']}_col_{order['col']}"
        if pattern == 'ring' and 'ring' in order:
            return f"ring_{order['ring']}_ray_{order['line']}"
        if pattern == 'line' and 'line_pos' in order:
            return f"point_{order['line_pos']}"
        return f"point_{getattr(point, 'id', idx + 1)}"

    # Point-position fields per sampling pattern (the only part of the file name
    # that varies by pattern). Kept in sync with _point_position_token.
    _POINTPOS_FORMAT = {
        'grid': 'row_<r>_col_<c>',
        'ring': 'ring_<r>_ray_<n>',
        'line': 'point_<n>',
        'random': 'point_<n>',
    }

    @staticmethod
    def filename_format(pattern: str = '', prefix: str = '',
                        current_fov: bool = False,
                        sample_prefix: str = '') -> str:
        """Human-readable file-name format for the given sampling pattern, shown
        in the Data Save panel. A non-empty ``prefix`` is prepended. In Current
        FoV mode the sample/droplet/position fields are dropped (naming starts at
        the pattern). When a config is loaded, ``sample_prefix`` (e.g. 'patient')
        is shown explicitly as ``<sample_prefix>_<name>`` instead of an opaque
        ``<sample_name>``. The point-position fields change with the pattern
        (grid -> row_/col_, ring -> ring_/ray_, line/random -> point_). Kept in
        sync with _build_filename."""
        pos = AcquisitionWorker._POINTPOS_FORMAT.get(pattern, 'point_<n>')
        ptype = pattern or '<type>'
        parts = []
        if prefix and prefix.strip():
            parts.append(AcquisitionWorker._sanitize(prefix))
        if not current_fov:
            sample = (f"{AcquisitionWorker._sanitize(sample_prefix)}_<name>"
                      if sample_prefix else "<sample_name>")
            parts.append(f"{sample}_droplet_<#>_position_<well>")
        parts.append(f"pattern_{ptype}_{pos}_spectra_<n>_date_<YYYYMMDD>_"
                     "repeat_<n>")
        return "_".join(parts)

    def _build_filename(self, point, idx, rep, k, z_index=None) -> str:
        """Structured spectrum file name, every field as ``name_value`` joined
        by '_':

            <sample_name>_droplet_<#>_position_<well>_pattern_<type>_<pointpos>_
            spectra_<n>_date_<YYYYMMDD>_repeat_<n>

        The sample name comes straight from the config (it already carries its
        prefix, e.g. 'patient_304'). The droplet position (well, e.g. B5) is
        included whenever known; sample name + droplet number only when a config
        is loaded. The point-position adapts to the pattern (row_/col_,
        ring_/ray_, or point_). A z-stack index is appended.
        """
        pattern = str(self._p.get('sample_pattern') or 'pattern')
        prefix = str(self._p.get('filename_prefix') or '').strip()
        parts = []
        if prefix:
            # Optional user prefix, prepended in front of the default name.
            parts.append(self._sanitize(prefix))
        # Current FoV passes no sample/droplet/position (they stay None), so the
        # name starts at the pattern there; Robotic fills them per droplet.
        if self._cur_sample_name:
            # The config sample name already carries its own prefix (e.g.
            # 'patient_1'), so use it as-is rather than a hard-coded 'sample_'.
            parts.append(self._sanitize(self._cur_sample_name))
        if self._cur_droplet_number not in (None, ''):
            parts.append(f"droplet_{self._sanitize(self._cur_droplet_number)}")
        if self._cur_droplet_position:
            parts.append(f"position_{self._sanitize(self._cur_droplet_position)}")
        parts.append(f"pattern_{self._sanitize(pattern)}")
        parts.append(self._point_position_token(point, idx, pattern))
        parts.append(f"spectra_{k + 1}")
        parts.append(f"date_{datetime.now().strftime('%Y%m%d')}")
        parts.append(f"repeat_{rep + 1}")
        if z_index is not None:
            parts.append(f"zstep_{z_index + 1}")
        return "_".join(parts)

    # --- per-sample image saving --------------------------------------------
    def _sample_image_basename(self, rep: Optional[int] = None) -> str:
        """Structured base name for a sample's saved images (no per-point /
        spectrum fields). Mirrors _build_filename's identity fields so the images
        sit alongside that sample's spectra."""
        pattern = str(self._p.get('sample_pattern') or 'pattern')
        prefix = str(self._p.get('filename_prefix') or '').strip()
        parts = []
        if prefix:
            parts.append(self._sanitize(prefix))
        if self._cur_sample_name:
            parts.append(self._sanitize(self._cur_sample_name))
        if self._cur_droplet_number not in (None, ''):
            parts.append(f"droplet_{self._sanitize(self._cur_droplet_number)}")
        if self._cur_droplet_position:
            parts.append(f"position_{self._sanitize(self._cur_droplet_position)}")
        parts.append(f"pattern_{self._sanitize(pattern)}")
        parts.append(f"date_{datetime.now().strftime('%Y%m%d')}")
        if rep is not None:
            parts.append(f"repeat_{rep + 1}")
        return "_".join(parts)

    @staticmethod
    def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
        """Convert a camera image (grayscale or RGB) to a uint8 BGR array ready
        for cv2 (camera frames are RGB, OpenCV writes BGR)."""
        img = np.asarray(image)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def _render_segmentation_overlay(self, image, mask, positions) -> np.ndarray:
        """Draw the segmentation mask (blue tint + outline) and the generated
        sample points (numbered orange markers) over the captured image."""
        img = self._to_bgr_uint8(image)
        if mask is not None:
            m = np.asarray(mask) > 0
            if m.shape[:2] == img.shape[:2] and m.any():
                tint = img.copy()
                tint[m] = (255, 144, 30)  # dodger-blue-ish (BGR)
                img = cv2.addWeighted(tint, 0.35, img, 0.65, 0)
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, (255, 200, 0), 2)
        for i, p in enumerate(positions):
            x = int(round(getattr(p, 'pixel_x', 0)))
            y = int(round(getattr(p, 'pixel_y', 0)))
            cv2.circle(img, (x, y), 6, (0, 165, 255), -1)  # orange fill (BGR)
            cv2.circle(img, (x, y), 6, (0, 0, 0), 1)
            cv2.putText(img, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    @staticmethod
    def _imwrite(path: str, img: np.ndarray) -> bool:
        """Write a PNG, tolerant of non-ASCII paths on Windows (cv2.imwrite is
        not). Returns True on success."""
        ok, buf = cv2.imencode('.png', img)
        if ok:
            buf.tofile(path)
        return bool(ok)

    def _save_sample_images(self, image, mask, positions,
                            rep: Optional[int] = None):
        """Save the captured image and a segmentation+points overlay for one
        sample (the captured FoV in Current FoV, each droplet in Robotic). The
        overlay includes EVERY generated point passed in, manually-added ones
        included. No-op when there's no save directory or no image."""
        if image is None:
            return
        save_dir = self._p.get('save_path') or ''
        if not save_dir:
            return
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError as e:
            self.error.emit(f"Cannot create save directory: {e}")
            return
        base = os.path.join(save_dir, self._sample_image_basename(rep))
        try:
            self._imwrite(f"{base}_captured.png", self._to_bgr_uint8(image))
            self._imwrite(f"{base}_segmented.png",
                          self._render_segmentation_overlay(image, mask, positions))
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Failed to save sample images: {e}")

    def _save(self, point, idx: int, rep: int, k: int, spectrum: np.ndarray,
              z_index: Optional[int] = None, z_um: Optional[float] = None):
        save_dir = self._p.get('save_path') or ''
        if not save_dir:
            logger.warning("No save directory set, spectrum not written")
            return
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError as e:
            self.error.emit(f"Cannot create save directory: {e}")
            return

        base = self._build_filename(point, idx, rep, k, z_index)
        path = os.path.join(save_dir, f"{base}.csv")

        x_nm = self._ctx.wavelengths_nm_provider()
        try:
            if x_nm is not None:
                n_px = min(len(x_nm), len(spectrum))
                data = np.column_stack([np.asarray(x_nm[:n_px]),
                                        spectrum[:n_px]])
                header = "wavelength_nm,intensity"
            else:
                data = np.column_stack([np.arange(len(spectrum)), spectrum])
                header = "pixel,intensity"
            np.savetxt(path, data, delimiter=",", header=header, comments="")
        except OSError as e:
            self.error.emit(f"Failed to save {path}: {e}")
            return

        self._save_metadata(path, point, idx, rep, k, z_index, z_um)
        self._n_saved += 1
        logger.info(f"Saved spectrum to {path}")

    def _save_metadata(self, spectra_path, point, idx, rep, k, z_index, z_um):
        pattern = str(self._p.get('sample_pattern') or '')
        meta = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "Current FoV",
            "droplet_position": self._cur_droplet_position or "",
            "sample_name": self._cur_sample_name or "",
            "droplet_number": (self._cur_droplet_number
                               if self._cur_droplet_number is not None else ""),
            "sampling_pattern": pattern,
            "point_position": self._point_position_token(point, idx, pattern),
            "point_id": getattr(point, 'id', idx + 1),
            "point_index": idx + 1,
            "sample_repeat": rep + 1,
            "rep_at_point": k + 1,
            "exposure_time_ms": self._p_float('acq_exposure', 0.0),
            "laser_power_mW": self._p_float('acq_power', 0.0),
            "excitation_wavelength_nm": self._ctx.excitation_nm,
            "pixel_x": getattr(point, 'pixel_x', ''),
            "pixel_y": getattr(point, 'pixel_y', ''),
            "stage_x_mm": getattr(point, 'stage_x', ''),
            "stage_y_mm": getattr(point, 'stage_y', ''),
            "obj_stage_x_mm": getattr(point, 'obj_stage_x', ''),
            "obj_stage_y_mm": getattr(point, 'obj_stage_y', ''),
        }
        if z_index is not None:
            meta["zstack_index"] = z_index + 1
            meta["nanodrive_z_um"] = z_um
        meta["spectra_file"] = os.path.basename(spectra_path)
        meta_path = os.path.splitext(spectra_path)[0] + "_metadata.txt"
        try:
            with open(meta_path, "w") as f:
                f.write("Raman Acquisition Metadata\n")
                f.write("==========================\n")
                for key, value in meta.items():
                    f.write(f"{key}: {value}\n")
        except OSError as e:
            logger.error(f"Failed to save metadata to {meta_path}: {e}")
