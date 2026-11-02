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
        f. Save ALL of that point's repetitions to ONE file, in the folder
           layout the NoodlePy spectrum viewer loads directly:
           <save path>/<YYYY_MM_DD_n>/spectra/<name>.txt plus
           <save path>/<YYYY_MM_DD_n>/metadata/<name>_metadata.txt.

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
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from raman_control.core.autofocus import AutofocusEngine
from raman_control.config.logging_config import get_logger

logger = get_logger('acquisition_thread')

# Physical marker diameter (micrometers) for the saved sample-point overlay.
# Kept in sync with live_view.MARKER_DIAMETER_UM so the saved image's markers
# are the same size as the ones drawn in the GUI.
MARKER_DIAMETER_UM = 60.0


def repeat_conditions(params: dict) -> List[Dict[str, float]]:
    """Spectrum acquisition settings for each protocol repeat (one dict per
    repeat: ``exposure_ms``, ``power_mw``, ``reps_per_point``).

    Repeat 1 always uses the main Spectrum Acquisition settings (``acq_exposure``
    / ``acq_power`` / ``reps_per_point``). With 'different settings for each
    repeat' on, repeats 2..N take theirs from ``repeat_conditions`` (missing
    entries fall back to the main settings); otherwise every repeat uses the
    main settings. Shared by the worker and the protocol preview."""
    n = max(1, int(params.get('repeat_each_sample') or 1))
    base = {
        'exposure_ms': float(params.get('acq_exposure') or 100.0),
        'power_mw': float(params.get('acq_power') or 50.0),
        'reps_per_point': max(1, int(params.get('reps_per_point') or 1)),
    }
    extra = (list(params.get('repeat_conditions') or [])
             if params.get('repeat_conditions_enabled') else [])
    conds = [dict(base)]
    for i in range(1, n):
        c = dict(base)
        if i - 1 < len(extra) and extra[i - 1]:
            src = extra[i - 1]
            c['exposure_ms'] = float(src.get('exposure_ms', c['exposure_ms']))
            c['power_mw'] = float(src.get('power_mw', c['power_mw']))
            c['reps_per_point'] = max(
                1, int(src.get('reps_per_point', c['reps_per_point'])))
        conds.append(c)
    return conds


def all_conditions_per_point(params: dict) -> bool:
    """True when every repeat condition is acquired at each point (after ONE
    round of autofocus / photobleaching) before moving to the next point,
    instead of repeating the whole protocol once per condition."""
    return (bool(params.get('repeat_all_conditions_per_point'))
            and max(1, int(params.get('repeat_each_sample') or 1)) > 1)


class PrusaAfPlan(NamedTuple):
    """Spectrometer Prusa autofocus for ONE sample point (see prusa_af_plan)."""
    reuse_sample_z: bool   # move to / bleach at the sample's stored focus Z
    before_bleach: bool    # scan at this point before any photobleach
    after_bleach: bool     # scan again after this point's photobleach
    store_after: bool      # the after-bleach result replaces the stored Z


def prusa_af_plan(params: dict, first_in_sample: bool) -> PrusaAfPlan:
    """Which spectrometer Prusa autofocus scans a sample point runs.

    * Every point, no photobleaching: one scan at every point.
    * Every point + photobleaching: only the sample's FIRST point scans before
      its bleach; its Z is stored and every other point moves to and bleaches
      at that same Z. The post-bleach re-focus still runs at EVERY point (the
      bleach can drift the focus), without changing the stored bleach Z.
    * Per sample: the first point scans (and re-scans after its bleach, which
      refreshes the stored Z); the other points reuse the Z with no scans.

    Shared by the worker and the protocol preview / time estimate so they
    always agree."""
    per_point = bool(params.get('autofocus_wasatch_prusa'))
    per_sample = bool(params.get('autofocus_wasatch_prusa_per_sample'))
    bleach = bool(params.get('wait_before_measure'))
    if per_point and bleach:
        return PrusaAfPlan(True, first_in_sample, True, False)
    if per_point:
        return PrusaAfPlan(False, True, False, False)
    if per_sample:
        return PrusaAfPlan(True, first_in_sample, bleach and first_in_sample,
                           True)
    return PrusaAfPlan(False, False, False, False)


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

    # Wide -> objective calibration offset (mm). ``offset`` is the fixed
    # triple. The Y component drifts linearly along the widefield X, so use
    # ``objective_offset_y(wide_x)`` instead of ``offset[1]`` wherever a
    # widefield point is converted to objective coordinates; it consults the
    # calibration curve when enabled and falls back to ``offset[1]`` otherwise.
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_y_provider: Optional[Callable[[float], float]] = None
    excitation_nm: float = 785.0

    # Spectrometer focus-score bands (min_nm, max_nm), from SpectrometerConfig:
    # the score is sum(signal band) / sum(reference band), minimised. The
    # coarse (Prusa) and fine (NanoDrive) scans have their own pair. None falls
    # back to AutofocusEngine's noodlepy defaults.
    af_prusa_signal_band_nm: Optional[Tuple[float, float]] = None
    af_prusa_reference_band_nm: Optional[Tuple[float, float]] = None
    af_nano_signal_band_nm: Optional[Tuple[float, float]] = None
    af_nano_reference_band_nm: Optional[Tuple[float, float]] = None
    # Focus metric: 'ratio' (bands above) or 'entropy' (Shannon entropy over
    # af_entropy_band_nm). A callable is accepted so the metric can follow the
    # spectrometer that is actually connected (SpectrumThread.focus_metric).
    af_metric: object = 'ratio'          # str or Callable[[], str]
    af_entropy_band_nm: Optional[Tuple[float, float]] = None
    # Extra metadata about the spectrometer in use (model, grating, CCD
    # temperature...). Called when each metadata file is written.
    spectrometer_metadata_provider: Optional[Callable[[], dict]] = None

    # Z scan ranges (lower, upper)
    wf_prusa_range: Tuple[float, float] = (0.0, 0.0)   # mm, widefield view
    obj_prusa_range: Tuple[float, float] = (0.0, 0.0)  # mm, objective view
    nano_range: Tuple[float, float] = (0.0, 0.0)       # um

    # Robotic Deposition slide-change hop (see RoboticDepositionWorker): Z
    # clearance above the objective "Highest" limit, and the stage Z max the
    # lifted height is clamped to.
    slide_change_z_clearance: float = 2.0  # mm
    stage_z_max: float = 200.0             # mm

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

    # Object-plane micrometers per widefield pixel, so the saved overlay's point
    # markers are the same physical size as the ones drawn in the GUI.
    pixel_size_um: float = 1.0

    # Sample/droplet identity for structured file names. droplet_position (well
    # name, e.g. 'B5') is used whenever known; sample_name/droplet_number come
    # from a loaded config. droplet_map (well -> info) is used by Robotic.
    droplet_map: Optional[dict] = None
    droplet_position: Optional[str] = None  # Current FoV: captured well name
    sample_name: Optional[str] = None
    droplet_number: object = None

    def objective_offset_y(self, wide_x: float) -> float:
        """Wide->objective Y offset (mm) at a WIDEFIELD X position.

        Uses the host's calibration-curve-aware provider when one was passed,
        otherwise the fixed Y offset.
        """
        if self.offset_y_provider is not None:
            return self.offset_y_provider(wide_x)
        return self.offset[1]

    def af_bands(self, stage: str) -> Tuple[Optional[Tuple[float, float]],
                                            Optional[Tuple[float, float]]]:
        """(signal_band, reference_band) for 'prusa' or 'nano'."""
        if stage == 'nano':
            return (self.af_nano_signal_band_nm, self.af_nano_reference_band_nm)
        return (self.af_prusa_signal_band_nm, self.af_prusa_reference_band_nm)

    def resolve_af_metric(self) -> str:
        """'ratio' or 'entropy' for the spectrometer in use."""
        m = self.af_metric
        if callable(m):
            try:
                m = m()
            except Exception:  # noqa: BLE001
                m = 'ratio'
        return 'entropy' if str(m or 'ratio').lower() == 'entropy' else 'ratio'

    def spectrometer_metadata(self) -> dict:
        if self.spectrometer_metadata_provider is None:
            return {}
        try:
            return dict(self.spectrometer_metadata_provider() or {})
        except Exception:  # noqa: BLE001
            return {}


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
    # 0-based index of the sample point now being measured, so the GUI can show
    # it in green on the captured-image preview, synced with the step cursor.
    measuring_point = pyqtSignal(int)
    # Spectrometer autofocus -> Live-tab focus-score plot (mirrors the
    # interactive autofocus). focus_started(stage) clears that stage's curve;
    # focus_point(stage, z, score) adds a scored point. stage = 'prusa'/'nano'.
    focus_started = pyqtSignal(str)
    focus_point = pyqtSignal(str, float, float)
    # Best-focus Z found by an autofocus scan, so the Live-tab autofocus panel's
    # focus-Z labels update during a run. kind = 'cam_prusa' / 'spec_prusa' (mm)
    # or 'spec_nano' (um); value is the best Z in that unit.
    focus_result = pyqtSignal(str, float)
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
        # Per-sample Prusa spectrometer autofocus: when that option is on, the
        # best Z found at a sample's first point is stored here and reused (as
        # the move target) for the rest of that sample's points. Reset to None
        # at the start of each sample (the FoV in Current FoV; each droplet,
        # per repeat, in Robotic Deposition).
        self._sample_prusa_z: Optional[float] = None
        # Most recent autofocus scan per focus kind ('cam_prusa', 'spec_prusa',
        # 'spec_nano'): the best Z found and the full (z, score) curve. Written
        # by _scan, read by _save_metadata so each saved spectrum records the
        # focus that was actually in effect when it was acquired.
        self._af_best: Dict[str, float] = {}
        self._af_curve: Dict[str, List[Tuple[float, float]]] = {}
        # Raw spectra behind each scored point of the latest spectrometer
        # autofocus scan per kind: [(z, (m, pixels) frames), ...]. Only written
        # to the metadata when "Save autofocus curves" is on.
        self._af_spectra: Dict[str, List[Tuple[float, np.ndarray]]] = {}
        # Frames of the most recent _spectrum_score call, picked up by _scan.
        self._last_af_frames: Optional[np.ndarray] = None
        # This run's experiment folder (<save path>/<YYYY_MM_DD_n>), created on
        # the first save so an aborted-before-anything run leaves no folder.
        self._exp_dir: Optional[str] = None
        self._steps_saved = False  # acquisition_steps.txt written for this run
        # Spectrum acquisition settings per protocol repeat ("condition"), and
        # the one currently in effect (see repeat_conditions / _cond).
        self._conditions = repeat_conditions(self._p)
        self._cond_idx = 0
        self._run_started = datetime.now()
        # (height, width) of the current sample's captured image, used to place
        # each point relative to the FoV centre in the file name.
        self._cur_image_shape: Optional[Tuple[int, int]] = None
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

    # --- repeat conditions (spectrum settings per repeat) -------------------
    def _cond(self) -> Dict[str, float]:
        """Spectrum acquisition settings currently in effect."""
        return self._conditions[min(self._cond_idx, len(self._conditions) - 1)]

    def _acq_exposure(self) -> float:
        return float(self._cond()['exposure_ms'])

    def _acq_power(self) -> float:
        return float(self._cond()['power_mw'])

    def _acq_reps(self) -> int:
        return int(self._cond()['reps_per_point'])

    def _all_conditions_per_point(self) -> bool:
        return all_conditions_per_point(self._p)

    def _outer_repeats(self) -> int:
        """How many times the run loops over all points/droplets: once per
        repeat, or once in total when every condition is done at each point."""
        if self._all_conditions_per_point():
            return 1
        return len(self._conditions)

    def _point_conditions(self, rep: int) -> List[int]:
        """Condition indices acquired at a point during outer repeat ``rep``."""
        if self._all_conditions_per_point():
            return list(range(len(self._conditions)))
        return [min(rep, len(self._conditions) - 1)]

    def _ensure_laser_on(self):
        """Turn the laser on before the first spectrometer measurement of the
        run. The first power-up can take a few seconds to actually fire, so wait
        once after enabling it (only the first time)."""
        if self._laser_warmed:
            return
        spec = self._ctx.spectrometer
        spec.set_laser_power_mw(self._acq_power())
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
        # Pause the free-running live spectrum for the whole run: every
        # spectrum the run needs (autofocus scores, saved data) is requested
        # explicitly, and live frames in between only add wait + laser time.
        spec = self._ctx.spectrometer
        held = spec is not None and hasattr(spec, 'hold_live')
        if held:
            spec.hold_live()
        try:
            self._run()
        except Exception as e:  # noqa: BLE001 - never let the thread crash silently
            logger.exception("Acquisition worker crashed")
            self.error.emit(f"Acquisition failed: {e}")
            self.finished_acq.emit(True, self._n_saved)
        finally:
            if held:
                spec.release_live()

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

        repeats = self._outer_repeats()
        do_wf_af = bool(self._p.get('autofocus_widefield_prusa'))

        # Spectra saved per point already accounts for the z-stack (reps at every
        # Z step), so the progress total and the bar stay correct in both modes.
        total = len(self._points) * self._run_spectra_per_point()
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
                        maximize=True, settle=ctx.af_settle_s,
                        focus_kind='cam_prusa')
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
        # Current FoV is a single sample (the captured field of view); re-focus
        # the per-sample Prusa autofocus at the first point of each repeat.
        for rep in range(repeats):
            self._sample_prusa_z = None
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
                                           done, total, label, move_note,
                                           first_in_sample=(idx == 0))

        self._finish()

    def _objective_xy(self, point) -> Tuple[float, float]:
        """Objective stage XY for a sample point: its precomputed obj coords,
        or its widefield stage coords + the wide->obj calibration offset (the
        path the runtime-generated robotic points take)."""
        ctx = self._ctx
        ox = getattr(point, 'obj_stage_x', None)
        oy = getattr(point, 'obj_stage_y', None)
        if ox is None or oy is None:
            base = ctx.position_provider()
            wide_x = getattr(point, 'stage_x', base[0])
            ox = wide_x + ctx.offset[0]
            oy = getattr(point, 'stage_y', base[1]) + ctx.objective_offset_y(wide_x)
        return ox, oy

    def _measure_point(self, point, idx, rep, base_obj_z, done, total,
                       label, move_note="", first_in_sample=True) -> int:
        """Measure one sample point and return the updated ``done`` counter.

        Moves the objective over the point, (optionally) Wasatch-autofocuses,
        then acquires + saves ``reps_per_point`` spectra (or a Z-stack). Shared
        by Current FoV (`_run`) and Robotic Deposition (per generated point).
        Objective XY come from the point's precomputed obj coords, or fall back
        to its widefield stage coords + the wide->obj calibration offset (this is
        the path the runtime-generated robotic points take).

        ``first_in_sample`` marks the first point of the current sample; with
        per-sample Prusa autofocus it gates the single autofocus scan and the
        focus Z it finds is reused (as the move Z) for the rest of the sample."""
        ctx = self._ctx
        # Spectrum settings for this point's autofocus restore / photobleach:
        # this repeat's condition (the first one when every condition is done
        # at each point).
        conds = self._point_conditions(rep)
        self._cond_idx = conds[0]
        z_stack = bool(self._p.get('z_stack'))
        # Which Prusa scans this point runs, and whether it reuses the
        # sample's stored focus Z (see prusa_af_plan).
        plan = prusa_af_plan(self._p, first_in_sample)
        do_wasatch_nano = bool(self._p.get('autofocus_wasatch_nanodrive'))

        # Tell the GUI which point we're about to measure (green highlight).
        self.measuring_point.emit(idx)
        self.progress.emit(done, total, f"Moving to {label}{move_note}")

        ox, oy = self._objective_xy(point)
        # Per-sample Prusa autofocus (and every-point AF with photobleaching)
        # reuses the Z found at the sample's first point for the remaining
        # points (instead of the nominal objective Z).
        z_target = base_obj_z
        if plan.reuse_sample_z and self._sample_prusa_z is not None:
            z_target = self._sample_prusa_z
        if not ctx.motion.move_xyz_blocking(ox, oy, z_target, timeout=60.0):
            self.error.emit(f"Move to {label} failed, skipping")
            return done

        # The laser must be on for any spectrometer signal (autofocus or
        # measurement). Turn it on once, before the first point's spectrometer
        # work, with a one-time warm-up wait.
        if not self._abort:
            self._ensure_laser_on()

        # Coarse Wasatch autofocus (Prusa) before any bleach: every point in
        # per-point mode, or only the sample's first point when the Z is reused
        # (its result is stored in _sample_prusa_z for the rest of the sample).
        if plan.before_bleach and not self._abort:
            self.status.emit(f"Spectrometer autofocus (Prusa) at {label}…")
            best = self._wasatch_prusa_af()
            if plan.reuse_sample_z and best is not None:
                self._sample_prusa_z = best

        # Optional photobleaching: after the Prusa autofocus, before the
        # NanoDrive autofocus / z-stack (so the fine focus + measurement happen
        # on the already-bleached spot).
        bleached = bool(self._p.get('wait_before_measure'))
        if bleached and not self._abort:
            self._wait_with_laser(self._p_float('wait_time_s', 0.0))

        # Re-run the coarse Prusa autofocus AFTER photobleaching: the bleach can
        # heat / shift the sample and drift the focus, so re-establish it before
        # the fine focus + measurement. Every point in per-point mode (the
        # stored bleach Z is kept); per-sample only its first point, where it
        # refreshes the reused focus Z.
        if bleached and plan.after_bleach and not self._abort:
            self.status.emit(
                f"Spectrometer autofocus (Prusa) after photobleach at {label}…")
            best = self._wasatch_prusa_af()
            if plan.store_after and best is not None:
                self._sample_prusa_z = best

        # Per-point fine focus: NanoDrive AF, OR a Z-stack if enabled. The
        # z-stack acquires reps_per_point spectra at EACH Z step and reports its
        # own per-step progress, so it owns the rest of this point's work.
        if z_stack and not self._abort:
            return self._z_stack(point, idx, conds, done, total, label,
                                 move_note)
        if do_wasatch_nano and not self._abort:
            self.status.emit(f"Spectrometer autofocus (NanoDrive) at {label}…")
            self._wasatch_nano_af()

        # Acquire at this point (no more autofocus / photobleaching): this
        # repeat's condition, or every condition in turn when all conditions
        # are completed per point.
        for c in conds:
            if self._abort:
                break
            self._cond_idx = c
            done = self._acquire_condition(point, idx, c, done, total, label,
                                           self._condition_note(c))
        return done

    def _condition_note(self, c: int) -> str:
        """' condition k/N' when every condition is done at each point (the
        step-preview cursor matches it to that condition's line), else ''."""
        if not self._all_conditions_per_point():
            return ""
        return f" condition {c + 1}/{len(self._conditions)}"

    def _acquire_condition(self, point, idx, c, done, total, label,
                           cnote="") -> int:
        """Acquire the current condition's repetitions at this point, then
        write them all into ONE file (one intensity column per repetition),
        tagged with the condition's repeat number ``c``."""
        reps = self._acq_reps()
        spectra = []
        for k in range(reps):
            if self._abort:
                break
            self._wait_if_paused()
            if self._abort:
                break
            self.progress.emit(done, total,
                               f"Acquiring {label}{cnote}  {k + 1}/{reps}")
            spectrum = self._acquire_one()
            if spectrum is not None:
                spectra.append(spectrum)
            done += 1
            self.progress.emit(min(done, total), total,
                               f"Acquired {label}{cnote}  {k + 1}/{reps}")
        # Written even when the run was aborted part-way, so the repetitions
        # already acquired at this point are not lost.
        self._save_group(point, idx, c, spectra)
        if spectra:
            self.progress.emit(min(done, total), total,
                               f"Saved {label}{cnote}  ({len(spectra)} spectra)")
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
    def _scan(self, positions, move_fn, score_fn, maximize, settle, stage=None,
              focus_kind=None):
        """Generic focus scan: move to each Z, score, return to the best.

        If ``stage`` is given ('prusa'/'nano'), each scored point is streamed to
        the Live-tab focus-score plot (clearing it first), exactly like the
        interactive autofocus does. If ``focus_kind`` is given ('cam_prusa' /
        'spec_prusa' / 'spec_nano'), the best Z is reported to the Live-tab
        autofocus panel's focus-Z labels.

        Returns the best Z (and leaves the hardware there), or None if the
        scan was aborted before any point was scored.

        When ``focus_kind`` is given the scored (z, score) pairs and the best Z
        are also kept in ``self._af_curve`` / ``self._af_best`` so the metadata
        written for each spectrum records the focus in effect for it."""
        if stage:
            self.focus_started.emit(stage)
        best_z = None
        best_s = None
        curve = []
        spectra = []
        for z in positions:
            if self._abort:
                break
            self._wait_if_paused()
            if self._abort:
                break
            move_fn(float(z))
            time.sleep(settle)
            self._last_af_frames = None
            s = score_fn()
            if s is None:
                continue
            curve.append((float(z), float(s)))
            # Spectrometer scores leave their raw frames behind; camera
            # scores leave None, so a camera scan records no spectra.
            if self._last_af_frames is not None:
                spectra.append((float(z), self._last_af_frames))
            if stage:
                self.focus_point.emit(stage, float(z), float(s))
            if best_s is None or (s > best_s if maximize else s < best_s):
                best_s, best_z = s, float(z)
        if focus_kind:
            # Kept even for an aborted scan: a partial curve still shows what
            # was measured. Replaces the previous scan of the same kind.
            self._af_curve[focus_kind] = curve
            self._af_spectra[focus_kind] = spectra
            if best_z is not None:
                self._af_best[focus_kind] = best_z
        if best_z is not None and not self._abort:
            move_fn(best_z)
            if focus_kind:
                self.focus_result.emit(focus_kind, best_z)
        return best_z

    def _camera_score(self) -> float:
        frame = self._ctx.frame_provider()
        if frame is None:
            return 0.0
        return float(AutofocusEngine.calculate_tenengrad(frame))

    def _spectrum_score(self, reps: int, expo_ms: float,
                        stage: str = 'prusa') -> float:
        """Median of `reps` FRESH spectra → Raman ratio score (lower is
        better). The median rejects cosmic rays; using independent exposures
        (acquire_frames) makes the rejection meaningful.

        ``stage`` ('prusa'/'nano') picks that scan's configured score bands."""
        spec = self._ctx.spectrometer
        frames = spec.acquire_frames(max(1, reps))
        if frames is None or not len(frames):
            return float('inf')
        self._last_af_frames = np.atleast_2d(np.asarray(frames, dtype=float))
        median = np.median(frames, axis=0)
        wl_nm = self._ctx.wavelengths_nm_provider()
        try:
            signal_band, reference_band = self._ctx.af_bands(stage)
            return float(AutofocusEngine.spectrum_focus_score(
                self._ctx.resolve_af_metric(), wl_nm, median,
                signal_band=signal_band,
                reference_band=reference_band,
                entropy_band=self._ctx.af_entropy_band_nm))
        except Exception:  # noqa: BLE001
            return float('inf')

    def _reset_nanodrive_to_initial(self):
        """Park the NanoDrive at its configured initial position.

        Called before every coarse Prusa scan. Otherwise the NanoDrive is still
        at the fine focus found for the PREVIOUS point, which biases this
        point's coarse scan and leaves the following fine scan off-center in
        its range.
        """
        nano = self._ctx.nanodrive
        if not (nano and getattr(nano, 'is_connected', False)):
            return
        z0 = float(nano.initial_position)
        if nano.move_to(z=z0):
            logger.info(f"NanoDrive reset to initial position {z0:.1f} um "
                        f"before the Prusa autofocus scan")
        else:
            logger.warning(f"NanoDrive reset to {z0:.1f} um failed; "
                           f"the Prusa scan continues from its current Z")

    def _wasatch_prusa_af(self) -> Optional[float]:
        """Coarse spectrometer autofocus over the objective Prusa Z range.
        Returns the best-focus Z (mm), leaving the stage there, or None if the
        scan couldn't run / was aborted."""
        ctx = self._ctx
        lo, hi = ctx.obj_prusa_range
        if hi <= lo:
            return None
        self._reset_nanodrive_to_initial()
        steps = max(2, self._p_int('spec_af_coarse', 20))
        expo = self._p_float('spec_af_prusa_exposure', 100.0)
        power = self._p_float('spec_af_power',
                              self._acq_power())
        reps = self._p_int('spec_af_reps', 3)
        restore = self._apply_spec_settings(expo, power)
        try:
            return self._scan(
                np.linspace(lo, hi, steps),
                move_fn=lambda z: ctx.motion.move_z_blocking(z, timeout=30.0),
                score_fn=lambda: self._spectrum_score(reps, expo, 'prusa'),
                maximize=False, settle=0.05, stage='prusa',
                focus_kind='spec_prusa')
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
        expo = self._p_float('spec_af_nano_exposure', 100.0)
        power = self._p_float('spec_af_power',
                              self._acq_power())
        reps = self._p_int('spec_af_reps', 3)
        restore = self._apply_spec_settings(expo, power)
        try:
            self._scan(np.linspace(lo, hi, steps),
                       move_fn=lambda z: (nano.move_to(z=float(z)) or True),
                       score_fn=lambda: self._spectrum_score(reps, expo, 'nano'),
                       maximize=False, settle=0.03, stage='nano',
                       focus_kind='spec_nano')
        finally:
            restore()

    def _apply_spec_settings(self, expo_ms: float, power_mw: float):
        """Set spectrometer exposure/power; return a callable restoring the
        acquisition exposure/power afterwards."""
        spec = self._ctx.spectrometer
        prev_expo = self._acq_exposure()
        prev_power = self._acq_power()
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
        self.status.emit(f"Photobleaching {int(seconds)} s with laser on")
        spec = self._ctx.spectrometer
        spec.set_laser_power_mw(self._acq_power())
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
        expo = self._acq_exposure()
        power = self._acq_power()
        spec.set_integration_time(expo)
        spec.set_laser_power_mw(power)
        return spec.acquire_average(1)

    def _z_steps(self) -> int:
        """Number of NanoDrive Z positions in a z-stack (matches the NanoDrive
        Steps setting; at least 2)."""
        return max(2, self._p_int('spec_af_fine', 10))

    def _per_point_spectra(self) -> int:
        """Spectra saved per point for the CURRENT condition: its reps per
        point, multiplied by the z-stack step count when z-stack is enabled
        (reps at every Z step)."""
        return self._acq_reps() * (self._z_steps() if self._p.get('z_stack')
                                   else 1)

    def _run_spectra_per_point(self) -> int:
        """Spectra saved per point over the WHOLE run (every repeat condition,
        each with its own reps per point; z-stack folded in)."""
        zf = self._z_steps() if self._p.get('z_stack') else 1
        return sum(int(c['reps_per_point']) for c in self._conditions) * zf

    def _z_stack(self, point, idx: int, conds: List[int], done: int,
                 total: int, label: str, note: str = "") -> int:
        """Acquire a NanoDrive Z-stack at the current point: at EACH Z step,
        acquire every condition in ``conds`` (this repeat's, or all of them when
        every condition is done per point), each condition's repetitions written
        into ONE file for that z-step (tagged with its z-step index and the
        condition's repeat number, one intensity column per repetition).
        Reports per-step progress and returns the updated ``done`` counter."""
        # Clear the NanoDrive focus-score curve so a stale autofocus curve from a
        # previous point/run isn't left on the plot during the z-stack.
        self.focus_started.emit('nano')
        nano = self._ctx.nanodrive
        if not (nano and getattr(nano, 'is_connected', False)):
            # No piezo -> acquire each condition at the current Z (single step).
            for c in conds:
                if self._abort:
                    break
                self._cond_idx = c
                done = self._acquire_condition(point, idx, c, done, total,
                                               label, self._condition_note(c))
            return done
        lo, hi = self._ctx.nano_range
        steps = self._z_steps()
        for zi, z in enumerate(np.linspace(lo, hi, steps)):
            if self._abort:
                return done
            self._wait_if_paused()
            nano.move_to(z=float(z))
            time.sleep(0.03)
            for c in conds:
                if self._abort:
                    return done
                self._cond_idx = c
                cnote = self._condition_note(c)
                reps = self._acq_reps()
                spectra = []
                for k in range(reps):
                    if self._abort:
                        break
                    self._wait_if_paused()
                    if self._abort:
                        break
                    self.progress.emit(
                        done, total,
                        f"Z-stack {label}: z-step {zi + 1}/{steps},{cnote} "
                        f"spectrum {k + 1}/{reps}{note}")
                    spectrum = self._acquire_one()
                    if spectrum is not None:
                        spectra.append(spectrum)
                    done += 1
                # One file per z-step (and condition), holding its repetitions.
                # Written even on abort so the completed repetitions survive.
                self._save_group(point, idx, c, spectra,
                                 z_index=zi, z_um=float(z))
            if self._abort:
                return done
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
        """Point-position field for the file name. ALWAYS starts with the
        sequential point number (``point_<n>``, n = idx+1) for every pattern and
        mode, then appends the pattern-specific spatial detail for grid/ring."""
        base = f"point_{idx + 1}"
        order = getattr(point, 'coordinates_order', None) or {}
        if pattern == 'grid' and 'row' in order:
            return f"{base}_row_{order['row']}_col_{order['col']}"
        if pattern == 'ring' and 'ring' in order:
            return f"{base}_ring_{order['ring']}_ray_{order['line']}"
        return base

    # Point-position fields per sampling pattern (the part of the file name that
    # varies by pattern). Every pattern leads with point_<n>. Kept in sync with
    # _point_position_token.
    _POINTPOS_FORMAT = {
        'grid': 'point_<n>_row_<r>_col_<c>',
        'ring': 'point_<n>_ring_<r>_ray_<a>',
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
        ``<sample_name>``. The point-position always leads with point_<n>, with
        the pattern-specific spatial detail appended (grid -> _row_/_col_, ring
        -> _ring_/_ray_). ``spectra_<n>`` is how many spectra the file holds:
        every repetition taken at that position goes into a single file. Kept in
        sync with _build_filename."""
        pos = AcquisitionWorker._POINTPOS_FORMAT.get(pattern, 'point_<n>')
        ptype = pattern or '<type>'
        parts = []
        if prefix and prefix.strip():
            parts.append(f"prefix_{AcquisitionWorker._token(prefix)}")
        if not current_fov:
            # Mirrors _build_filename: the value is the config sample name, with
            # a leading 'patient' dropped so 'patient_406' reads patient_406
            # rather than patient_patient406.
            sp = AcquisitionWorker._token(sample_prefix) if sample_prefix else ''
            if sp.lower().startswith('patient'):
                sp = sp[len('patient'):]
            sample = f"patient_{sp}<name>" if sp else "patient_<sample_name>"
            parts.append(f"{sample}_sample_<#>_position_<well>")
        parts.append(f"pattern_{ptype}_{pos}_rep_<n>_spectra_<n>_"
                     "date_<YYYYMMDD>_x<um>_y<um>")
        return "_".join(parts)

    def _build_filename(self, point, idx, rep, n_spectra, z_index=None) -> str:
        """Structured spectrum file name, every field as ``name_value`` joined
        by '_':

            <sample_name>_droplet_<#>_position_<well>_pattern_<type>_<pointpos>_
            spectra_<n>_date_<YYYYMMDD>_repeat_<n>

        The sample name comes straight from the config (it already carries its
        prefix, e.g. 'patient_304'). The droplet position (well, e.g. B5) is
        included whenever known; sample name + droplet number only when a config
        is loaded. The point-position adapts to the pattern (row_/col_,
        ring_/ray_, or point_). A z-stack index is appended.

        ``spectra_<n>`` is the NUMBER of spectra the file holds (all the
        repetitions taken at this position), so a single-repetition run still
        produces ``spectra_1`` exactly as before.
        """
        pattern = str(self._p.get('sample_pattern') or 'pattern')
        prefix = str(self._p.get('filename_prefix') or '').strip()
        parts = []
        if prefix:
            # Optional user prefix. Paired as a key/value like every other
            # field so it doesn't shift the viewer's token pairing.
            parts.append(f"prefix_{self._token(prefix)}")
        # Current FoV passes no sample/droplet/position (they stay None), so the
        # name starts at the pattern there; Robotic fills them per droplet.
        # 'patient' and 'sample' are the two tokens the viewer groups droplets
        # by, so the config sample name and droplet number map onto those.
        if self._cur_sample_name:
            # The config sample name usually already carries its own prefix
            # ('patient_406'), and the token key is 'patient', so drop a
            # leading 'patient' to get patient_406 rather than
            # patient_patient406.
            pid = self._token(self._cur_sample_name)
            if pid.lower().startswith('patient') and pid[len('patient'):]:
                pid = pid[len('patient'):]
            parts.append(f"patient_{pid}")
        if self._cur_droplet_number not in (None, ''):
            parts.append(f"sample_{self._token(self._cur_droplet_number)}")
        elif self._cur_droplet_position:
            parts.append(f"sample_{self._token(self._cur_droplet_position)}")
        if self._cur_droplet_position:
            parts.append(f"position_{self._token(self._cur_droplet_position)}")
        parts.append(f"pattern_{self._token(pattern)}")
        parts.append(self._point_position_token(point, idx, pattern))
        parts.append(f"rep_{rep + 1}")
        parts.append(f"spectra_{n_spectra}")
        parts.append(f"date_{datetime.now().strftime('%Y%m%d')}")
        if z_index is not None:
            parts.append(f"zstep_{z_index + 1}")
        # Trailing x/y (micrometers, relative to the sample's FoV centre). The
        # viewer pulls these out with an 'x<num>_y<num>' regex and uses them to
        # place the point on its hyperspectral map, so they must stay adjacent,
        # in this order, and last.
        px_um, py_um = self._point_xy_um(point)
        parts.append(f"x{px_um:.2f}_y{py_um:.2f}")
        return "_".join(parts)

    @staticmethod
    def _token(value) -> str:
        """Make a value safe to use as a file-name token.

        The viewer splits the name on '_' and reads the tokens as alternating
        key/value pairs, so a VALUE containing '_' (or a space) would shift
        every pair after it. Keep alphanumerics only. A value that would look
        like a coordinate ('x1', 'y-2', 'z3') is prefixed, because the viewer
        strips those tokens out before pairing.
        """
        out = ''.join(c for c in str(value) if c.isalnum()) or 'na'
        if re.match(r'^[xyz][-+]?\d', out):
            out = f"v{out}"
        return out

    def _point_xy_um(self, point) -> Tuple[float, float]:
        """The point's offset from its sample's FoV centre, in micrometers.

        The viewer reads x/y straight out of the file name to position each
        spectrum on the hyperspectral map (and derives
        ``distance_from_center_mm = hypot(x, y) * 0.001``, i.e. it wants um).
        Returns (0, 0) when the point carries no pixel coordinates or the
        sample's image size is unknown — the spectra still load, they just
        stack at the map origin.
        """
        px = getattr(point, 'pixel_x', None)
        py = getattr(point, 'pixel_y', None)
        shape = self._cur_image_shape
        if px is None or py is None or not shape:
            return (0.0, 0.0)
        h, w = shape[0], shape[1]
        um_per_px = float(self._ctx.pixel_size_um or 1.0)
        return ((float(px) - w / 2.0) * um_per_px,
                (float(py) - h / 2.0) * um_per_px)

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
        """Draw the captured image with the same overlays the GUI shows: a
        translucent blue mask fill (no contour) and the generated sample points
        as filled orange markers with the point number centred inside, matching
        live_view._update_capture_preview."""
        img = self._to_bgr_uint8(image)
        if mask is not None:
            m = np.asarray(mask) > 0
            if m.shape[:2] == img.shape[:2] and m.any():
                # Dodger blue (RGB 30,144,255 -> BGR), alpha 100/255 (as the GUI).
                tint = img.copy()
                tint[m] = (255, 144, 30)
                img = cv2.addWeighted(tint, 100 / 255.0, img, 1 - 100 / 255.0, 0)
        # Marker radius from the same physical diameter the GUI uses.
        px_um = self._ctx.pixel_size_um or 1.0
        r = max(3, int(round((MARKER_DIAMETER_UM / 2.0) / px_um)))
        orange = (18, 156, 243)  # #f39c12 in BGR
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, p in enumerate(positions):
            x = int(round(getattr(p, 'pixel_x', 0)))
            y = int(round(getattr(p, 'pixel_y', 0)))
            cv2.circle(img, (x, y), r, orange, -1, lineType=cv2.LINE_AA)
            # Point number, black + centred inside the marker, sized to fit.
            text = str(i + 1)
            thick = max(1, int(round(r / 9)))
            (bw, bh), _ = cv2.getTextSize(text, font, 1.0, thick)
            scale = min(1.3 * r / max(bw, 1), 1.3 * r / max(bh, 1))
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            cv2.putText(img, text, (x - tw // 2, y + th // 2), font, scale,
                        (0, 0, 0), thick, cv2.LINE_AA)
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
        # Remember the FoV size so each point's x/y offset from the centre can
        # go into its file name (see _point_xy_um).
        self._cur_image_shape = tuple(image.shape[:2])
        # Images live at the top level of the experiment folder, next to the
        # spectra/ and metadata/ subfolders. The viewer ignores them, which is
        # exactly how its own reference datasets are laid out.
        spectra_dir, _ = self._experiment_dirs()
        if not spectra_dir:
            return
        base = os.path.join(self._exp_dir, self._sample_image_basename(rep))
        try:
            self._imwrite(f"{base}_captured.png", self._to_bgr_uint8(image))
            self._imwrite(f"{base}_segmented.png",
                          self._render_segmentation_overlay(image, mask, positions))
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Failed to save sample images: {e}")

    # --- experiment folder (spectrum-viewer layout) -------------------------
    def _experiment_dirs(self) -> Tuple[Optional[str], Optional[str]]:
        """Create (once per run) and return this run's ``spectra`` and
        ``metadata`` directories.

        The NoodlePy spectrum viewer is pointed at an EXPERIMENT FOLDER and
        expects exactly two subfolders inside it::

            <save path>/<YYYY_MM_DD_n>/spectra/<name>.txt
            <save path>/<YYYY_MM_DD_n>/metadata/<name>_metadata.txt

        It also uses the experiment folder's own name as the ``date`` of every
        spectrum in it, so each run gets its own dated folder; the ``_n``
        suffix counts up so a second run on the same day is a separate
        experiment instead of merging into the first.

        Returns ``(None, None)`` when no save path is set or the folders
        cannot be created.
        """
        save_dir = self._p.get('save_path') or ''
        if not save_dir:
            return None, None
        if self._exp_dir is None:
            day = datetime.now().strftime('%Y_%m_%d')
            n = 1
            while os.path.isdir(os.path.join(save_dir, f"{day}_{n}")):
                n += 1
            self._exp_dir = os.path.join(save_dir, f"{day}_{n}")
        spectra_dir = os.path.join(self._exp_dir, 'spectra')
        metadata_dir = os.path.join(self._exp_dir, 'metadata')
        try:
            os.makedirs(spectra_dir, exist_ok=True)
            os.makedirs(metadata_dir, exist_ok=True)
        except OSError as e:
            self.error.emit(f"Cannot create save directory: {e}")
            return None, None
        if not self._steps_saved:
            self._steps_saved = True
            self._save_steps_preview()
        return spectra_dir, metadata_dir

    # File name of the saved step preview, at the experiment folder's top level
    # (next to the sample images and the spectra/ + metadata/ subfolders).
    STEPS_FILENAME = 'acquisition_steps.txt'

    def _save_steps_preview(self):
        """Write the protocol panel's acquisition-steps preview (as shown when
        Start was pressed) into the experiment folder. Skipped when the run
        was started without one."""
        text = self._p.get('steps_preview') or ''
        if not text or not self._exp_dir:
            return
        path = os.path.join(self._exp_dir, self.STEPS_FILENAME)
        header = (f"Acquisition steps preview "
                  f"(run started {self._run_started:%Y-%m-%d %H:%M:%S})\n\n")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(header + text.rstrip() + "\n")
            logger.info(f"Saved acquisition steps preview to {path}")
        except OSError as e:
            self.error.emit(f"Failed to save acquisition steps: {e}")

    def _save_group(self, point, idx: int, rep: int,
                    spectra: Sequence[np.ndarray],
                    z_index: Optional[int] = None,
                    z_um: Optional[float] = None):
        """Write every repetition acquired at ONE position into a single file,
        in the layout the spectrum viewer loads directly.

        ``spectra/<name>.txt`` holds two comma-separated columns and NO header
        row: wavelength (nm) and intensity. The repetitions are stacked as
        consecutive BLOCKS that each repeat the same wavelength column — the
        viewer detects the repeats, splits on them and takes the element-wise
        median, so this is its native way of carrying several measurements of
        one position in one file. ``metadata/<name>_metadata.txt`` is written
        alongside it.

        Does nothing when ``spectra`` is empty.
        """
        if not spectra:
            return
        spectra_dir, metadata_dir = self._experiment_dirs()
        if not spectra_dir:
            logger.warning("No save directory set, spectra not written")
            return

        n = len(spectra)
        base = self._build_filename(point, idx, rep, n, z_index)
        path = os.path.join(spectra_dir, f"{base}.txt")

        x_nm = self._ctx.wavelengths_nm_provider()
        try:
            # Trim to the shortest spectrum (and the X axis) so every block is
            # the same length — the viewer stacks them and would fail on ragged
            # blocks.
            n_px = min(len(s) for s in spectra)
            if x_nm is not None:
                n_px = min(n_px, len(x_nm))
                axis = np.asarray(x_nm[:n_px], dtype=float)
            else:
                # No wavelength axis available: the viewer still needs two
                # numeric columns, so fall back to the pixel index.
                axis = np.arange(n_px, dtype=float)
            blocks = [np.column_stack([axis, np.asarray(s[:n_px], dtype=float)])
                      for s in spectra]
            # 4 decimals on the X axis keeps every wavelength distinct within a
            # block (the viewer finds the block boundaries by looking for the
            # repeated X value, so duplicates inside one block would break it).
            np.savetxt(path, np.vstack(blocks), delimiter=", ",
                       fmt=("%.4f", "%.4f"))
        except OSError as e:
            self.error.emit(f"Failed to save {path}: {e}")
            return

        meta_path = os.path.join(metadata_dir, f"{base}_metadata.txt")
        self._save_metadata(meta_path, path, point, idx, rep, n, z_index, z_um)
        self._n_saved += n
        logger.info(f"Saved {n} spectra to {path}")

    def _save_metadata(self, meta_path, spectra_path, point, idx, rep,
                       n_spectra, z_index, z_um):
        """Write the ``<name>_metadata.txt`` sidecar for one spectra file.

        The viewer parses this as plain ``Key: value`` lines, lower-casing each
        key and turning spaces and '(' into '_' (so "Laser power (mW)" becomes
        ``laser_power_mw``). The first block below uses the exact key spellings
        it looks for; everything after is extra detail that simply shows up as
        additional filter / colour-by fields.
        """
        pattern = str(self._p.get('sample_pattern') or '')
        meta = {
            # --- keys the viewer itself consumes ---
            "Number of repetitions": n_spectra,
            "Integration time (ms)": self._acq_exposure(),
            "Laser power (mW)": self._acq_power(),
            "Prusa position (mm)": f"{self._current_prusa_z():.4f}",
            "Nanodrive position (um)": f"{self._current_nano_z(z_um):.4f}",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # --- extra detail (filterable in the viewer) ---
            "Mode": self._p.get('acquisition_mode') or "Current FoV",
            "Droplet position": self._cur_droplet_position or "",
            "Sample name": self._cur_sample_name or "",
            "Droplet number": (self._cur_droplet_number
                               if self._cur_droplet_number is not None else ""),
            "Sampling pattern": pattern,
            "Point position": self._point_position_token(point, idx, pattern),
            "Point id": getattr(point, 'id', idx + 1),
            "Point index": idx + 1,
            "Sample repeat": rep + 1,
            # Every repetition taken at this position lives in this one file,
            # stacked as consecutive blocks.
            "Reps in file": n_spectra,
            "Excitation wavelength (nm)": self._ctx.excitation_nm,
            # Best-focus Z of each autofocus scan in effect for this spectrum.
            # Blank when that scan was not enabled / found nothing.
            "Widefield AF best z (mm)": self._af_best_str('cam_prusa'),
            "Prusa AF best z (mm)": self._af_best_str('spec_prusa'),
            "Nanodrive AF best z (um)": self._af_best_str('spec_nano'),
            "Pixel x": getattr(point, 'pixel_x', ''),
            "Pixel y": getattr(point, 'pixel_y', ''),
            "Stage x (mm)": getattr(point, 'stage_x', ''),
            "Stage y (mm)": getattr(point, 'stage_y', ''),
            "Obj stage x (mm)": getattr(point, 'obj_stage_x', ''),
            "Obj stage y (mm)": getattr(point, 'obj_stage_y', ''),
        }
        if z_index is not None:
            meta["Zstack index"] = z_index + 1
        # Which spectrometer took this (Wasatch / Kymera), its grating, CCD
        # temperature... so files from the two instruments are distinguishable.
        meta.update(self._ctx.spectrometer_metadata())
        meta["Autofocus metric"] = self._ctx.resolve_af_metric()
        meta["Spectra file"] = os.path.basename(spectra_path)
        try:
            with open(meta_path, "w") as f:
                for key, value in meta.items():
                    f.write(f"{key}: {value}\n")
                if self._p.get('save_autofocus_curves'):
                    self._write_af_curves(f)
        except OSError as e:
            logger.error(f"Failed to save metadata to {meta_path}: {e}")

    def _current_prusa_z(self) -> float:
        """Prusa Z (mm) this spectrum was measured at: the coarse autofocus
        result when one ran, otherwise the live stage position."""
        z = self._af_best.get('spec_prusa')
        if z is not None:
            return float(z)
        try:
            return float(self._ctx.position_provider()[2])
        except Exception:  # noqa: BLE001
            return 0.0

    def _current_nano_z(self, z_um: Optional[float]) -> float:
        """NanoDrive Z (um) this spectrum was measured at: the z-stack step
        when in a stack, else the device position, else the fine autofocus
        result."""
        if z_um is not None:
            return float(z_um)
        nano = self._ctx.nanodrive
        if nano is not None and getattr(nano, 'is_connected', False):
            try:
                return float(nano.position)
            except Exception:  # noqa: BLE001
                pass
        return float(self._af_best.get('spec_nano') or 0.0)

    def _af_best_str(self, kind: str) -> str:
        """Best-focus Z for ``kind`` formatted for the metadata, or '' if the
        scan never ran (option off, no valid range, nothing scored)."""
        z = self._af_best.get(kind)
        return "" if z is None else f"{z:.4f}"

    # Curve blocks appended to the metadata: (focus kind, block title, Z unit).
    _AF_CURVE_BLOCKS = (
        ('spec_prusa', "Prusa Autofocus Curve", "z_mm"),
        ('spec_nano', "NanoDrive Autofocus Curve", "z_um"),
    )

    def _write_af_curves(self, f) -> None:
        """Append the full autofocus curves to an open metadata file.

        One block per stage, each a two-column table of Z position and focus
        score in scan order. A block whose scan did not run says so instead of
        showing an empty table.

        Every line here is free of ':' and '=', so the spectrum viewer's
        ``Key: value`` metadata parser skips the whole appendix instead of
        turning it into junk fields. (The underline is '-' for exactly that
        reason: a row of '=' would parse as an empty-named key.)
        """
        for kind, title, z_col in self._AF_CURVE_BLOCKS:
            curve = self._af_curve.get(kind) or []
            f.write(f"\n{title}\n")
            f.write(f"{'-' * len(title)}\n")
            if not curve:
                f.write("(not run)\n")
                continue
            f.write(f"{z_col},focus_score\n")
            for z, score in curve:
                f.write(f"{z:.4f},{score:.6g}\n")
        for kind, title, z_col in self._AF_CURVE_BLOCKS:
            self._write_af_spectra(f, title.replace("Curve", "Spectra"),
                                   self._af_spectra.get(kind) or [])

    def _write_af_spectra(self, f, title, points) -> None:
        """Append every raw spectrum an autofocus scan collected.

        A wide table: one row per pixel, first column the wavelength axis, then
        one column per exposure named ``z<Z>_rep<k>`` (Z in the same unit as
        the matching curve block), in scan order. Same ':'/'=' free rule as the
        curve blocks so the viewer's metadata parser ignores it.
        """
        f.write(f"\n{title}\n")
        f.write(f"{'-' * len(title)}\n")
        if not points:
            f.write("(not run)\n")
            return
        names, cols = [], []
        for z, frames in points:
            for k, frame in enumerate(frames):
                names.append(f"z{z:.4f}_rep{k + 1}")
                cols.append(np.asarray(frame, dtype=float))
        # Trim to the shortest exposure (and the X axis) so the table is
        # rectangular, as _save_group does for the spectra file.
        n_px = min(len(c) for c in cols)
        x_nm = self._ctx.wavelengths_nm_provider()
        if x_nm is not None:
            n_px = min(n_px, len(x_nm))
            axis = np.asarray(x_nm[:n_px], dtype=float)
        else:
            axis = np.arange(n_px, dtype=float)
        table = np.column_stack([axis] + [c[:n_px] for c in cols])
        f.write("wavelength_nm," + ",".join(names) + "\n")
        np.savetxt(f, table, delimiter=",", fmt="%.4f")
