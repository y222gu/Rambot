"""
Acquisition worker for the Robotic Deposition mode.

Subclasses :class:`AcquisitionWorker` and reuses ALL of its per-point measurement
machinery (`_measure_point`, the Wasatch autofocus scans, LED/laser handling,
saving). The only thing it adds is the per-droplet outer loop:

    for each selected droplet (well):
        1. move the stage over the droplet (widefield view)
        2. illuminate + widefield-autofocus the camera (optional)
        3. capture a widefield frame and segment it
        4. generate the chosen sample pattern on that droplet at run time
        5. switch to the objective and measure every generated point

Points-per-droplet are NOT known until each droplet is imaged + segmented, so the
progress total is an UPPER BOUND (exact for line/random, max for grid/ring) and
the actual generated count per droplet is logged.

Everything hardware-facing goes through the same thread-safe accessors the base
worker uses (``move_xyz_blocking``, the camera ``frame_provider``); segmentation
runs on this worker thread via the (already loaded) ``UNetSegmenter`` handed in on
the context. Nothing here touches the GUI directly — view switches and LED changes
are emitted as signals handled by ``MainWindow``.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import pyqtSignal

from raman_control.config.logging_config import get_logger
from raman_control.gui.threads.acquisition_thread import (
    AcquisitionWorker, AcquisitionContext)
from raman_control.segmentation import pattern_layouts
from raman_control.segmentation.pattern_generator import (
    PatternGenerator, CalibrationParams)

logger = get_logger('robotic_thread')


class RoboticDepositionWorker(AcquisitionWorker):
    """Run a Robotic Deposition acquisition over selected droplets (wells)."""

    # (droplet_index, droplet_name, n_points_generated)
    droplet_started = pyqtSignal(int, str, int)
    # Per-droplet captured image + segmentation + generated points, so the GUI
    # can show what the robot is actually measuring on each droplet. Payload is a
    # dict: {'image', 'mask', 'positions', 'capture_xyz'} (mask/positions may be
    # empty when nothing was segmented).
    droplet_segmented = pyqtSignal(object)

    def __init__(self, ctx: AcquisitionContext,
                 wells: Sequence[Tuple[str, float, float]],
                 params: dict, parent=None):
        # No pre-generated points — they're produced per droplet at run time.
        super().__init__(ctx, [], params, parent)
        self._wells = list(wells)  # [(name, logical_x, logical_y), …]

    # --- upper-bound point count per droplet (for the progress total) -------
    @staticmethod
    def _max_points(ps: Optional[dict]) -> int:
        """Upper bound on points/droplet from the pattern params (exact for
        line/random; max for grid/ring since only points inside each droplet's
        shape are kept)."""
        if not ps:
            return 1
        t = ps.get('type', 'ring')
        p = ps.get(t, {})
        if t == 'grid':
            return max(1, int(p.get('rows', 1)) * int(p.get('cols', 1)))
        if t == 'ring':
            return max(1, int(p.get('rays', 1)) * int(p.get('rings', 1)))
        return max(1, int(p.get('points', 1)))  # line / random

    # --- main loop ----------------------------------------------------------
    def _run(self):
        ctx = self._ctx
        motion = ctx.motion

        if not self._wells:
            self.error.emit("No droplets selected to acquire")
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
        if ctx.segmenter is None:
            self.error.emit("Segmenter not ready")
            self.finished_acq.emit(True, 0)
            return

        repeats = max(1, self._p_int('repeat_each_sample', 1))
        do_wf_af = bool(self._p.get('autofocus_widefield_prusa'))
        lo, hi = ctx.wf_prusa_range

        # Progress total = UPPER BOUND of spectra to save (see _max_points).
        # _per_point_spectra() already folds in the z-stack (reps at every Z).
        est_pts = self._max_points(ctx.pattern_settings)
        total = repeats * len(self._wells) * est_pts * self._per_point_spectra()
        done = 0

        # Start the widefield AF / first droplet move near the current Z.
        wf_z = ctx.position_provider()[2]

        for rep in range(repeats):
            for di, (name, wx, wy) in enumerate(self._wells):
                if self._abort:
                    return self._finish()
                self._wait_if_paused()
                if self._abort:
                    return self._finish()

                rep_note = (f" (repeat {rep + 1}/{repeats})"
                            if repeats > 1 else "")
                self.progress.emit(done, total,
                                   f"Droplet {name}: moving{rep_note}")

                # Identify this droplet for the structured spectrum file names:
                # its well position always, plus sample name + number from the
                # loaded config (if any).
                self._cur_droplet_position = name
                info = (ctx.droplet_map or {}).get(name) or {}
                self._cur_sample_name = info.get('sample_name')
                self._cur_droplet_number = info.get('droplet_number')

                # 1. Move to the droplet in WIDEFIELD view (logical == physical
                #    in widefield; the objective offset is applied per point).
                #    The droplet-scoped status lands the cursor on this droplet's
                #    "Switch to widefield view" step.
                self.status.emit(f"Droplet {name}: switching to widefield view")
                self.view_changed.emit('widefield')
                if not motion.move_xyz_blocking(wx, wy, wf_z, timeout=60.0):
                    self.error.emit(f"Move to droplet {name} failed, skipping")
                    continue

                # 2-3. Illuminate, (optionally) widefield-autofocus, capture.
                #      LED stays on through AF + capture, then off for the Raman.
                frame = None
                self._led_on()
                try:
                    if do_wf_af and hi > lo and not self._abort:
                        self.status.emit(
                            f"Droplet {name}: widefield autofocus…")
                        best = self._scan(
                            np.linspace(lo, hi, max(2, self._p_int('cam_af_steps', 20))),
                            move_fn=lambda z: motion.move_z_blocking(z, timeout=30.0),
                            score_fn=self._camera_score,
                            maximize=True, settle=ctx.af_settle_s)
                        if best is not None:
                            wf_z = best
                    # Extra stabilization at the best focus so the stage is fully
                    # still before the segmentation capture (slightly longer than
                    # the per-step settle, so the captured frame is the sharpest).
                    self._interruptible_sleep(ctx.af_capture_stabilization_s)
                    self.status.emit("Capture image")
                    frame = ctx.frame_provider()
                finally:
                    self._led_off()

                if self._abort:
                    return self._finish()
                if frame is None:
                    self.error.emit(f"Droplet {name}: no camera frame, skipping")
                    continue

                # 4. Segment + generate the pattern on this droplet (also saves
                #    the captured + segmented images and updates the GUI).
                positions = self._segment_and_generate(frame, (wx, wy, wf_z), rep)
                n_pts = len(positions)
                self.droplet_started.emit(di, name, n_pts)
                logger.info(f"Droplet {name}: generated {n_pts} sample points")
                if n_pts == 0:
                    self.status.emit(
                        f"Droplet {name}: nothing segmented, skipping")
                    continue

                # 5. Measure every generated point through the objective.
                base_obj_z = wf_z + ctx.offset[2]
                self.status.emit(f"Droplet {name}: switching to objective view")
                self.view_changed.emit('objective')
                for pi, point in enumerate(positions):
                    if self._abort:
                        return self._finish()
                    self._wait_if_paused()
                    if self._abort:
                        return self._finish()
                    label = f"{name} · pt {pi + 1}/{n_pts}"
                    done = self._measure_point(point, pi, rep, base_obj_z,
                                               done, total, label, rep_note)

        self._finish()

    def _emit_droplet_view(self, frame, mask, positions, capture_xyz):
        """Push the per-droplet captured image + segmentation + points to the GUI
        (handled on the GUI thread by MainWindow). Arrays are copied because the
        worker keeps acquiring frames after this returns."""
        self.droplet_segmented.emit({
            'image': None if frame is None else np.array(frame, copy=True),
            'mask': None if mask is None else np.array(mask, copy=True),
            'positions': list(positions),
            'capture_xyz': tuple(capture_xyz),
        })

    # --- per-droplet segmentation + pattern generation ----------------------
    def _segment_and_generate(self, frame, capture_xyz, rep):
        """Segment ``frame`` and generate the snapshot pattern on it. Returns a
        list of SamplePosition (widefield stage coords); [] if nothing usable.

        On every path it both pushes the droplet to the GUI (droplet_segmented)
        AND saves the captured image + segmentation/points overlay, so the live
        view and the saved record always reflect the droplet just imaged."""
        ctx = self._ctx
        ps = ctx.pattern_settings or {}
        mask = None
        positions = []
        self.status.emit("Segment droplet")
        try:
            seg = ctx.segmenter.segment(frame)
        except Exception as e:  # noqa: BLE001
            logger.exception("Segmentation failed")
            self.error.emit(f"Segmentation failed: {e}")
            seg = None
        if seg is not None and np.any(seg):
            mask = seg
            self.status.emit("Generate sample points")
            ptype = ps.get('type', 'ring')
            params = dict(ps.get(ptype, {}))
            try:
                points, orders = pattern_layouts.generate(ptype, mask, params)
            except Exception as e:  # noqa: BLE001
                logger.exception("Pattern generation failed")
                self.error.emit(f"Pattern generation failed: {e}")
                points, orders = [], []
            if points:
                positions = self._build_positions(points, orders, mask, capture_xyz)

        self._emit_droplet_view(frame, mask, positions, capture_xyz)
        self._save_sample_images(frame, mask, positions, rep)
        return positions

    def _build_positions(self, points, orders, mask, capture_xyz):
        """Turn the generated pixel points into SamplePositions carrying both
        widefield and objective stage coordinates."""
        ctx = self._ctx
        ps = ctx.pattern_settings or {}
        h, w = mask.shape[:2]
        cal = CalibrationParams(
            pixel_size_x_um=ps.get('pixel_size_x_um') or 1.0,
            pixel_size_y_um=ps.get('pixel_size_y_um') or 1.0,
            image_width=w,
            image_height=h,
            capture_stage_x=capture_xyz[0],
            capture_stage_y=capture_xyz[1],
            capture_stage_z=capture_xyz[2],
            invert_y=bool(ps.get('invert_y', True)),
        )
        gen = PatternGenerator(cal)
        positions = gen.from_centroids(
            points, coordinates_orders=orders if orders else None)

        # Fill in each point's OBJECTIVE coords (widefield + wide->obj offset),
        # mirroring MainWindow._on_positions_generated for the Current-FoV path.
        # Without this, SamplePosition.obj_stage_* keep their 0.0 default and
        # _measure_point would drive the stage to the origin (X0 Y0) instead of
        # the droplet (the bug that sent it to the corner).
        ox_off, oy_off, oz_off = ctx.offset
        for p in positions:
            p.obj_stage_x = p.stage_x + ox_off
            p.obj_stage_y = p.stage_y + oy_off
            p.obj_stage_z = p.stage_z + oz_off
        if positions:
            p0 = positions[0]
            logger.info(
                f"First point obj target: X={p0.obj_stage_x:.2f} "
                f"Y={p0.obj_stage_y:.2f} (widefield X={p0.stage_x:.2f} "
                f"Y={p0.stage_y:.2f}, offset X={ox_off:.2f} Y={oy_off:.2f})")
        return positions
