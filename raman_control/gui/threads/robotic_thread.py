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
from raman_control.core.slide_layout import plan_slide_hop, slide_of_well
from raman_control.segmentation import pattern_layouts
from raman_control.segmentation.pattern_generator import (
    PatternGenerator, CalibrationParams)

logger = get_logger('robotic_thread')


class RoboticDepositionWorker(AcquisitionWorker):
    """Run a Robotic Deposition acquisition over selected droplets (wells)."""

    # (droplet_index, droplet_name, n_points_generated)
    droplet_started = pyqtSignal(int, str, int)
    # Well name of the droplet the robot is now moving to / processing, emitted at
    # the start of each droplet so the sample map can highlight it as current.
    droplet_active = pyqtSignal(str)
    # Well name of the droplet being CAPTURED during the upfront capture phase.
    # The map highlights it (cyan edge) WITHOUT a check mark — it has not been
    # measured yet, so it must not look 'done'.
    droplet_capturing = pyqtSignal(str)
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
        self._obj_slide = None  # slide the objective is over (upfront flow)

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

        if bool(self._p.get('capture_upfront')):
            self._run_upfront()
        else:
            self._run_interleaved()

    def _progress_total(self) -> int:
        """Upper-bound spectra count for the progress bar (see _max_points);
        _run_spectra_per_point() folds in every repeat condition + z-stack."""
        est_pts = self._max_points(self._ctx.pattern_settings)
        return len(self._wells) * est_pts * self._run_spectra_per_point()

    def _set_droplet_identity(self, name: str, capturing: bool = False):
        """Tag the structured spectrum file names with this droplet (its well
        position always, plus sample name + number from the loaded config) and
        highlight it on the sample map.

        ``capturing`` picks the highlight: during the upfront capture phase the
        droplet is only being imaged (not measured), so it gets a cyan edge with
        NO check mark (droplet_capturing). Otherwise it is being measured, so it
        gets the cyan + check 'current' highlight (droplet_active)."""
        self._cur_droplet_position = name
        info = (self._ctx.droplet_map or {}).get(name) or {}
        self._cur_sample_name = info.get('sample_name')
        self._cur_droplet_number = info.get('droplet_number')
        if capturing:
            self.droplet_capturing.emit(name)
        else:
            self.droplet_active.emit(name)

    # --- interleaved (default) ----------------------------------------------
    def _run_interleaved(self):
        """Default flow: per droplet, image + segment + generate in widefield,
        then switch to the objective and measure that droplet's points before
        moving on to the next."""
        ctx = self._ctx
        # Once per repeat, or once in total when every repeat condition is
        # completed at each point.
        repeats = self._outer_repeats()
        total = self._progress_total()
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
                self._set_droplet_identity(name)

                positions, wf_z, capture_xyz, _frame, _mask = (
                    self._acquire_droplet_widefield(name, wx, wy, wf_z, rep))
                n_pts = len(positions)
                self.droplet_started.emit(di, name, n_pts)
                logger.info(f"Droplet {name}: generated {n_pts} sample points")
                if self._abort:
                    return self._finish()
                if n_pts == 0:
                    self.status.emit(
                        f"Droplet {name}: nothing segmented, skipping")
                    continue

                # Measure every generated point through the objective.
                base_obj_z = wf_z + ctx.offset[2]
                self.status.emit(f"Droplet {name}: switching to objective view")
                self.view_changed.emit('objective')
                # Per-sample Prusa autofocus re-focuses at each droplet.
                self._sample_prusa_z = None
                for pi, point in enumerate(positions):
                    if self._abort:
                        return self._finish()
                    self._wait_if_paused()
                    if self._abort:
                        return self._finish()
                    label = f"{name} · pt {pi + 1}/{n_pts}"
                    done = self._measure_point(point, pi, rep, base_obj_z,
                                               done, total, label, rep_note,
                                               first_in_sample=(pi == 0))

        self._finish()

    # --- upfront capture, then measure --------------------------------------
    def _run_upfront(self):
        """Capture/segment/generate EVERY droplet first (widefield, LED on per
        droplet), then switch to the objective once and measure every droplet.
        The captures happen a single time; only the measurement phase repeats
        per protocol repeat, so the objective view is never left mid-run."""
        ctx = self._ctx
        # Once per repeat, or once in total when every repeat condition is
        # completed at each point.
        repeats = self._outer_repeats()
        total = self._progress_total()
        done = 0
        wf_z = ctx.position_provider()[2]

        # The slide-change hop in phase 2 needs the objective "Highest" limit;
        # check it before spending time on the captures.
        obj_lo, obj_hi = ctx.obj_prusa_range
        if not obj_hi > obj_lo:
            self.error.emit(
                "Objective Z limits not set (Highest must be greater than "
                "Lowest); needed to lift over the barriers between slides")
            return self._finish()

        # --- Phase 1: capture every droplet (widefield) ---------------------
        captured = []  # one record per droplet that yielded points
        for di, (name, wx, wy) in enumerate(self._wells):
            if self._abort:
                return self._finish()
            self._wait_if_paused()
            if self._abort:
                return self._finish()

            self.progress.emit(done, total,
                               f"Droplet {name}: capturing (widefield)")
            # Capture phase: highlight without a check (not measured yet).
            self._set_droplet_identity(name, capturing=True)
            positions, wf_z, capture_xyz, frame, mask = (
                self._acquire_droplet_widefield(name, wx, wy, wf_z, rep=0))
            n_pts = len(positions)
            self.droplet_started.emit(di, name, n_pts)
            logger.info(f"Droplet {name}: generated {n_pts} sample points")
            if self._abort:
                return self._finish()
            if n_pts == 0:
                self.status.emit(
                    f"Droplet {name}: nothing segmented, skipping")
                continue
            captured.append({
                'name': name,
                'positions': positions,
                'base_obj_z': wf_z + ctx.offset[2],
                'sample_name': self._cur_sample_name,
                'droplet_number': self._cur_droplet_number,
                # Kept so phase 2 can re-show this droplet's image when it is
                # measured (the live view otherwise shows the last captured one).
                'image': None if frame is None else np.array(frame, copy=True),
                'mask': None if mask is None else np.array(mask, copy=True),
                'capture_xyz': tuple(capture_xyz),
            })

        if self._abort:
            return self._finish()
        if not captured:
            self.status.emit("Nothing segmented on any droplet")
            return self._finish()

        # --- Phase 2: measure every droplet (objective), repeated -----------
        # The LED is already off (turned off per droplet in phase 1). Switch to
        # the objective a single time and stay there for all the measurements.
        self.status.emit("Switching to objective view")
        self.view_changed.emit('objective')
        # Slide the objective is currently over. Unknown (None) right after the
        # switch from widefield, so the first droplet always hops.
        self._obj_slide = None

        for rep in range(repeats):
            rep_note = (f" (repeat {rep + 1}/{repeats})"
                        if repeats > 1 else "")
            for rec in captured:
                if self._abort:
                    return self._finish()
                self._wait_if_paused()
                if self._abort:
                    return self._finish()
                name = rec['name']
                # Restore this droplet's file-name identity + map highlight.
                self._cur_droplet_position = name
                self._cur_sample_name = rec['sample_name']
                self._cur_droplet_number = rec['droplet_number']
                self.droplet_active.emit(name)
                positions = rec['positions']
                base_obj_z = rec['base_obj_z']
                n_pts = len(positions)
                # Re-show this droplet's captured image + segmentation + points
                # so the measuring-point highlight lands on the right image (the
                # live view otherwise still shows the last droplet from phase 1).
                self._emit_droplet_view(rec['image'], rec['mask'], positions,
                                        rec['capture_xyz'])
                # Lift over the barrier if this droplet is on another slide.
                if not self._slide_hop_if_needed(name, positions[0]):
                    if self._abort:
                        return self._finish()
                    self.error.emit(
                        f"Droplet {name}: slide-change move failed, skipping")
                    continue
                # Per-sample Prusa autofocus re-focuses at each droplet.
                self._sample_prusa_z = None
                for pi, point in enumerate(positions):
                    if self._abort:
                        return self._finish()
                    self._wait_if_paused()
                    if self._abort:
                        return self._finish()
                    label = f"{name} · pt {pi + 1}/{n_pts}"
                    done = self._measure_point(point, pi, rep, base_obj_z,
                                               done, total, label, rep_note,
                                               first_in_sample=(pi == 0))

        self._finish()

    # --- slide-change hop (objective view, upfront flow) --------------------
    def _slide_hop_if_needed(self, name, first_point) -> bool:
        """Carry the objective over the physical barrier between slides.

        If droplet ``name`` is on a different slide than the objective is over
        (or that is unknown, e.g. right after switching from widefield), then:
        raise Z to the objective "Highest" limit + clearance at the current XY,
        travel XY to the droplet's first point at that height, and lower Z back
        to "Highest". ``_measure_point`` then descends to the focus Z as usual.

        Returns False if a hop move failed (the objective position is then
        unknown, so the next droplet hops again)."""
        ctx = self._ctx
        motion = ctx.motion
        slide = slide_of_well(name)
        if slide is not None and slide == self._obj_slide:
            return True

        obj_hi = ctx.obj_prusa_range[1]
        ox, oy = self._objective_xy(first_point)
        moves = plan_slide_hop(ox, oy, ctx.position_provider()[2], obj_hi,
                               ctx.slide_change_z_clearance, ctx.stage_z_max)
        lift_z = moves[0][2]
        src = "?" if self._obj_slide is None else str(self._obj_slide)
        dst = "?" if slide is None else str(slide)
        self.status.emit(
            f"Slide {src} -> {dst}: lifting objective over barrier")
        logger.info(
            f"Slide hop {src} -> {dst} (droplet {name}): Z up to {lift_z:.3f}, "
            f"XY to ({ox:.3f}, {oy:.3f}), Z down to {obj_hi:.3f}")

        self._obj_slide = None
        for x, y, z in moves:
            if self._abort:
                return False
            ok = (motion.move_z_blocking(z, timeout=30.0) if x is None
                  else motion.move_xyz_blocking(x, y, z, timeout=60.0))
            if not ok:
                return False
        self._obj_slide = slide
        return True

    # --- per-droplet widefield phase ----------------------------------------
    def _acquire_droplet_widefield(self, name, wx, wy, wf_z, rep):
        """Move to a droplet in widefield, illuminate, (optionally) widefield-
        autofocus, capture, (optionally) recenter + re-capture, then segment and
        generate the pattern. Saves the final captured + segmented images and
        pushes them to the GUI. The LED stays on from the first capture through
        any recenter re-capture, then off before segmentation.

        Returns ``(positions, wf_z, capture_xyz, frame, mask)`` — ``positions``
        is [] if the move or capture failed or nothing was segmented; ``wf_z`` is
        the (possibly autofocus-updated) widefield Z; ``capture_xyz`` is the stage
        position the final frame was captured at (the pattern's image-center
        reference); ``frame``/``mask`` are the final captured image + segmentation
        (so the upfront flow can re-show each droplet when it later measures it)."""
        ctx = self._ctx
        motion = ctx.motion
        do_wf_af = bool(self._p.get('autofocus_widefield_prusa'))
        do_center = bool(self._p.get('center_sample'))
        lo, hi = ctx.wf_prusa_range

        # Move to the droplet in WIDEFIELD view (logical == physical in
        # widefield; the objective offset is applied per point). The droplet-
        # scoped status lands the cursor on this droplet's widefield step.
        self.status.emit(f"Droplet {name}: switching to widefield view")
        self.view_changed.emit('widefield')
        if not motion.move_xyz_blocking(wx, wy, wf_z, timeout=60.0):
            self.error.emit(f"Move to droplet {name} failed, skipping")
            return [], wf_z, (wx, wy, wf_z), None, None

        cap_x, cap_y = wx, wy
        frame = None
        # Illuminate, (optionally) widefield-autofocus, capture. The LED stays on
        # through AF, capture, and any recenter re-capture, then off for the Raman.
        self._led_on()
        try:
            if do_wf_af and hi > lo and not self._abort:
                self.status.emit(f"Droplet {name}: widefield autofocus…")
                best = self._scan(
                    np.linspace(lo, hi, max(2, self._p_int('cam_af_steps', 20))),
                    move_fn=lambda z: motion.move_z_blocking(z, timeout=30.0),
                    score_fn=self._camera_score,
                    maximize=True, settle=ctx.af_settle_s,
                    focus_kind='cam_prusa')
                if best is not None:
                    wf_z = best
            # Extra stabilization at the best focus so the stage is fully still
            # before the segmentation capture (slightly longer than the per-step
            # settle, so the captured frame is the sharpest).
            self._interruptible_sleep(ctx.af_capture_stabilization_s)
            self.status.emit("Capture image")
            frame = ctx.frame_provider()

            # Optional: recenter the stage on the droplet centroid and re-capture
            # so the generated pattern is centered in the field of view.
            if do_center and frame is not None and not self._abort:
                cap_x, cap_y, frame = self._center_on_droplet(
                    name, frame, cap_x, cap_y, wf_z)
        finally:
            self._led_off()

        if self._abort:
            return [], wf_z, (cap_x, cap_y, wf_z), None, None
        if frame is None:
            self.error.emit(f"Droplet {name}: no camera frame, skipping")
            return [], wf_z, (cap_x, cap_y, wf_z), None, None

        # Segment + generate the FINAL frame (also saves the captured + segmented
        # images and pushes them to the GUI).
        positions, mask = self._segment_and_generate(
            frame, (cap_x, cap_y, wf_z), rep)
        return positions, wf_z, (cap_x, cap_y, wf_z), frame, mask

    # Centering passes: always run at least this many recentering moves, and up
    # to the max if the droplet mask still touches an image edge afterwards.
    _CENTER_MIN_PASSES = 2
    _CENTER_MAX_PASSES = 3

    @staticmethod
    def _mask_touches_edge(mask) -> bool:
        """True if any mask pixel lies on the image border (the droplet runs off
        the edge of the field of view, so it isn't fully centred yet)."""
        return bool(mask[0, :].any() or mask[-1, :].any()
                    or mask[:, 0].any() or mask[:, -1].any())

    def _center_on_droplet(self, name, frame, cap_x, cap_y, wf_z):
        """Iteratively recenter the stage on the droplet centroid so the
        generated pattern is centred in the field of view.

        Runs at least ``_CENTER_MIN_PASSES`` recentering moves, and a third (up
        to ``_CENTER_MAX_PASSES``) if the mask still touches an image edge after
        the minimum passes. Each pass: segment the current frame, move so the
        centroid is at the FoV center, wait the capture-stabilization time so the
        stage is fully still (otherwise the re-captured frame is a blurred
        mid-move one), then re-capture. The intermediate segmentations are only
        for centering and are NOT saved (only the final centered images are,
        by the caller).

        Returns the final ``(cap_x, cap_y, frame)`` — the inputs unchanged if
        nothing could be segmented / moved on the first pass."""
        ctx = self._ctx
        ps = ctx.pattern_settings or {}
        cur_frame = frame
        moves = 0
        while moves < self._CENTER_MAX_PASSES:
            if self._abort:
                break
            try:
                seg = ctx.segmenter.segment(cur_frame)
            except Exception as e:  # noqa: BLE001
                logger.exception("Centering segmentation failed")
                self.error.emit(f"Centering segmentation failed: {e}")
                break
            if seg is None or not np.any(seg):
                self.status.emit(f"Droplet {name}: nothing to center on")
                break
            # After the minimum passes, stop once the droplet no longer runs off
            # an image edge (it is well centred); otherwise keep going.
            if moves >= self._CENTER_MIN_PASSES and not self._mask_touches_edge(seg):
                break

            ys, xs = np.nonzero(seg)
            centroid_x = float(xs.mean())
            centroid_y = float(ys.mean())
            h, w = seg.shape[:2]
            # The new stage center is the physical location currently under the
            # centroid pixel; moving there brings the centroid to the FoV center.
            cal = CalibrationParams(
                pixel_size_x_um=ps.get('pixel_size_x_um') or 1.0,
                pixel_size_y_um=ps.get('pixel_size_y_um') or 1.0,
                image_width=w,
                image_height=h,
                capture_stage_x=cap_x,
                capture_stage_y=cap_y,
                capture_stage_z=wf_z,
                invert_y=bool(ps.get('invert_y', True)),
            )
            new_x, new_y, _ = PatternGenerator(cal).pixel_to_stage(
                centroid_x, centroid_y)
            moves += 1
            self.status.emit(
                f"Droplet {name}: centering on droplet "
                f"(pass {moves}/{self._CENTER_MAX_PASSES})")
            if not ctx.motion.move_xyz_blocking(new_x, new_y, wf_z, timeout=60.0):
                self.error.emit(f"Droplet {name}: centering move failed")
                break
            cap_x, cap_y = new_x, new_y
            # Settle so the re-captured frame is sharp (not a blurred mid-move
            # one), on EVERY pass.
            self._interruptible_sleep(ctx.af_capture_stabilization_s)
            if self._abort:
                break
            self.status.emit("Capture image")
            new_frame = ctx.frame_provider()
            if new_frame is None:
                break
            cur_frame = new_frame
        return cap_x, cap_y, cur_frame

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
        """Segment ``frame`` and generate the snapshot pattern on it. Returns
        ``(positions, mask)`` — ``positions`` a list of SamplePosition (widefield
        stage coords), [] if nothing usable; ``mask`` the segmentation mask (or
        None).

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
        return positions, mask

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
        # The Y offset is position dependent (it drifts linearly along the
        # widefield X), so it is evaluated per point at that point's X.
        ox_off, _fixed_oy_off, oz_off = ctx.offset
        for p in positions:
            p.obj_stage_x = p.stage_x + ox_off
            p.obj_stage_y = p.stage_y + ctx.objective_offset_y(p.stage_x)
            p.obj_stage_z = p.stage_z + oz_off
        if positions:
            p0 = positions[0]
            logger.info(
                f"First point obj target: X={p0.obj_stage_x:.2f} "
                f"Y={p0.obj_stage_y:.2f} (widefield X={p0.stage_x:.2f} "
                f"Y={p0.stage_y:.2f}, offset X={ox_off:.2f} "
                f"Y={ctx.objective_offset_y(p0.stage_x):.2f})")
        return positions
