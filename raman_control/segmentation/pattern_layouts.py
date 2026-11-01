"""
Standalone sample-pattern layout algorithms (grid / ring / line / random).

These are the pure geometric algorithms that turn a binary droplet **mask** plus
a set of numeric **params** into a list of pixel sample points (and their ordering
metadata). They were extracted verbatim from ``LiveViewWidget`` so they can be
called BOTH from the interactive Live tab (GUI thread) AND from the Robotic
Deposition acquisition worker (off the GUI thread), where each droplet is
segmented at run time and its points generated without any Qt widgets.

Every generator takes ``(mask, params)`` and returns ``(points, orders)`` where
  * ``points``  -> list of ``(pixel_x, pixel_y)`` floats in image coordinates,
  * ``orders``  -> parallel list of small dicts describing each point's place in
                   the pattern (row/col, ring/line, …) — empty for 'random'.

``params`` keys (buffers/intervals are already in PIXELS — convert µm→px at the
call site using the widefield pixel-size calibration):
  grid   -> rows, cols, buffer_px
  ring   -> rays, rings, interval_px, buffer_px
  line   -> points, buffer_px
  random -> points
"""

from typing import Dict, List, Tuple

import numpy as np
import cv2

Points = List[Tuple[float, float]]
Orders = List[dict]


def apply_buffer(mask, buffer_px) -> np.ndarray:
    """Shrink or expand a binary mask by ``buffer_px`` pixels.

    buffer_px > 0  -> erode: keep points at least that far INSIDE the edge
    buffer_px < 0  -> dilate: allow points that far OUTSIDE the mask
    buffer_px == 0 -> unchanged
    """
    b = int(buffer_px)
    if b == 0:
        return (mask > 0).astype(np.uint8)
    ksize = 2 * abs(b) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m = (mask > 0).astype(np.uint8)
    return cv2.erode(m, kernel) if b > 0 else cv2.dilate(m, kernel)


def generate_grid(mask, params: Dict) -> Tuple[Points, Orders]:
    """Grid pattern: meshgrid over the buffered mask bbox. By default ALL grid
    points are kept; set ``remove_outside`` to drop the ones outside the mask."""
    h, w = mask.shape[:2]
    rows_n = int(params['rows'])
    cols_n = int(params['cols'])
    remove_outside = bool(params.get('remove_outside', False))

    bmask = apply_buffer(mask, params.get('buffer_px', 0))
    bys, bxs = np.where(bmask == 1)
    if bxs.size == 0:
        return [], []

    x_lin = np.linspace(bxs.min(), bxs.max(), cols_n)
    y_lin = np.linspace(bys.min(), bys.max(), rows_n)
    xx, yy = np.meshgrid(x_lin, y_lin)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    ix = np.clip(np.round(xx_flat).astype(int), 0, w - 1)
    iy = np.clip(np.round(yy_flat).astype(int), 0, h - 1)
    # "Remove outside" keeps points that are inside BOTH the placement margin
    # (bmask) AND the RAW segmented mask the user sees. For a positive buffer
    # (eroded) that intersection is the eroded mask, preserving the margin; for a
    # negative buffer (dilated) it collapses to the raw mask, so points that sit
    # outside the droplet are dropped instead of being kept by the dilation.
    keep = (bmask > 0) & (mask > 0)
    inside = (keep[iy, ix] if remove_outside
              else np.ones(xx_flat.shape, dtype=bool))

    col_indices = np.tile(np.arange(cols_n), rows_n)
    row_indices = np.repeat(np.arange(rows_n), cols_n)

    points = [(float(xx_flat[i]), float(yy_flat[i]))
              for i in range(len(xx_flat)) if inside[i]]
    orders = [
        {'row': int(row_indices[i]) + 1, 'col': int(col_indices[i]) + 1}
        for i in range(len(xx_flat)) if inside[i]
    ]
    return points, orders


def generate_ring(mask, params: Dict) -> Tuple[Points, Orders]:
    """Ring/radial pattern matching old noodlepy edgedetectorSAM.py."""
    h, w = mask.shape[:2]
    num_rays = int(params['rays'])
    num_rings = int(params['rings'])
    interval = float(params['interval_px'])
    buffer_px = float(params.get('buffer_px', 0))
    remove_outside = bool(params.get('remove_outside', False))
    # Outside-the-mask removal is judged against the RAW segmented mask the user
    # sees (line below), independent of the placement buffer above.
    raw = mask > 0

    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return [], []
    center_x = float(xs.mean())
    center_y = float(ys.mean())

    all_x, all_y = [], []
    ring_numbers, line_numbers = [], []

    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

    for ray_idx, angle in enumerate(angles):
        dx = np.cos(angle)
        dy = np.sin(angle)

        max_dist = max(w, h)
        edge_dist = 0
        for dist in range(int(max_dist)):
            x_i = int(center_x + dist * dx)
            y_i = int(center_y + dist * dy)
            if x_i < 0 or x_i >= w or y_i < 0 or y_i >= h or mask[y_i, x_i] == 0:
                edge_dist = dist - 1
                break

        if edge_dist <= 0:
            continue

        start_dist = edge_dist - buffer_px
        min_dist_from_center = 10

        for ring_idx in range(num_rings):
            d = start_dist - ring_idx * interval
            if d > min_dist_from_center:
                px = center_x + d * dx
                py = center_y + d * dy
                ipx, ipy = int(round(px)), int(round(py))
                if (0 <= ipx < w and 0 <= ipy < h
                        and (not remove_outside or raw[ipy, ipx])):
                    all_x.append(px)
                    all_y.append(py)
                    ring_numbers.append(ring_idx + 1)
                    line_numbers.append(ray_idx + 1)

    if not all_x:
        return [], []

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    ring_numbers = np.array(ring_numbers)
    line_numbers = np.array(line_numbers)

    # Points are already grouped ray by ray (increasing angle), each ray outer
    # edge -> inner. Keep that grouping (a per-point angle sort interleaves the
    # rings of one ray, since they share an angle up to float noise) and only
    # rotate whole rays so the run starts on the ray holding the topmost point.
    start_ray = line_numbers[int(np.argmin(all_y))]
    shift = int(np.argmax(line_numbers == start_ray))
    all_x = np.roll(all_x, -shift)
    all_y = np.roll(all_y, -shift)
    ring_numbers = np.roll(ring_numbers, -shift)
    line_numbers = np.roll(line_numbers, -shift)

    points = [(float(x), float(y)) for x, y in zip(all_x, all_y)]
    orders = [
        {'ring': int(r), 'line': int(l)}
        for r, l in zip(ring_numbers, line_numbers)
    ]
    return points, orders


def generate_line(mask, params: Dict) -> Tuple[Points, Orders]:
    """Line pattern: find largest contiguous segment at centroid row (buffered)."""
    h, w = mask.shape[:2]
    num = int(params['points'])
    bmask = apply_buffer(mask, params.get('buffer_px', 0))

    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return [], []
    cx = int(np.round(xs.mean()))
    cy = int(np.round(ys.mean()))

    found_row = None
    sampling_x = None
    for off in range(h):
        for r in ((cy + off, cy - off) if off else (cy,)):
            if 0 <= r < h:
                cols = np.where(bmask[r] == 1)[0]
                if cols.size > 0:
                    diffs = np.diff(cols)
                    splits = np.where(diffs > 1)[0]
                    segments = []
                    start = 0
                    for s in splits:
                        segments.append(cols[start:s + 1])
                        start = s + 1
                    segments.append(cols[start:])
                    largest = max(segments, key=len)
                    xmin, xmax = float(largest[0]), float(largest[-1])
                    sampling_x = np.linspace(xmin, xmax, num)
                    found_row = r
                    break
        if found_row is not None:
            break

    if sampling_x is None:
        xmin, xmax = float(xs.min()), float(xs.max())
        sampling_x = np.linspace(xmin, xmax, num)
        found_row = cy

    points = [(float(x), float(found_row)) for x in sampling_x]
    orders = [{'line_pos': i + 1} for i in range(len(points))]
    return points, orders


def generate_random(mask, params: Dict) -> Tuple[Points, Orders]:
    """Random pattern: random sample from mask pixels."""
    num = int(params['points'])
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return [], []
    n = min(num, len(xs))
    indices = np.random.choice(len(xs), n, replace=False)
    points = [(float(xs[i]), float(ys[i])) for i in indices]
    orders: Orders = []  # No meaningful order for random
    return points, orders


_GENERATORS = {
    'grid': generate_grid,
    'ring': generate_ring,
    'line': generate_line,
    'random': generate_random,
}


def generate(pattern: str, mask, params: Dict) -> Tuple[Points, Orders]:
    """Dispatch to the named pattern generator ('grid'/'ring'/'line'/'random')."""
    fn = _GENERATORS.get(pattern)
    if fn is None:
        raise ValueError(f"Unknown pattern: {pattern!r}")
    return fn(mask, params)
