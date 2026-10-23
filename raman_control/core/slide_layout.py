"""
Quartz-slide layout on the 384-well plate, and the slide-change Z hop.

The plate holds 12 slides (3 row-groups x 4 column-groups, 16 spots each)
separated by physical barriers. Moving the objective from one slide to another
at focus height can hit a barrier, so the stage first lifts to the objective
autofocus "Highest" limit + a clearance, travels XY, then lowers back to
"Highest" (see :func:`plan_slide_hop`). Used by both the Robotic Deposition
worker and the manual objective-view moves in the main window.

Slides are numbered 1-12 exactly like the labels on the sample map: snake
order, COL_GROUPS[0] is the bottom row of slides running left -> right (A..P),
the next row right -> left, and so on.
"""

from typing import List, Optional, Tuple

# Slide clusters: row indices (0-based, A=0) and column NUMBERS (1-based).
ROW_GROUPS = [
    [0, 1, 2, 3, 4],      # A-E
    [6, 7, 8, 9],          # G-J
    [11, 12, 13, 14, 15],  # L-P
]
COL_GROUPS = [
    [3, 4, 5, 6],
    [9, 10, 11, 12],
    [15, 16, 17, 18],
    [21, 22, 23, 24],
]


def slide_at_rc(row: int, col_num: int) -> Optional[int]:
    """Slide number (1-12) for 0-based row index + 1-based column number, or
    None if that grid position is not on a slide (a gap/barrier or off plate)."""
    slide_num = 1
    for cgi, cg in enumerate(COL_GROUPS):
        rgi_order = (range(len(ROW_GROUPS)) if cgi % 2 == 0
                     else range(len(ROW_GROUPS) - 1, -1, -1))
        for rgi in rgi_order:
            if row in ROW_GROUPS[rgi] and col_num in cg:
                return slide_num
            slide_num += 1
    return None


def slide_of_well(name: str) -> Optional[int]:
    """Slide number (1-12) a well (e.g. 'H10') sits on. None for a name that
    doesn't parse or isn't on any slide."""
    try:
        name = name.strip().upper()
        row = ord(name[0]) - ord('A')
        col = int(name[1:])
    except (AttributeError, IndexError, ValueError):
        return None
    return slide_at_rc(row, col)


def slide_at_position(x: float, y: float, stage_cfg, drop_cfg) -> Optional[int]:
    """Slide under a LOGICAL (widefield-frame) stage position, by snapping to
    the nearest well of the plate grid (same geometry as the well-click move:
    x steps rows, y steps columns, anchored to home). None if the nearest grid
    position is not on a slide."""
    pitch = drop_cfg.well_pitch_mm
    if pitch <= 0:
        return None
    row = round((x - stage_cfg.home_x - drop_cfg.offset_home_to_a1_x) / pitch)
    col = round((y - stage_cfg.home_y - drop_cfg.offset_home_to_a1_y) / pitch)
    return slide_at_rc(row, col + 1)


Move = Tuple[Optional[float], Optional[float], Optional[float]]


def plan_slide_hop(target_x: float, target_y: float, current_z: float,
                   obj_highest_z: float, clearance: float,
                   z_max: float) -> List[Move]:
    """The (x, y, z) absolute moves that carry the objective over a barrier to
    (target_x, target_y): lift Z at the current XY, travel XY at that height,
    lower Z to ``obj_highest_z``. Larger Z = further from the slide. The lift
    never goes lower than the current Z and never above ``z_max``."""
    lift_z = min(max(obj_highest_z + clearance, current_z), z_max)
    return [
        (None, None, lift_z),
        (target_x, target_y, lift_z),
        (None, None, obj_highest_z),
    ]
