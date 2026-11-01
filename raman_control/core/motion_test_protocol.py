"""
Unit 1 motion test (instr_unit1_motion_test_2026-10): test definitions and
G-code safety checks.

Each test is a list of steps that maps 1:1 onto a row ("#" column) of the
matching tab of instr_unit1_motion_checklist_2026-10.xlsx. A step's G-code is
exactly what the checklist lists; it is executed in RELATIVE mode inside one
atomic block (G91 ... G90, see StepperStage.run_relative_block), and the image
named on the checklist is taken once the block (ending in M400 + G4 S2) has
been acknowledged.

Pure Python (no Qt, no hardware) so it can be unit-tested.
"""

from dataclasses import dataclass
import math
import re
from typing import Dict, List, Optional, Tuple

# Checklist tab names (exact sheet titles in the template workbook).
SHEET_SETUP = 'Setup'
SHEET_T0 = 'T0 Noise'
SHEET_T1 = 'T1 Calibration'
SHEET_T2 = {'X': 'T2 Steps X', 'Y': 'T2 Steps Y'}
SHEET_T3 = {'X': 'T3 Repeat X', 'Y': 'T3 Repeat Y'}
SHEET_T4 = 'T4 Settling'
SHEET_T5 = 'T5 Drift'
SHEET_T6 = 'T6 Z'
SHEET_HANDOFF = 'Hand-off'

# Session sub-folders (protocol setup step 6).
FOLDERS = ['T0_noise', 'T1_calibration', 'T2_steps_X', 'T2_steps_Y',
           'T3_repeat_X', 'T3_repeat_Y', 'T4_settle', 'T5_drift', 'T6_Z']

# "Before you send" image counts per folder.
EXPECTED_IMAGE_COUNTS = {
    'T0_noise': 10,
    'T2_steps_X': 16, 'T2_steps_Y': 16,
    'T3_repeat_X': 21, 'T3_repeat_Y': 21,
    'T5_drift': 31,
    'T6_Z': 54,
}

SESSION_PREFIX = 'unit1_motion_'
CHECKLIST_TEMPLATE_NAME = 'instr_unit1_motion_checklist_2026-10.xlsx'

# Setup rule 3: never be more than this far below the focus (start) Z.
MAX_BELOW_START_Z_MM = 0.050
# Prusa MK3/MK3S firmware default max Z feedrate is 12 mm/s.
MAX_Z_FEEDRATE_MM_MIN = 720.0
MAX_XY_FEEDRATE_MM_MIN = 12000.0

# Wait after the block ACK before the next frame counts as "fresh": the
# exposure must have STARTED after the dwell ended.
FRAME_MARGIN_S = 0.02

WAIT = ('M400', 'G4 S2')


@dataclass(frozen=True)
class Step:
    """One checklist row."""
    row: int                       # value of the '#' column on the sheet
    action: str
    gcode: Tuple[str, ...] = ()    # relative-mode lines, run atomically
    image: Optional[str] = None    # file name taken after the G-code
    at_s: Optional[float] = None   # capture time relative to test start
    kind: str = 'auto'             # 'auto' | 'manual' | 'record' | 'check'
    record_s_after: float = 5.0    # 'record': keep recording after motion


@dataclass(frozen=True)
class TestDef:
    key: str
    title: str
    sheet: str
    folder: str
    steps: Tuple[Step, ...]
    compare: str = ''              # 'previous' | 'first' | '' (live shift check)
    needs_motion: bool = True


def _move(axis: str, mm: float, feed: int) -> str:
    """G1 line formatted exactly as the checklist writes it."""
    if abs(mm) >= 1 and float(mm).is_integer():
        val = f"{int(mm)}"
    elif abs(round(mm, 3) - mm) < 1e-12:
        val = f"{mm:.3f}"
    else:
        val = f"{mm:.4f}"
    return f"G1 {axis}{val} F{feed}"


# ---------------------------------------------------------------------------
# Test builders
# ---------------------------------------------------------------------------

def build_t0() -> TestDef:
    steps = tuple(
        Step(row=i, action=f'Image {i}', image=f'T0_{i:02d}.tif',
             at_s=(i - 1) * 5.0)
        for i in range(1, 11))
    return TestDef('T0', 'Test 0: noise floor', SHEET_T0, 'T0_noise', steps,
                   compare='first', needs_motion=False)


def build_t1() -> TestDef:
    """Interactive: framing jogs + two captures, driven from the panel."""
    steps = (
        Step(1, 'Sharp image of the VERTICAL bars', image='T1_vertical_bars.tif',
             kind='manual'),
        Step(2, 'Sharp image of the HORIZONTAL bars',
             image='T1_horizontal_bars.tif', kind='manual'),
        Step(3, 'Write the group and element used in Notes', kind='manual'),
        Step(4, 'Return to the coarse-element spot chosen in Setup',
             kind='manual'),
    )
    return TestDef('T1', 'Test 1: pixel calibration', SHEET_T1,
                   'T1_calibration', steps, needs_motion=False)


def build_t2(axis: str, step_um: float = 10.0) -> TestDef:
    """Test 2 staircase. ``step_um`` other than the protocol's 10 um builds
    the optional EXTRA run: same sequence, own folder / file names / key, and
    a sheet name that is not on the checklist (so it never fills T2 rows)."""
    a = axis.upper()
    extra = abs(step_um - 10.0) > 1e-9
    tag = f"_{step_um:g}um" if extra else ''
    fwd = _move(a, step_um / 1000.0, 60)
    back = _move(a, -step_um / 1000.0, 60)
    steps = [
        Step(1, 'Take-up move 1 (no image)', gcode=(fwd,)),
        Step(2, 'Take-up move 2 (no image)', gcode=(fwd,)),
        Step(3, 'Start image', gcode=WAIT, image=f'T2_{a}{tag}_step00.tif'),
    ]
    row = 4
    for n in range(1, 16):
        if n <= 5:
            label, line = f'Forward: step {n}', fwd
        elif n <= 10:
            label, line = f'Back: step {n}', back
        else:
            label, line = f'Forward again: step {n}', fwd
        steps.append(Step(row, label, gcode=(line,) + WAIT,
                          image=f'T2_{a}{tag}_step{n:02d}.tif'))
        row += 1
    steps.append(Step(19, 'Check: image shifted at most steps?', kind='check'))
    if extra:
        return TestDef(f'T2{a}{tag}', f'Test 2 extra: {step_um:g} um staircase {a}',
                       f'T2 extra {a}{tag}', f'T2_steps_{a}{tag}', tuple(steps),
                       compare='previous')
    return TestDef(f'T2{a}', f'Test 2: step staircase {a}', SHEET_T2[a],
                   f'T2_steps_{a}', tuple(steps), compare='previous')


def build_t3(axis: str) -> TestDef:
    a = axis.upper()
    lift, drop = _move('Z', 1, 300), _move('Z', -1, 300)
    minus = (lift, _move(a, -5, 1500), _move(a, 5, 1500), drop) + WAIT
    plus = (lift, _move(a, 5, 1500), _move(a, -5, 1500), drop) + WAIT
    steps = [Step(1, 'Reference image at the Test 2 spot', gcode=WAIT,
                  image=f'T3_{a}_ref.tif')]
    for i in range(1, 11):
        steps.append(Step(1 + i, f'Return from minus side, {i} of 10',
                          gcode=minus, image=f'T3_{a}_minus{i:02d}.tif'))
    for i in range(1, 11):
        steps.append(Step(11 + i, f'Return from plus side, {i} of 10',
                          gcode=plus, image=f'T3_{a}_plus{i:02d}.tif'))
    return TestDef(f'T3{a}', f'Test 3: repeatability {a}', SHEET_T3[a],
                   f'T3_repeat_{a}', tuple(steps), compare='first')


def build_t4() -> TestDef:
    steps = [Step(1, 'Record at max frame rate; fps goes in Notes',
                  kind='manual')]
    row = 2
    for a in ('X', 'Y'):
        for run in (1, 2, 3):
            steps.append(Step(
                row, 'Record, send both moves, keep recording 5 s after stop',
                gcode=(_move(a, 5, 1500), _move(a, -5, 1500), 'M400'),
                image=f'T4_{a}_run{run}', kind='record'))
            row += 1
    return TestDef('T4', 'Test 4: settling', SHEET_T4, 'T4_settle',
                   tuple(steps))


def build_t5() -> TestDef:
    steps = [Step(1, 'Room temperature at start (Notes)', kind='manual')]
    for m in range(0, 31):
        steps.append(Step(2 + m, f'Minute {m}', image=f'T5_{m:02d}.tif',
                          at_s=m * 60.0))
    steps.append(Step(33, 'Room temperature at end (Notes)', kind='manual'))
    return TestDef('T5', 'Test 5: drift', SHEET_T5, 'T5_drift', tuple(steps),
                   compare='first', needs_motion=False)


def build_t6a() -> TestDef:
    up20, dn = _move('Z', 0.020, 60), _move('Z', -0.0025, 60)
    up = _move('Z', 0.0025, 60)
    steps = [Step(1, '6A: target at sharpest focus by eye', kind='manual'),
             Step(2, '6A: go 20 um up (safe)', gcode=(up20,) + WAIT,
                  image='T6_sweep_down00.tif')]
    for i in range(1, 17):
        steps.append(Step(2 + i, f'6A down sweep {i} of 16', gcode=(dn,) + WAIT,
                          image=f'T6_sweep_down{i:02d}.tif'))
    for i in range(1, 17):
        steps.append(Step(18 + i, f'6A up sweep {i} of 16', gcode=(up,) + WAIT,
                          image=f'T6_sweep_up{i:02d}.tif'))
    return TestDef('T6A', 'Test 6A: focus sweep', SHEET_T6, 'T6_Z',
                   tuple(steps), compare='previous')


def build_t6b() -> TestDef:
    above = (_move('Z', 1, 300), _move('Z', -1, 300)) + WAIT
    below = (_move('Z', -0.020, 60), _move('Z', 0.020, 60)) + WAIT
    steps = [Step(35, '6B: target at best focus; reference image', gcode=WAIT,
                  image='T6_ref.tif')]
    for i in range(1, 11):
        steps.append(Step(35 + i, f'6B from above {i} of 10', gcode=above,
                          image=f'T6_above{i:02d}.tif'))
    for i in range(1, 11):
        steps.append(Step(45 + i, f'6B from below {i} of 10', gcode=below,
                          image=f'T6_below{i:02d}.tif'))
    return TestDef('T6B', 'Test 6B: Z repeatability', SHEET_T6, 'T6_Z',
                   tuple(steps), compare='first')


def all_tests() -> Dict[str, TestDef]:
    tests = [build_t0(), build_t1(), build_t2('X'), build_t2('Y'),
             build_t3('X'), build_t3('Y'), build_t4(), build_t5(),
             build_t6a(), build_t6b()]
    return {t.key: t for t in tests}


# ---------------------------------------------------------------------------
# G-code parsing / safety checks
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    level: str      # 'error' (blocks the run) | 'warning' | 'info'
    message: str


def parse_g1(line: str) -> Tuple[Dict[str, float], Optional[float]]:
    """Axis deltas and feedrate of a G0/G1 line ({}, None for other lines)."""
    parts = line.split(';', 1)[0].split()
    if not parts or parts[0].upper() not in ('G0', 'G1'):
        return {}, None
    deltas: Dict[str, float] = {}
    feed = None
    for p in parts[1:]:
        k, v = p[:1].upper(), p[1:]
        if k in 'XYZ':
            deltas[k] = float(v)
        elif k == 'F':
            feed = float(v)
    return deltas, feed


def parse_m92(text: str) -> Dict[str, float]:
    """Steps/mm from M503 output (the line containing 'M92')."""
    for line in text.splitlines():
        if 'M92' in line:
            out = {}
            for axis, val in re.findall(r'\b([XYZE])\s*([-+]?\d+(?:\.\d+)?)',
                                        line.split('M92', 1)[1]):
                out[axis] = float(val)
            if out:
                return out
    return {}


def m92_line(text: str) -> str:
    for line in text.splitlines():
        if 'M92' in line:
            return line.replace('echo:', '').strip()
    return ''


def check_blocks(
    blocks: List[Tuple[str, ...]],
    start_xyz: Optional[Tuple[float, float, float]] = None,
    limits: Optional[Dict[str, Tuple[float, float]]] = None,
    steps_per_mm: Optional[Dict[str, float]] = None,
    max_below_z_mm: float = MAX_BELOW_START_Z_MM,
) -> List[Issue]:
    """Simulate relative blocks and report rule violations.

    Rules:
      * only G0/G1/G4/M400/M114 inside a relative block;
      * Z never goes more than ``max_below_z_mm`` below the start Z (a big
        negative Z move is only allowed as the return of a preceding lift);
      * XYZ stays inside the travel limits (soft endstops would silently
        clip the move and corrupt the measurement);
      * every move is a whole number of microsteps (if M92 is known);
      * feedrates within firmware maxima.
    """
    issues: List[Issue] = []
    off = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
    lo = dict(off)
    hi = dict(off)
    allowed = ('G0', 'G1', 'G4', 'M400', 'M114')
    seen_step_warn = set()
    for block in blocks:
        for line in block:
            head = line.split(';', 1)[0].split()
            if not head or head[0].upper() not in allowed:
                issues.append(Issue('error', f"'{line}' is not allowed in a "
                                             f"relative block"))
                continue
            deltas, feed = parse_g1(line)
            for axis, d in deltas.items():
                off[axis] += d
                lo[axis] = min(lo[axis], off[axis])
                hi[axis] = max(hi[axis], off[axis])
                if axis == 'Z' and off['Z'] < -max_below_z_mm - 1e-9:
                    issues.append(Issue(
                        'error',
                        f"'{line}' takes Z {abs(off['Z']) * 1000:.1f} um below "
                        f"the start (focus); limit is "
                        f"{max_below_z_mm * 1000:.0f} um"))
                if steps_per_mm and axis in steps_per_mm:
                    n = d * steps_per_mm[axis]
                    if abs(n - round(n)) > 1e-6 and (axis, d) not in seen_step_warn:
                        seen_step_warn.add((axis, d))
                        issues.append(Issue(
                            'warning',
                            f"{axis}{d:+g} mm = {n:g} microsteps "
                            f"(M92 {axis}{steps_per_mm[axis]:g}); the firmware "
                            f"rounds it"))
            if feed is not None and deltas:
                fmax = (MAX_Z_FEEDRATE_MM_MIN if set(deltas) == {'Z'}
                        else MAX_XY_FEEDRATE_MM_MIN)
                if feed > fmax:
                    issues.append(Issue('warning', f"'{line}' feedrate F{feed:g} "
                                                   f"exceeds {fmax:g} mm/min"))
    if start_xyz is not None and limits:
        for i, axis in enumerate('XYZ'):
            if axis not in limits:
                continue
            amin, amax = limits[axis]
            if start_xyz[i] + lo[axis] < amin - 1e-9:
                issues.append(Issue(
                    'error', f"{axis} would reach {start_xyz[i] + lo[axis]:.3f} mm, "
                             f"below the travel limit {amin:g} mm"))
            if start_xyz[i] + hi[axis] > amax + 1e-9:
                issues.append(Issue(
                    'error', f"{axis} would reach {start_xyz[i] + hi[axis]:.3f} mm, "
                             f"above the travel limit {amax:g} mm"))
    net = {k: v for k, v in off.items() if abs(v) > 1e-9}
    if net:
        txt = ', '.join(f"{k}{v * 1000:+.1f} um" for k, v in net.items())
        issues.append(Issue('info', f"Net displacement at the end: {txt}"))
    return issues


def check_test(test: TestDef, **kwargs) -> List[Issue]:
    blocks = [s.gcode for s in test.steps if s.gcode]
    return check_blocks(blocks, **kwargs)


def stage_limits(stage_cfg) -> Dict[str, Tuple[float, float]]:
    return {'X': (stage_cfg.x_min, stage_cfg.x_max),
            'Y': (stage_cfg.y_min, stage_cfg.y_max),
            'Z': (stage_cfg.z_min, stage_cfg.z_max)}


def jog_block(axis: str, um: float) -> Tuple[str, ...]:
    """Framing / focusing jog as a relative block (F60, then M400)."""
    return (_move(axis.upper(), um / 1000.0, 60), 'M400')


def estimate_duration_s(test: TestDef) -> float:
    """Rough wall-clock estimate for the progress display."""
    total = 0.0
    for s in test.steps:
        if s.at_s is not None:
            total = max(total, s.at_s + 1.0)
            continue
        for line in s.gcode:
            deltas, feed = parse_g1(line)
            if deltas and feed:
                dist = math.sqrt(sum(d * d for d in deltas.values()))
                total += dist / (feed / 60.0) + 0.1
            elif line.startswith('G4 S'):
                total += float(line[4:])
        if s.image:
            total += 0.3
        if s.kind == 'record':
            total += 1.0 + s.record_s_after
    return total
