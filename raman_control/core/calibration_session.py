"""
Session storage for the unit 1 motion test (calibration tab).

Owns the ``unit1_motion_YYYY-MM-DD`` folder: creates the per-test
sub-folders, writes TIFFs and per-folder capture logs, keeps the checklist
state in ``session.json`` (so a session survives an app restart), and writes
the filled-in checklist workbook and data sheet.

No Qt and no hardware access, so it can be unit-tested.
"""

import csv
import datetime as _dt
import json
import os
from typing import Dict, Optional

import numpy as np

from raman_control.config.logging_config import get_logger
from raman_control.core import motion_test_protocol as mtp

logger = get_logger('calibration_session')

STATUS_DONE = 'Done'
STATUS_SKIPPED = 'Skipped'
STATUS_PROBLEM = 'Problem'

# PDF "Data sheet and hand-off" table, in order: (key, label).
DATA_SHEET_FIELDS = [
    ('date', 'Date'),
    ('operator', 'Operator'),
    ('power_on_time', 'Time machine was powered on'),
    ('m503_m92', 'Output of M503 (line starting M92)'),
    ('exposure_gain', 'Camera exposure (ms) and gain'),
    ('t4_fps', 'Recording frame rate for Test 4 (fps)'),
    ('usaf_group_element', 'USAF group and element used'),
    ('t1_group_element', 'USAF group and element for Test 1'),
    ('room_temp', 'Room temperature (and humidity) at start / end'),
    ('t0_notes', 'Test 0 notes'),
    ('t1_notes', 'Test 1 notes'),
    ('t2_notes', 'Test 2 notes (X / Y)'),
    ('t3_notes', 'Test 3 notes (X / Y), including any blurry returns'),
    ('t4_notes', 'Test 4 notes'),
    ('t5_notes', 'Test 5 notes (anyone walk by, doors, HVAC?)'),
    ('t6_notes', 'Test 6 notes'),
    ('unexpected', 'Anything unexpected'),
]


def default_template_path() -> str:
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(pkg_root, 'resources', mtp.CHECKLIST_TEMPLATE_NAME)


def write_tiff(path: str, image: np.ndarray) -> None:
    """Write an uncompressed TIFF at the array's native dtype (8 or 16 bit)."""
    import cv2
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = np.ascontiguousarray(image)
    if arr.dtype not in (np.uint8, np.uint16):
        arr = arr.astype(np.uint16)
    # COMPRESSION_NONE = 1
    if not cv2.imwrite(path, arr, [cv2.IMWRITE_TIFF_COMPRESSION, 1]):
        raise IOError(f"Could not write {path}")


def now_hms() -> str:
    return _dt.datetime.now().strftime('%H:%M:%S')


class CalibrationSession:
    """State + files of one unit1_motion_YYYY-MM-DD session."""

    def __init__(self, base_dir: str, date: Optional[_dt.date] = None):
        self.date = date or _dt.date.today()
        self.base_dir = os.path.abspath(base_dir)
        self.root = os.path.join(
            self.base_dir, f"{mtp.SESSION_PREFIX}{self.date.isoformat()}")
        self.rows: Dict[str, Dict[str, dict]] = {}
        self.fields: Dict[str, str] = {'date': self.date.isoformat()}
        self.meta: Dict[str, object] = {}

    # --- folders / persistence --------------------------------------------

    def create(self) -> None:
        for sub in mtp.FOLDERS:
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)
        self.load()
        self.save()

    def folder(self, name: str) -> str:
        return os.path.join(self.root, name)

    @property
    def json_path(self) -> str:
        return os.path.join(self.root, 'session.json')

    def load(self) -> None:
        if not os.path.isfile(self.json_path):
            return
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.rows = data.get('rows', {})
            self.fields.update(data.get('fields', {}))
            self.meta.update(data.get('meta', {}))
        except Exception as e:  # noqa: BLE001
            logger.error(f"Could not read {self.json_path}: {e}")

    def save(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        tmp = self.json_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'rows': self.rows, 'fields': self.fields,
                       'meta': self.meta}, f, indent=2, default=str)
        os.replace(tmp, self.json_path)

    # --- checklist rows ------------------------------------------------------

    def set_row(self, sheet: str, row: int, status: Optional[str] = None,
                notes: Optional[str] = None, time: Optional[str] = None) -> None:
        entry = self.rows.setdefault(sheet, {}).setdefault(str(row), {})
        if status is not None:
            entry['status'] = status
            entry['time'] = time or now_hms()
        if notes is not None:
            entry['notes'] = notes

    def get_row(self, sheet: str, row: int) -> dict:
        return dict(self.rows.get(sheet, {}).get(str(row), {}))

    def clear_rows(self, sheet: str, rows) -> None:
        sheet_rows = self.rows.get(sheet, {})
        for r in rows:
            sheet_rows.pop(str(r), None)

    def existing_files(self, test: mtp.TestDef) -> list:
        out = []
        for s in test.steps:
            if not s.image:
                continue
            p = os.path.join(self.folder(test.folder), s.image)
            if os.path.exists(p):
                out.append(p)
        return out

    # --- verification ------------------------------------------------------------

    def verify_files(self) -> dict:
        """Image counts and names per folder vs. the protocol.

        Returns {'counts': {folder: (found, expected)}, 'missing': [...],
        'unexpected': [...], 'ok_counts': bool, 'ok_names': bool}.
        """
        tests = mtp.all_tests()
        expected: Dict[str, set] = {}
        for t in tests.values():
            for s in t.steps:
                if s.image and s.kind != 'record':
                    expected.setdefault(t.folder, set()).add(s.image)
        counts, missing, unexpected = {}, [], []
        for folder in mtp.FOLDERS:
            path = self.folder(folder)
            found = set()
            if os.path.isdir(path):
                found = {f for f in os.listdir(path)
                         if f.lower().endswith(('.tif', '.tiff'))}
            exp = expected.get(folder, set())
            if folder in mtp.EXPECTED_IMAGE_COUNTS or exp:
                counts[folder] = (len(found), mtp.EXPECTED_IMAGE_COUNTS.get(
                    folder, len(exp)))
            missing += [f"{folder}/{f}" for f in sorted(exp - found)]
            unexpected += [f"{folder}/{f}" for f in sorted(found - exp)]
        # Test 4 recordings: one sub-folder per run, each with frames.
        t4_runs = [s.image for s in tests['T4'].steps if s.kind == 'record']
        for run in t4_runs:
            run_dir = os.path.join(self.folder('T4_settle'), run)
            n = 0
            if os.path.isdir(run_dir):
                n = len([f for f in os.listdir(run_dir)
                         if f.lower().endswith('.tif')])
            if n == 0:
                missing.append(f"T4_settle/{run}/")
        ok_counts = all(found == exp for found, exp in counts.values())
        return {'counts': counts, 'missing': missing, 'unexpected': unexpected,
                'ok_counts': ok_counts and not missing,
                'ok_names': not unexpected and not missing}

    # --- outputs ------------------------------------------------------------------

    def checklist_path(self) -> str:
        return os.path.join(self.root, mtp.CHECKLIST_TEMPLATE_NAME)

    def write_checklist(self, template_path: Optional[str] = None) -> str:
        """Fill Status / Time / Notes (columns E/F/G) of every row we have
        state for and save the workbook into the session folder.

        The template is never modified; the copy in the session folder is
        regenerated from it each time, so it always reflects session.json.
        """
        import openpyxl
        template_path = template_path or default_template_path()
        out = self.checklist_path()
        wb = openpyxl.load_workbook(template_path)
        for sheet, rows in self.rows.items():
            if sheet not in wb.sheetnames:
                continue
            ws = wb[sheet]
            row_index = {}
            for r in range(5, ws.max_row + 1):
                v = ws.cell(row=r, column=1).value
                if isinstance(v, (int, float)):
                    row_index[int(v)] = r
            for key, entry in rows.items():
                r = row_index.get(int(key))
                if r is None:
                    continue
                if entry.get('status'):
                    ws.cell(row=r, column=5, value=entry['status'])
                if entry.get('time'):
                    ws.cell(row=r, column=6, value=entry['time'])
                if entry.get('notes'):
                    ws.cell(row=r, column=7, value=entry['notes'])
        tmp = out + '.tmp.xlsx'
        wb.save(tmp)
        os.replace(tmp, out)
        return out

    def write_data_sheet(self) -> str:
        path = os.path.join(self.root, 'data_sheet.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['Field', 'Value'])
            for key, label in DATA_SHEET_FIELDS:
                w.writerow([label, self.fields.get(key, '')])
        return path

    def write_metadata(self) -> str:
        path = os.path.join(self.root, 'metadata.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.meta, f, indent=2, default=str)
        return path
