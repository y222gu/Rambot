"""
Calibration tab: unit 1 motion test (instr_unit1_motion_test_2026-10).

Runs Tests 0-6 with the Prusa stage and the brightfield camera, saves the
images under unit1_motion_YYYY-MM-DD/<test folder>/ with the protocol's file
names, and fills in instr_unit1_motion_checklist_2026-10.xlsx.

Isolation from the Live / Auto Acquisition tabs:
  * every move is an atomic relative block (G91 ... G90) on the motion
    thread's queue, so the firmware is back in absolute mode after each
    block and Live-tab moves can never interleave with a relative move;
  * images come from an opt-in raw-frame tap on the camera thread; both
    cameras keep acquiring and the Live view is untouched;
  * a test cannot start while an acquisition or autofocus runs, and the
    main window locks the rest of the UI while a test runs.
"""

import dataclasses
import os
import threading
import time
from typing import Callable, Dict, Optional

import numpy as np
from PyQt6.QtCore import Qt, QSettings, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QVBoxLayout, QWidget)

from raman_control.config.logging_config import get_logger
from raman_control.core import motion_test_protocol as mtp
from raman_control.core.calibration_session import (
    CalibrationSession, STATUS_DONE, STATUS_PROBLEM)
from raman_control.gui.styles import COLORS
from raman_control.gui.threads.calibration_thread import (
    CalibrationWorker, FrameGrabber)
from raman_control.gui.widgets.live_view import ImageDisplayWidget

logger = get_logger('calibration_panel')

_SETTINGS = ('Rambot', 'calibration')

# Setup tab rows the operator confirms by hand: (row, checkbox text).
_SETUP_MANUAL = [
    (1, "Raman laser OFF, key / interlock removed"),
    (2, "Normal start-up and homing done"),
    (3, "Z rule read: no negative Z move > 0.050 mm near focus"),
    (4, "USAF target flat, chrome side up, corners taped"),
    (5, "Brightfield LED on, bars in focus"),
    (6, "Centered on the chosen group / element (enter below)"),
    (10, "Waited 30 min after power-on (temp / humidity below)"),
]

# Data-sheet text fields: key -> placeholder.
_FIELD_PLACEHOLDERS = {
    'operator': 'Operator name',
    'power_on_time': 'HH:MM',
    'usaf_group_element': 'e.g. Group 5, Element 1',
    'env_start': 'e.g. 21.5 C, 40 %',
    'env_end': 'e.g. 21.8 C, 41 %',
    't1_group_element': 'e.g. Group 7, Element 1',
    't5_env_start': 'T / RH at start',
    't5_env_end': 'T / RH at end',
    'unexpected': 'Anything unexpected',
}

_TEST_NOTE_KEYS = {
    'T0': 't0_notes', 'T1': 't1_notes', 'T2': 't2_notes', 'T3': 't3_notes',
    'T4': 't4_notes', 'T5': 't5_notes', 'T6': 't6_notes',
}


def _default_base_dir() -> str:
    """The Rambot folder (parent of the raman_control package), the same
    place the Auto Acquisition save dialog starts in."""
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return os.path.dirname(pkg_root)


class CalibrationPanelWidget(QWidget):
    """The Calibration tab."""

    running_changed = pyqtSignal(bool)
    status_message = pyqtSignal(str, int)
    progress = pyqtSignal(int, str)
    _m503_done = pyqtSignal(bool, str)

    def __init__(self, motion_thread, camera, stage_config,
                 busy_reason: Optional[Callable[[], Optional[str]]] = None,
                 parent=None):
        super().__init__(parent)
        self._motion = motion_thread
        self._camera_thread = camera      # widefield CameraThread (fixed)
        self._stage_cfg = stage_config
        self._busy_reason = busy_reason or (lambda: None)
        self._tests = mtp.all_tests()
        self._session: Optional[CalibrationSession] = None
        self._worker: Optional[CalibrationWorker] = None
        self._running_key: Optional[str] = None
        self._grabber = FrameGrabber()
        self._preview_cam = None
        self._last_preview = 0.0
        self._t5_start: Optional[float] = None
        self._fields: Dict[str, QLineEdit] = {}
        self._checks: Dict[tuple, QCheckBox] = {}
        self._test_status: Dict[str, QLabel] = {}
        self._settings = QSettings(*_SETTINGS)

        self._build_ui()
        self._m503_done.connect(self._on_m503_done)

        self._sat_timer = QTimer(self)
        self._sat_timer.setInterval(500)
        self._sat_timer.timeout.connect(self._update_camera_info)

        self._refresh_enabled()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(12)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFixedWidth(620)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left = QWidget()
        lay = QVBoxLayout(left)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(8)
        lay.addWidget(self._build_session_group())
        lay.addWidget(self._build_setup_group())
        lay.addWidget(self._build_jog_group())
        lay.addWidget(self._build_tests_group())
        lay.addWidget(self._build_handoff_group())
        lay.addStretch(1)
        scroll.setWidget(left)
        root.addWidget(scroll)

        right = QVBoxLayout()
        right.setSpacing(6)
        self._preview = ImageDisplayWidget()
        self._preview.setMinimumSize(480, 360)
        right.addWidget(self._preview, 3)
        self._cam_info = QLabel("Camera: -")
        self._cam_info.setStyleSheet(f"color: {COLORS['text_secondary']};")
        right.addWidget(self._cam_info)
        self._sat_label = QLabel("")
        right.addWidget(self._sat_label)

        stop_row = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        stop_row.addWidget(self._progress, 1)
        self._abort_btn = QPushButton("Abort test")
        self._abort_btn.setToolTip(
            "Stop after the current move block finishes (about 3 s at most).\n"
            "The printer is left in absolute mode; nothing else is changed.")
        self._abort_btn.clicked.connect(self._on_abort)
        stop_row.addWidget(self._abort_btn)
        self._estop_btn = QPushButton("E-STOP (M112)")
        self._estop_btn.setProperty('danger', True)
        self._estop_btn.setToolTip(
            "Emergency stop: sends M112 immediately, bypassing the command "
            "queue.\nThe printer halts and must be reset (or power-cycled), "
            "reconnected and re-homed.\nUse Abort for a normal stop.")
        self._estop_btn.clicked.connect(self._on_estop)
        stop_row.addWidget(self._estop_btn)
        right.addLayout(stop_row)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        mono = QFont('Consolas')
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self._log.setFont(mono)
        right.addWidget(self._log, 2)
        root.addLayout(right, 1)

    def _line(self, key: str, width: Optional[int] = None) -> QLineEdit:
        edit = QLineEdit()
        edit.setPlaceholderText(_FIELD_PLACEHOLDERS.get(key, ''))
        if width:
            edit.setFixedWidth(width)
        edit.editingFinished.connect(lambda k=key: self._on_field_edited(k))
        self._fields[key] = edit
        return edit

    def _build_session_group(self) -> QGroupBox:
        box = QGroupBox("Session")
        g = QGridLayout(box)
        g.setColumnStretch(1, 1)

        g.addWidget(QLabel("Save in"), 0, 0)
        self._base_edit = QLineEdit(
            self._settings.value('base_dir', _default_base_dir()))
        g.addWidget(self._base_edit, 0, 1)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._on_browse)
        g.addWidget(browse, 0, 2)

        g.addWidget(QLabel("Operator"), 1, 0)
        g.addWidget(self._line('operator'), 1, 1, 1, 2)
        g.addWidget(QLabel("Powered on at"), 2, 0)
        g.addWidget(self._line('power_on_time', 90), 2, 1)

        g.addWidget(QLabel("Camera"), 3, 0)
        g.addWidget(QLabel("Widefield camera"), 3, 1, 1, 2)

        self._open_btn = QPushButton("Open session folder (today)")
        self._open_btn.setProperty('primary', True)
        self._open_btn.clicked.connect(self._on_open_session)
        g.addWidget(self._open_btn, 4, 0, 1, 3)
        self._session_label = QLabel("No session open")
        self._session_label.setWordWrap(True)
        self._session_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        g.addWidget(self._session_label, 5, 0, 1, 3)
        return box

    def _build_setup_group(self) -> QGroupBox:
        box = QGroupBox("Setup")
        v = QVBoxLayout(box)
        for row, text in _SETUP_MANUAL:
            cb = QCheckBox(text)
            cb.toggled.connect(lambda on, r=row: self._on_check(mtp.SHEET_SETUP, r, on))
            self._checks[(mtp.SHEET_SETUP, row)] = cb
            v.addWidget(cb)
        g = QGridLayout()
        g.setColumnStretch(1, 1)
        g.addWidget(QLabel("USAF group / element"), 0, 0)
        g.addWidget(self._line('usaf_group_element'), 0, 1)
        g.addWidget(QLabel("Room temp, humidity (start)"), 1, 0)
        g.addWidget(self._line('env_start'), 1, 1)
        v.addLayout(g)

        m = QHBoxLayout()
        self._m503_btn = QPushButton("Read M503")
        self._m503_btn.setToolTip("Query the printer settings and record the "
                                  "M92 (steps/mm) line.")
        self._m503_btn.clicked.connect(self._on_read_m503)
        m.addWidget(self._m503_btn)
        self._m92_label = QLabel("M92: not read")
        self._m92_label.setWordWrap(True)
        m.addWidget(self._m92_label, 1)
        v.addLayout(m)
        note = QLabel("Exposure: set it in the Live tab (widefield view) before "
                      "Test 0, then do "
                      "not change it. This software has no auto-exposure / "
                      "auto-gain.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {COLORS['text_secondary']};")
        v.addWidget(note)
        return box

    def _build_jog_group(self) -> QGroupBox:
        box = QGroupBox("Fine jog (Prusa, relative, F60)")
        v = QVBoxLayout(box)
        self._jog_buttons = []
        rows = [('X', (-20, -10, 10, 20)), ('Y', (-20, -10, 10, 20)),
                ('Z', (-10, -2.5, 2.5, 10, 20))]
        for axis, steps in rows:
            h = QHBoxLayout()
            lbl = QLabel(axis)
            lbl.setFixedWidth(16)
            h.addWidget(lbl)
            for um in steps:
                btn = QPushButton(f"{um:+g} um")
                btn.setFixedWidth(76)
                btn.clicked.connect(lambda _=False, a=axis, d=um: self._on_jog(a, d))
                h.addWidget(btn)
                self._jog_buttons.append(btn)
            h.addStretch(1)
            v.addLayout(h)
        hint = QLabel("For framing (Test 1) and focusing. Each click is one "
                      "G91 move + M400, then G90. Use the Live tab for larger "
                      "moves.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {COLORS['text_secondary']};")
        v.addWidget(hint)
        return box

    def _status_label(self, key: str) -> QLabel:
        lbl = QLabel("")
        lbl.setMinimumWidth(110)
        self._test_status[key] = lbl
        return lbl

    def _run_button(self, text: str, key: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedWidth(190)
        btn.clicked.connect(lambda _=False, k=key: self._start_test(k))
        return btn

    def _build_tests_group(self) -> QGroupBox:
        box = QGroupBox("Tests")
        g = QGridLayout(box)
        g.setColumnStretch(2, 1)
        self._run_buttons = []
        r = 0

        def notes(test_group: str) -> QLineEdit:
            edit = self._line(_TEST_NOTE_KEYS[test_group])
            edit.setPlaceholderText(f"Test {test_group[1]} notes")
            return edit

        def add_run(label, key, row):
            btn = self._run_button(label, key)
            self._run_buttons.append(btn)
            g.addWidget(btn, row, 0)
            g.addWidget(self._status_label(key), row, 1)
            return btn

        # T0
        add_run("Run T0: noise (50 s)", 'T0', r)
        g.addWidget(notes('T0'), r, 2)
        r += 1

        # T1 (interactive)
        self._t1_vert_btn = QPushButton("T1: capture vertical")
        self._t1_vert_btn.setFixedWidth(190)
        self._t1_vert_btn.clicked.connect(lambda: self._capture_t1(1))
        self._t1_horz_btn = QPushButton("T1: capture horizontal")
        self._t1_horz_btn.setFixedWidth(190)
        self._t1_horz_btn.clicked.connect(lambda: self._capture_t1(2))
        self._run_buttons += [self._t1_vert_btn, self._t1_horz_btn]
        g.addWidget(self._t1_vert_btn, r, 0)
        g.addWidget(self._status_label('T1'), r, 1)
        g.addWidget(notes('T1'), r, 2)
        r += 1
        g.addWidget(self._t1_horz_btn, r, 0)
        t1_ge = self._line('t1_group_element')
        g.addWidget(t1_ge, r, 1, 1, 2)
        r += 1
        cb = QCheckBox("T1: back at the coarse-element spot from Setup")
        cb.toggled.connect(lambda on: self._on_check(mtp.SHEET_T1, 4, on))
        self._checks[(mtp.SHEET_T1, 4)] = cb
        g.addWidget(cb, r, 0, 1, 3)
        r += 1

        add_run("Run T2 X: steps", 'T2X', r)
        g.addWidget(notes('T2'), r, 2)
        r += 1
        add_run("Run T2 Y: steps", 'T2Y', r)
        r += 1
        # Optional extra staircase at a larger step (not part of the protocol
        # checklist): shows where the XY dead band ends.
        extra = QHBoxLayout()
        extra.addWidget(QLabel("Extra T2, step"))
        self._t2_extra_combo = QComboBox()
        for um in (30, 60, 120):
            self._t2_extra_combo.addItem(f"{um} um", float(um))
        self._t2_extra_combo.setCurrentIndex(1)
        extra.addWidget(self._t2_extra_combo)
        for a in ('X', 'Y'):
            btn = QPushButton(f"Run extra T2 {a}")
            btn.setToolTip("Same staircase as Test 2 with the chosen step. "
                           "Saved to T2_steps_<axis>_<step>um; not written to "
                           "the checklist or counted in the hand-off check.")
            btn.clicked.connect(lambda _=False, ax=a: self._start_t2_extra(ax))
            extra.addWidget(btn)
            self._run_buttons.append(btn)
        extra.addStretch(1)
        g.addLayout(extra, r, 0, 1, 3)
        r += 1
        add_run("Run T3 X: repeat", 'T3X', r)
        g.addWidget(notes('T3'), r, 2)
        r += 1
        add_run("Run T3 Y: repeat", 'T3Y', r)
        r += 1
        add_run("Run T4: settling (6 runs)", 'T4', r)
        g.addWidget(notes('T4'), r, 2)
        r += 1

        add_run("Run T5: drift (30 min)", 'T5', r)
        g.addWidget(notes('T5'), r, 2)
        r += 1
        env = QHBoxLayout()
        env.addWidget(self._line('t5_env_start'))
        env.addWidget(self._line('t5_env_end'))
        g.addLayout(env, r, 0, 1, 3)
        r += 1
        ev = QHBoxLayout()
        self._event_edit = QLineEdit()
        self._event_edit.setPlaceholderText(
            "T5 event: HVAC on/off, door, someone walked by...")
        self._event_edit.returnPressed.connect(self._on_log_event)
        self._event_btn = QPushButton("Log event")
        self._event_btn.clicked.connect(self._on_log_event)
        ev.addWidget(self._event_edit, 1)
        ev.addWidget(self._event_btn)
        g.addLayout(ev, r, 0, 1, 3)
        r += 1

        cb6a = QCheckBox("6A: target at sharpest focus by eye")
        cb6a.toggled.connect(lambda on: self._on_check(mtp.SHEET_T6, 1, on))
        self._checks[(mtp.SHEET_T6, 1)] = cb6a
        g.addWidget(cb6a, r, 0, 1, 3)
        r += 1
        self._t6a_btn = add_run("Run T6A: focus sweep", 'T6A', r)
        g.addWidget(notes('T6'), r, 2)
        r += 1
        self._t6b_check = QCheckBox("6B: target back at best focus")
        self._t6b_check.toggled.connect(lambda _on: self._refresh_enabled())
        g.addWidget(self._t6b_check, r, 0, 1, 3)
        r += 1
        self._t6b_btn = add_run("Run T6B: Z repeat", 'T6B', r)
        r += 1
        return box

    def _build_handoff_group(self) -> QGroupBox:
        box = QGroupBox("Hand-off")
        v = QVBoxLayout(box)
        g = QGridLayout()
        g.setColumnStretch(1, 1)
        g.addWidget(QLabel("Room temp, humidity (end)"), 0, 0)
        g.addWidget(self._line('env_end'), 0, 1)
        g.addWidget(QLabel("Anything unexpected"), 1, 0)
        g.addWidget(self._line('unexpected'), 1, 1)
        v.addLayout(g)
        for row, text in ((7, "Laser key returned, machine left as found"),
                          (8, "Folder sent to Maria")):
            cb = QCheckBox(text)
            cb.toggled.connect(lambda on, r=row: self._on_check(mtp.SHEET_HANDOFF, r, on))
            self._checks[(mtp.SHEET_HANDOFF, row)] = cb
            v.addWidget(cb)
        self._save_btn = QPushButton("Verify files and save checklist")
        self._save_btn.setProperty('primary', True)
        self._save_btn.clicked.connect(lambda: self._save_outputs(verify=True))
        v.addWidget(self._save_btn)
        self._verify_label = QLabel("")
        self._verify_label.setWordWrap(True)
        v.addWidget(self._verify_label)
        return box

    # ------------------------------------------------------------ helpers

    def _log_msg(self, level: str, msg: str):
        color = {'error': COLORS['error'], 'warning': COLORS['warning'],
                 'ok': COLORS['success']}.get(level, COLORS['text_primary'])
        stamp = time.strftime('%H:%M:%S')
        self._log.appendHtml(
            f'<span style="color:{COLORS["text_secondary"]}">{stamp}</span> '
            f'<span style="color:{color}">{_html(msg)}</span>')

    def _camera(self):
        return self._camera_thread

    def _is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _stage_ready(self) -> bool:
        return bool(self._motion and self._motion.is_connected
                    and self._motion.all_referenced)

    def controls_enabled_while_running(self) -> list:
        """Widgets the main window must leave enabled while a test runs."""
        return [self._abort_btn, self._estop_btn, self._event_edit,
                self._event_btn]

    def _refresh_enabled(self):
        running = self._is_running()
        have_session = self._session is not None
        for btn in self._run_buttons:
            btn.setEnabled(have_session and not running)
        if have_session and not running:
            self._t6a_btn.setEnabled(self._checks[(mtp.SHEET_T6, 1)].isChecked())
            self._t6b_btn.setEnabled(self._t6b_check.isChecked())
        for btn in self._jog_buttons:
            btn.setEnabled(not running)
        self._open_btn.setEnabled(not running)
        self._save_btn.setEnabled(have_session and not running)
        self._m503_btn.setEnabled(not running)
        self._abort_btn.setEnabled(running)
        # Ticks / fields are stored in the session, so they need one open.
        for cb in self._checks.values():
            cb.setEnabled(have_session and not running)
        self._t6b_check.setEnabled(have_session and not running)
        for edit in self._fields.values():
            edit.setEnabled(have_session)
        is_t5 = running and self._running_key == 'T5'
        self._event_edit.setEnabled(is_t5)
        self._event_btn.setEnabled(is_t5)

    # ----------------------------------------------------------- preview

    def showEvent(self, event):
        super().showEvent(event)
        self._attach_camera()
        self._sat_timer.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._sat_timer.stop()
        if not self._is_running():
            self._detach_camera()

    def _attach_camera(self):
        cam = self._camera()
        if cam is self._preview_cam and self._grabber.camera is cam:
            return
        self._detach_camera()
        if cam is None:
            return
        self._grabber.attach(cam)
        cam.frame_ready.connect(self._on_frame)
        self._preview_cam = cam

    def _detach_camera(self):
        if self._preview_cam is not None:
            try:
                self._preview_cam.frame_ready.disconnect(self._on_frame)
            except (TypeError, RuntimeError):
                pass
        self._preview_cam = None
        self._grabber.detach()

    def _on_frame(self, frame: np.ndarray, _t: float):
        now = time.time()
        if now - self._last_preview < 0.12 or not self.isVisible():
            return
        self._last_preview = now
        self._preview.update_image(frame)

    def _update_camera_info(self):
        cam = self._camera()
        if cam is None:
            return
        info = cam.sensor_info if cam.is_connected else {}
        bits = info.get('bit_depth', 0)
        txt = ("Widefield camera: "
               + (f"{info.get('width')}x{info.get('height')} px, "
                  f"{bits}-bit, exposure {info.get('exposure_ms', 0):.2f} ms, "
                  f"gain {info.get('gain', 'n/a')}, {cam.current_fps:.1f} fps"
                  if info else "not connected"))
        self._cam_info.setText(txt)
        latest = self._grabber.latest()
        if latest is None:
            self._sat_label.setText("")
            return
        raw = latest[0]
        top = (1 << bits) - 1 if bits else (255 if raw.dtype == np.uint8 else 65535)
        mx = int(raw.max())
        n_sat = int(np.count_nonzero(raw >= top))
        if n_sat:
            self._sat_label.setText(f"SATURATED: {n_sat} px at {top}. "
                                    f"Lower the exposure before Test 0.")
            self._sat_label.setStyleSheet(f"color: {COLORS['error']}; font-weight: bold;")
        else:
            self._sat_label.setText(f"Max raw value {mx} / {top} ({100 * mx / top:.0f} %)")
            self._sat_label.setStyleSheet(f"color: {COLORS['success']};")

    # ----------------------------------------------------------- session

    def _on_browse(self):
        d = QFileDialog.getExistingDirectory(self, "Save sessions in",
                                             self._base_edit.text())
        if d:
            self._base_edit.setText(d)

    def _on_open_session(self):
        base = self._base_edit.text().strip()
        if not base:
            return
        try:
            os.makedirs(base, exist_ok=True)
            session = CalibrationSession(base)
            session.create()
        except Exception as e:  # noqa: BLE001
            self._log_msg('error', f"Could not create the session folder: {e}")
            return
        self._settings.setValue('base_dir', base)
        self._session = session
        if not session.fields.get('operator'):
            session.fields['operator'] = self._settings.value('operator', '')
        session.set_row(mtp.SHEET_SETUP, 8, STATUS_DONE,
                        "Software saves uncompressed TIFF, full sensor, native "
                        "bit depth")
        session.set_row(mtp.SHEET_SETUP, 9, STATUS_DONE, session.root)
        session.set_row(mtp.SHEET_SETUP, 12, STATUS_DONE,
                        "Sent by the software as G91 at the start of every move "
                        "block, with G90 restored after it")
        session.save()
        self._load_widgets_from_session()
        self._session_label.setText(f"Session: {session.root}")
        self._log_msg('ok', f"Session folder ready: {session.root}")
        self._refresh_all_status()
        self._refresh_enabled()

    def _load_widgets_from_session(self):
        s = self._session
        for key, edit in self._fields.items():
            edit.blockSignals(True)
            edit.setText(str(s.fields.get(key, '')))
            edit.blockSignals(False)
        for (sheet, row), cb in self._checks.items():
            cb.blockSignals(True)
            cb.setChecked(s.get_row(sheet, row).get('status') == STATUS_DONE)
            cb.blockSignals(False)
        m92 = s.fields.get('m503_m92')
        if m92:
            self._m92_label.setText(f"M92: {m92}")

    def _on_field_edited(self, key: str):
        if key == 'operator':
            self._settings.setValue('operator', self._fields[key].text())
        if self._session is None:
            return
        self._session.fields[key] = self._fields[key].text().strip()
        self._sync_derived_rows()
        self._session.save()

    def _on_check(self, sheet: str, row: int, on: bool):
        self._refresh_enabled()
        if self._session is None:
            return
        if on:
            self._session.set_row(sheet, row, STATUS_DONE)
        else:
            self._session.clear_rows(sheet, [row])
        self._sync_derived_rows()
        self._session.save()

    def _sync_derived_rows(self):
        """Notes columns that come from typed-in fields."""
        s = self._session
        f = s.fields
        if f.get('usaf_group_element'):
            s.set_row(mtp.SHEET_SETUP, 6, notes=f['usaf_group_element'])
        if f.get('env_start'):
            s.set_row(mtp.SHEET_SETUP, 10, notes=f['env_start'])
        if f.get('t1_group_element'):
            s.set_row(mtp.SHEET_T1, 3, STATUS_DONE, f['t1_group_element'],
                      time=s.get_row(mtp.SHEET_T1, 3).get('time'))
        if f.get('t5_env_start'):
            s.set_row(mtp.SHEET_T5, 1, STATUS_DONE, f['t5_env_start'],
                      time=s.get_row(mtp.SHEET_T5, 1).get('time'))
        if f.get('t5_env_end'):
            s.set_row(mtp.SHEET_T5, 33, STATUS_DONE, f['t5_env_end'],
                      time=s.get_row(mtp.SHEET_T5, 33).get('time'))
        if f.get('operator'):
            s.set_row(mtp.SHEET_HANDOFF, 1, STATUS_DONE,
                      f"{f.get('date', '')} / {f['operator']}",
                      time=s.get_row(mtp.SHEET_HANDOFF, 1).get('time'))
        if f.get('power_on_time'):
            s.set_row(mtp.SHEET_HANDOFF, 2, STATUS_DONE, f['power_on_time'],
                      time=s.get_row(mtp.SHEET_HANDOFF, 2).get('time'))
        if f.get('env_start') or f.get('env_end'):
            env = f"{f.get('env_start', '?')} / {f.get('env_end', '?')}"
            f['room_temp'] = env
            s.set_row(mtp.SHEET_HANDOFF, 3, STATUS_DONE if f.get('env_end') else None,
                      env, time=s.get_row(mtp.SHEET_HANDOFF, 3).get('time'))
        if f.get('unexpected'):
            s.set_row(mtp.SHEET_HANDOFF, 4, STATUS_DONE, f['unexpected'],
                      time=s.get_row(mtp.SHEET_HANDOFF, 4).get('time'))

    def _refresh_all_status(self):
        if self._session is None:
            return
        for key, lbl in self._test_status.items():
            test = self._tests[key]
            # Count images (T4: recordings); other checklist rows (take-up
            # moves, notes, manual ticks) only contribute problems.
            img_rows = [s.row for s in test.steps if s.image]
            st = {s.row: self._session.get_row(test.sheet, s.row).get('status')
                  for s in test.steps}
            done = sum(1 for r in img_rows if st[r] == STATUS_DONE)
            prob = sum(1 for x in st.values() if x == STATUS_PROBLEM)
            if not done and not prob:
                lbl.setText("")
                continue
            unit = "recordings" if key == 'T4' else "images"
            txt = f"{done}/{len(img_rows)} {unit}"
            if prob:
                txt += f", {prob} problem"
            lbl.setText(txt)
            color = (COLORS['error'] if prob else COLORS['success']
                     if done == len(img_rows) else COLORS['warning'])
            lbl.setStyleSheet(f"color: {color};")

    # -------------------------------------------------------------- M503

    def _on_read_m503(self):
        if not (self._motion and self._motion.is_connected):
            self._log_msg('error', "Stage not connected")
            return
        self._m503_btn.setEnabled(False)

        def work():
            ok, text = self._motion.read_firmware_settings_blocking()
            self._m503_done.emit(ok, text)
        threading.Thread(target=work, name='M503', daemon=True).start()

    def _on_m503_done(self, ok: bool, text: str):
        self._refresh_enabled()
        line = mtp.m92_line(text) if ok else ''
        if not line:
            self._log_msg('error', "M503 failed or no M92 line in the reply")
            return
        spm = mtp.parse_m92(text)
        self._m92_label.setText(f"M92: {line}")
        self._log_msg('ok', f"M503: {line}")
        for axis, um in (('X', 10.0), ('Y', 10.0), ('Z', 2.5)):
            if axis in spm:
                n = um / 1000.0 * spm[axis]
                lvl = 'ok' if abs(n - round(n)) < 1e-6 else 'warning'
                self._log_msg(lvl, f"{axis}: {um:g} um = {n:g} microsteps "
                                   f"(1 microstep = {1000 / spm[axis]:g} um)")
        if self._session is not None:
            self._session.fields['m503_m92'] = line
            self._session.meta['m503'] = text
            self._session.meta['steps_per_mm'] = spm
            self._session.set_row(mtp.SHEET_SETUP, 11, STATUS_DONE, line)
            self._session.save()

    # --------------------------------------------------------------- jog

    def _on_jog(self, axis: str, um: float):
        if self._is_running():
            return
        if not self._stage_ready():
            self._log_msg('error', "Stage not connected / not referenced")
            return
        if axis == 'Z' and um < -mtp.MAX_BELOW_START_Z_MM * 1000:
            return
        block = mtp.jog_block(axis, um)
        if self._motion.request_relative_block(list(block)):
            self._log_msg('info', f"Jog {axis} {um:+g} um ({block[0]})")
        else:
            self._log_msg('error', "Jog refused (stage not ready)")

    # ------------------------------------------------------------- tests

    def _capture_t1(self, row: int):
        test = self._tests['T1']
        step = next(s for s in test.steps if s.row == row)
        auto = mtp.Step(step.row, step.action, gcode=('M400', 'G4 S2'),
                        image=step.image)
        self._start_test('T1', steps=[auto])

    def _start_t2_extra(self, axis: str):
        test = mtp.build_t2(axis, self._t2_extra_combo.currentData())
        self._tests[test.key] = test
        self._start_test(test.key)

    def _start_test(self, key: str, steps=None):
        if self._is_running():
            return
        if self._session is None:
            self._log_msg('error', "Open a session folder first")
            return
        reason = self._busy_reason()
        if reason:
            self._log_msg('error', f"Cannot start: {reason}")
            self.status_message.emit(f"Calibration: {reason}", 4000)
            return
        cam = self._camera()
        if cam is None or not cam.is_connected:
            self._log_msg('error', "Selected camera is not connected")
            return
        test = self._tests[key]
        run_steps = steps if steps is not None else list(test.steps)
        needs_stage = any(s.gcode for s in run_steps if s.kind != 'manual')
        if needs_stage and not self._stage_ready():
            self._log_msg('error', "Stage not connected / not referenced "
                                   "(home all axes first)")
            return

        # Overwrite check (data loss is the one thing worth a dialog here).
        # replace() keeps the test's other settings (e.g. compare mode).
        sub = dataclasses.replace(test, steps=tuple(run_steps))
        existing = self._session.existing_files(sub)
        if existing:
            reply = QMessageBox.question(
                self, "Overwrite images?",
                f"{len(existing)} file(s) of this test already exist in "
                f"{test.folder}. Run again and overwrite them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return

        info = cam.sensor_info
        exposure_ms = float(info.get('exposure_ms', 0.0))
        locked = self._session.meta.get('exposure_ms')
        if locked is not None and abs(float(locked) - exposure_ms) > 1e-6:
            reply = QMessageBox.question(
                self, "Exposure changed",
                f"The exposure was {float(locked):.3f} ms for earlier tests and "
                f"is now {exposure_ms:.3f} ms. The protocol says not to change "
                f"it. Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        meta = self._session.meta
        meta.setdefault('exposure_ms', exposure_ms)
        meta['camera'] = {**info, 'which': 'widefield'}
        cfg = getattr(cam, '_config', None)
        if cfg is not None:
            meta['camera_display_transforms'] = {
                'note': 'Saved TIFFs are the RAW sensor frames; these display '
                        'transforms are NOT applied to them.',
                'flip_horizontal': cfg.flip_horizontal,
                'rotate_minus90': cfg.rotate_minus90,
                'crop_enabled': cfg.crop_enabled,
                'pixel_size_um_config': cfg.pixel_size_x_um,
            }
        gain = info.get('gain', 'camera default')
        self._session.fields['exposure_gain'] = f"{exposure_ms:.3f} ms, gain {gain}"
        self._session.set_row(
            mtp.SHEET_SETUP, 7, STATUS_DONE,
            f"exposure {exposure_ms:.3f} ms, gain {gain}; no auto-exposure / "
            f"auto-gain in this software; {info.get('bit_depth', '?')}-bit raw")
        self._session.save()

        self._attach_camera()
        worker = CalibrationWorker(
            sub, self._motion, self._grabber,
            out_dir=self._session.folder(test.folder),
            exposure_s=exposure_ms / 1000.0,
            bit_depth=int(info.get('bit_depth', 0) or 0),
            stage_limits=mtp.stage_limits(self._stage_cfg),
            steps_per_mm=self._session.meta.get('steps_per_mm'),
        )
        worker.log.connect(self._log_msg)
        worker.row_result.connect(self._on_row_result)
        worker.progress.connect(self._on_progress)
        worker.fps_measured.connect(self._on_fps)
        worker.finished_test.connect(self._on_test_finished)
        self._worker = worker
        self._running_key = key
        if key == 'T5':
            self._t5_start = time.time()
            if not self._fields['t5_env_start'].text().strip():
                self._log_msg('warning', "Enter the room temperature / humidity "
                                         "at the start of Test 5")
        self._log_msg('info', f"Starting {test.title} -> {test.folder}")
        worker.start()
        self._refresh_enabled()
        self.running_changed.emit(True)

    def _on_progress(self, pct: int, text: str):
        self._progress.setValue(pct)
        self._progress.setFormat(f"{text}  %p%" if text else "%p%")
        self.progress.emit(pct, text)

    def _on_row_result(self, sheet: str, row: int, status: str, notes: str):
        if self._session is None:
            return
        self._session.set_row(sheet, row, status, notes)
        self._session.save()
        self._refresh_all_status()

    def _on_fps(self, fps: float):
        if self._session is None:
            return
        fps_list = self._session.meta.setdefault('t4_fps', [])
        fps_list.append(round(fps, 2))
        txt = f"{min(fps_list):.1f} to {max(fps_list):.1f} fps (measured)"
        self._session.fields['t4_fps'] = txt
        self._session.set_row(mtp.SHEET_T4, 1, STATUS_DONE, txt)

    def _on_test_finished(self, key: str, ok: bool, aborted: bool, summary: str):
        worker = self._worker
        self._worker = None
        self._running_key = None
        if worker is not None:
            worker.wait(5000)
        level = 'ok' if ok else ('warning' if aborted else 'error')
        self._log_msg(level, f"{self._tests[key].title}: {summary}")
        self.status_message.emit(f"Calibration {key}: {summary}", 6000)
        self._progress.setValue(0)
        self._progress.setFormat("%p%")
        self.progress.emit(0, "")
        if key == 'T5':
            self._t5_start = None
            if not self._fields['t5_env_end'].text().strip():
                self._log_msg('warning', "Enter the room temperature / humidity "
                                         "at the end of Test 5")
        self.running_changed.emit(False)
        self._refresh_enabled()
        if not self.isVisible():
            self._detach_camera()
        self._save_outputs(verify=False)

    def _on_abort(self):
        if self._is_running():
            self._worker.abort()
            self._log_msg('warning', "Abort requested: stopping after the "
                                     "current move block")

    def _on_estop(self):
        if self._worker is not None:
            self._worker.abort()
        if self._motion:
            self._motion.emergency_stop_immediate()
        self._log_msg('error', "E-STOP: M112 sent. Reset or power-cycle the "
                               "printer, then Disconnect / Connect and re-home.")
        self.status_message.emit("E-STOP sent (M112): reset the printer and "
                                 "re-home", 10000)

    def _on_log_event(self):
        text = self._event_edit.text().strip()
        if not text or self._session is None:
            return
        start = self._t5_start
        stamp = (time.strftime('%H:%M:%S') if start is None else
                 "min {:d}:{:02d}".format(*divmod(int(time.time() - start), 60)))
        events = self._session.meta.setdefault('t5_events', [])
        events.append(f"{stamp} {text}")
        self._session.save()
        self._event_edit.clear()
        self._log_msg('info', f"T5 event at {stamp}: {text}")

    # ----------------------------------------------------------- outputs

    def _save_outputs(self, verify: bool):
        s = self._session
        if s is None:
            return
        self._sync_derived_rows()
        events = s.meta.get('t5_events') or []
        if events:
            base = self._fields['t5_notes'].text().strip()
            s.fields['t5_notes'] = '; '.join(([base] if base else []) + events)
        if verify:
            res = s.verify_files()
            lines = [f"{k}: {found}/{exp}" for k, (found, exp) in res['counts'].items()]
            s.set_row(mtp.SHEET_HANDOFF, 5,
                      STATUS_DONE if res['ok_counts'] else STATUS_PROBLEM,
                      ', '.join(lines))
            name_note = ("All names follow the protocol" if res['ok_names'] else
                         f"missing: {len(res['missing'])}, unexpected: "
                         f"{len(res['unexpected'])}")
            s.set_row(mtp.SHEET_HANDOFF, 6,
                      STATUS_DONE if res['ok_names'] else STATUS_PROBLEM, name_note)
            for m in res['missing'][:20]:
                self._log_msg('warning', f"missing {m}")
            for u in res['unexpected'][:20]:
                self._log_msg('warning', f"unexpected {u}")
            self._verify_label.setText(
                ("Image counts OK. " if res['ok_counts'] else "Image counts NOT OK. ")
                + ("Names OK." if res['ok_names'] else "Names NOT OK (see log)."))
            self._verify_label.setStyleSheet(
                f"color: {COLORS['success'] if res['ok_counts'] and res['ok_names'] else COLORS['warning']};")
        s.save()
        try:
            s.write_metadata()
            s.write_data_sheet()
            path = s.write_checklist()
            self._log_msg('ok', f"Checklist saved: {path}")
        except PermissionError:
            self._log_msg('error', "Checklist not saved: close it in Excel and "
                                   "click 'Verify files and save checklist'")
        except Exception as e:  # noqa: BLE001
            self._log_msg('error', f"Checklist not saved: {e}")
        self._refresh_all_status()

    def shutdown(self):
        """Stop a running test (app exit)."""
        if self._is_running():
            self._worker.abort()
            self._worker.wait(10000)
        self._detach_camera()


def _html(text: str) -> str:
    return (text.replace('&', '&amp;').replace('<', '&lt;')
            .replace('>', '&gt;'))
