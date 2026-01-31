# Rambot Refactoring Memory File

**Purpose**: Track mistakes, solutions, and lessons learned during the tkinter → PyQt6 refactoring to prevent repeated errors and save tokens.

---

## Project Context
- **Goal**: Refactor Raman microscopy control software from ttkbootstrap/tkinter to PyQt6
- **Primary Problem**: Timeout errors in 3D printer stage communication
- **Root Cause**: Blocking serial reads + no command queuing in `stagecontrolmodule.py`

---

## Migration Status (ALL PHASES COMPLETE ✅)

**Total files created: 30**

### Config Layer
| File | Status | Notes |
|------|--------|-------|
| `config/logging_config.py` | ✅ Done | Structured logging with file rotation |
| `config/hardware_config.py` | ✅ Done | Dataclass-based configuration |

### Core Layer
| File | Status | Notes |
|------|--------|-------|
| `core/serial_protocol.py` | ✅ Done | **CRITICAL FIX** - Thread-safe command queuing |
| `core/state_machine.py` | ✅ Done | Transitions-based workflow management |

### Hardware Layer
| File | Status | Notes |
|------|--------|-------|
| `hardware/base_device.py` | ✅ Done | Abstract device interface |
| `hardware/stepper_stage.py` | ✅ Done | High-level stage API |
| `hardware/nano_piezo.py` | ✅ Done | MadCity Labs piezo control |

### GUI Threads
| File | Status | Notes |
|------|--------|-------|
| `gui/threads/motion_thread.py` | ✅ Done | QThread for stage control |
| `gui/threads/camera_thread.py` | ✅ Done | QThread for camera acquisition |
| `gui/threads/spectrum_thread.py` | ✅ Done | QThread for spectrometer |

### GUI Widgets (Dark Theme)
| File | Status | Notes |
|------|--------|-------|
| `gui/styles.py` | ✅ Done | Dark theme stylesheet |
| `gui/main_window.py` | ✅ Done | Application entry point |
| `gui/widgets/stage_control.py` | ✅ Done | Direction pads + position display |
| `gui/widgets/live_view.py` | ✅ Done | Dual camera display |
| `gui/widgets/spectrum_viewer.py` | ✅ Done | PyQtGraph real-time plot |
| `gui/widgets/sample_grid.py` | ✅ Done | Interactive position grid |
| `gui/widgets/protocol_panel.py` | ✅ Done | Acquisition workflow control |

### Utils & Tests
| File | Status | Notes |
|------|--------|-------|
| `utils/coordinate_utils.py` | ✅ Done | Image-to-stage conversion |
| `tests/test_serial_protocol.py` | ✅ Done | Unit + stress tests |

---

## TODO / Next Steps

### High Priority (Hardware Integration)
1. **Connect real PRUSA stage** - Test `serial_protocol.py` with actual hardware
2. **Thorlabs camera SDK** - Uncomment SDK calls in `camera_thread.py`, set `_simulated = False`
3. **Wasatch spectrometer SDK** - Uncomment SDK calls in `spectrum_thread.py`, set `_simulated = False`
4. **MadCity Labs piezo** - Test `nano_piezo.py` with real hardware

### Medium Priority (Features)
5. **Segmentation UI Panel** - Create GUI widget for segmentation module
6. **Autofocus UI Panel** - Create GUI widget for autofocus controls
7. **Data export** - Implement HDF5 data saving with metadata
8. **Settings dialog** - Create settings/calibration dialogs for hardware configuration

### Lower Priority (Polish)
9. **Keyboard shortcuts** - Add more shortcuts for common actions
10. **Position calibration** - Image-to-physical coordinate calibration workflow
11. **Error recovery** - Implement automatic error recovery in state machine
12. **Unit tests** - Add tests for GUI widgets and state machine transitions

---

## Key Decisions Made

1. Target framework: PyQt6 (not PyQt4)
2. Primary refactoring priority: Serial communication timeout handling
3. Thread architecture: One thread per hardware device, Qt signals for GUI updates
4. Command queuing: All G-code via `Queue.Queue()` with dedicated worker threads

---

## Patterns That Work

### Serial Communication
```python
# SerialProtocol pattern (core/serial_protocol.py)
class SerialProtocol:
    def __init__(self):
        self._command_queue = queue.Queue()  # Thread-safe
        self._response_queue = queue.Queue()

    def send_command(self, command, callback=None, timeout_ms=5000):
        """Queue command (non-blocking, thread-safe)"""
        cmd = GCodeCommand(command, callback, timeout_ms)
        self._command_queue.put(cmd)

    def _send_worker(self):
        """Process queue one command at a time"""
        while self._running:
            cmd = self._command_queue.get(timeout=0.1)
            self._send_with_retry(cmd)
            time.sleep(0.05)  # Inter-command delay
```

### Threading
```python
# QThread pattern (gui/threads/motion_thread.py)
class MotionThread(QThread):
    position_changed = pyqtSignal(float, float, float)

    def run(self):
        self._stage = StepperStage(config)
        self._stage.set_callbacks(
            on_position_changed=lambda p: self.position_changed.emit(p.x, p.y, p.z)
        )
        # ...
```

### GUI Updates
```python
# Signal/slot pattern for thread-safe updates
class StageWidget(QWidget):
    def __init__(self, motion_thread):
        motion_thread.position_changed.connect(self._on_position_changed)

    def _on_position_changed(self, x, y, z):
        self.x_label.setText(f"{x:.2f}")  # Safe - runs in GUI thread
```

---

## Patterns to Avoid

### 1. Blocking Serial Reads (BAD)
```python
# DON'T DO THIS - blocks forever if no response
while True:
    line = self.ser.readline()  # BLOCKS!
    if 'ok' in line:
        break
```

### 2. Direct Hardware Access from GUI (BAD)
```python
# DON'T DO THIS - blocks GUI thread
def button_clicked(self):
    self.ser.write(b'G28\n')  # BLOCKS GUI!
    response = self.ser.readline()  # BLOCKS FOREVER!
```

### 3. Polling Position in Tight Loop (BAD)
```python
# DON'T DO THIS - causes timeouts
while True:
    current_x = float(self.get_current_prusa_position('X'))  # Blocks!
    if current_x == target_x:
        break
```

---

## File Mapping: Old → New

| Old File | New Location | Status |
|----------|--------------|--------|
| noodlepy/gui/stagecontrolmodule.py | raman_control/core/serial_protocol.py + hardware/stepper_stage.py + gui/threads/motion_thread.py + gui/widgets/stage_control.py | Done |
| noodlepy/gui/publisher_subscriber.py | Qt signals/slots (built-in) | Done |
| noodlepy/gui/liveviewmodule.py | gui/threads/camera_thread.py + gui/widgets/live_view.py | Done |
| noodlepy/gui/wasatch_autofocus_module.py | gui/threads/spectrum_thread.py + gui/widgets/spectrum_viewer.py | Done |
| noodlepy/gui/protocolmodule.py | core/state_machine.py + gui/widgets/protocol_panel.py | Done |
| noodlepy/gui/acquisitiontab.py | gui/main_window.py | Done |

---

## Quick Reference

### Run the New Application
```bash
cd /Users/yifeigu/Documents/Carney_Lab/Rambot
python -m raman_control.main
```

**Note**: Do NOT use `python raman_control/main.py` - this won't work due to absolute imports.

### Run Tests
```bash
# Unit tests (no hardware)
python -m pytest raman_control/tests/test_serial_protocol.py -v

# Hardware stress tests
python -m pytest raman_control/tests/test_serial_protocol.py -v -m hardware
```

### Install Dependencies
```bash
pip install -r raman_control/requirements.txt
```

---

## Known Issues & Fixes

### Issue 3: Poor UI Layout Decisions (FIXED)
**Mistake**: Created separate tabs for Segmentation and Autofocus when they should be integrated into existing panels.

**Problems**:
- Segmentation should be part of Live View (always uses camera capture, not file load)
- Autofocus belongs in left Stage Control panel (related to Z movement)
- Multiple progress bars in different tabs - should be ONE in status bar
- Workflow state as big button - should be simple boolean indicator
- Acquisition parameters stacked vertically - hard to read, wastes space

**Fix**:
- Merge Segmentation into Live View tab
- Merge Autofocus into Stage Control (left panel)
- Single progress bar in status bar for all processes
- Use horizontal layouts and group boxes for better space utilization
- Add sample pattern section (ring, grid, line, random) with parameters
- Plot detected areas AND sample points on captured image

**Lesson**: Think about workflow - what panels does user look at together? Group related controls. Use space efficiently.

### Issue 1: `is_ready` property conflict (FIXED)
**Error**: `AttributeError: property 'is_ready' of 'AcquisitionStateMachine' object has no setter`

**Cause**: The `transitions` library auto-creates `is_<state>()` methods for each state. Having a state named "ready" conflicts with our `is_ready` property.

**Fix**: Renamed property from `is_ready` to `ready_to_acquire` in `core/state_machine.py`.

### Issue 2: Module import error
**Error**: Running `python raman_control/main.py` fails with import errors

**Cause**: The package uses absolute imports (`from raman_control.xxx import yyy`)

**Fix**: Use `python -m raman_control.main` to run as a module.

---

## Session Notes

### Session 1 (2026-01-29)
- Created complete Phase 1-3 implementation
- SerialProtocol is the key fix for timeout errors
- All files follow thread-safe patterns
- GUI responsive during hardware operations

### Session 2 (2026-01-30)
- **COMPLETED ALL PHASES**
- Implemented dark theme with COLORS palette and comprehensive stylesheet
- Created camera_thread.py with simulated frame generation
- Created spectrum_thread.py with simulated Raman spectrum
- Built live_view.py with dual camera display
- Built spectrum_viewer.py with PyQtGraph real-time plotting
- Created sample_grid.py with interactive position selection
- Created state_machine.py using transitions library
- Built protocol_panel.py for acquisition workflow
- Implemented nano_piezo.py hardware wrapper
- Updated main_window.py to use all actual widgets
- Fixed `is_ready` property conflict in state machine
- Total: 30 Python files, complete PyQt6 migration

### Session 3 (2026-01-30 continued)
- **MAJOR FEATURE ADDITIONS**
- Added coordinate manager (`core/coordinate_manager.py`) for camera offset handling
- Added autofocus engine (`core/autofocus.py`) with coarse/fine focus support
- Created segmentation module with SAM and watershed support:
  - `segmentation/sam_segmenter.py` - SAM vit_b model wrapper
  - `segmentation/watershed_segmenter.py` - Traditional segmentation
  - `segmentation/pattern_generator.py` - Convert masks to stage positions
- Redesigned Live View to single camera mode with safe switching
- Updated sample grid with drag-to-select (rubber band selection)
- Spectrometer: Changed laser power from % to mW
- Protocol panel: Added autofocus Z limits, data save settings
- Protocol panel: Removed redundant row/column select buttons
- Live view: Replaced exposure slider with spinbox entry
- Live view: Added text label to crosshair button
- Fixed spectrometer plot lag (disabled antialiasing, mouse tracking throttle)
- Fixed stage control freeze when device not connected

### Session 4 (2026-01-30 continued)
- **GUI INTEGRATION COMPLETE**
- Added `get_latest_frame()` to CameraThread for on-demand image access
- Added `get_latest_spectrum()` to SpectrumThread for on-demand spectrum access
- Added `move_z()`, `move_to()`, `get_position()` convenience methods to MotionThread

### Session 5 (2026-01-30 - UI REDESIGN)
- **MAJOR UI CONSOLIDATION**
- Merged Segmentation into Live View tab:
  - Capture button for camera frame
  - SAM segmentation with center point prompt
  - Sample pattern generation (Grid, Ring, Line, Random)
  - Pattern parameters: rows/cols, rays/rings/interval/offset, points
  - Overlay shows mask (blue) and sample points (orange) on image
- Merged Autofocus into Stage Control panel:
  - Z limits (Upper/Lower) with "Set from Current" buttons
  - Coarse AF (PRUSA stepper) and Fine AF (NanoDrive) buttons
  - `autofocus_requested` signal emitted with type and Z limits
- Added single global progress bar in status bar:
  - `set_progress(value, text)` method for all async operations
  - Device indicators, position, progress bar all in one row
- Removed separate Segmentation and Autofocus tabs (redundant)
- Tabs now: Live View, Spectrometer, Sample Grid, Protocol
- Total lines of code reduced, better space utilization

### Hardware Integration Notes
- Camera and spectrometer threads use `_simulated = True` flag
- To connect real hardware, set `_simulated = False` and uncomment SDK calls
- Thorlabs SDK: `thorlabs_tsi_sdk`
- Wasatch SDK: `wasatch` package
- MadCity Labs: `madlib` library
- SAM model: `segmentation/sam_vit_b_01ec64.pth`

---

## Camera Offset Configuration

**CRITICAL SAFETY**: The wide FOV and objective cameras have a physical offset.
Switching cameras requires stage movement. Incorrect coordinates can crash objective.

```python
# Offset from wide FOV to objective camera (configure per hardware)
CAMERA_OFFSETS = {
    'wide_to_obj': {
        'x': 0.0,      # XY typically same optical axis
        'y': 0.0,
        'z': -5.0,     # Objective focal plane may be lower
    }
}
```

See `core/coordinate_manager.py` for safe coordinate conversion.

---

## Directory Structure

```
raman_control/
├── __init__.py
├── main.py                          # Entry point
├── requirements.txt
├── PROJECT_REQUIREMENTS.md          # Feature requirements doc
├── config/
│   ├── __init__.py
│   ├── logging_config.py           # Structured logging
│   └── hardware_config.py          # Dataclass configs
├── core/
│   ├── __init__.py
│   ├── serial_protocol.py          # CRITICAL - Thread-safe G-code
│   ├── state_machine.py            # Workflow coordination
│   ├── coordinate_manager.py       # Camera offset handling
│   └── autofocus.py                # Coarse/fine autofocus
├── hardware/
│   ├── __init__.py
│   ├── base_device.py              # Abstract interface
│   ├── stepper_stage.py            # PRUSA stage wrapper
│   └── nano_piezo.py               # MadCity Labs piezo
├── segmentation/                    # NEW - Image segmentation
│   ├── __init__.py
│   ├── sam_segmenter.py            # SAM model wrapper
│   ├── watershed_segmenter.py      # Traditional segmentation
│   ├── pattern_generator.py        # Mask to positions
│   └── sam_vit_b_01ec64.pth        # SAM model checkpoint
├── gui/
│   ├── __init__.py
│   ├── main_window.py              # Main application window
│   ├── styles.py                   # Dark theme stylesheet
│   ├── threads/
│   │   ├── __init__.py
│   │   ├── motion_thread.py        # Stage control thread
│   │   ├── camera_thread.py        # Camera acquisition thread
│   │   └── spectrum_thread.py      # Spectrometer thread
│   └── widgets/
│       ├── __init__.py
│       ├── stage_control.py        # Direction pads, position + autofocus
│       ├── live_view.py            # Camera display + segmentation + pattern
│       ├── spectrum_viewer.py      # PyQtGraph spectrum plot
│       ├── sample_grid.py          # Position grid + drag select
│       └── protocol_panel.py       # Acquisition + data settings
├── utils/
│   ├── __init__.py
│   └── coordinate_utils.py         # Coordinate transforms
└── tests/
    ├── __init__.py
    └── test_serial_protocol.py     # Serial communication tests
```

---

*Last Updated: 2026-01-30*
