# Raman Microscopy Robot Control Software - Claude Code Context

## Project Overview

Vision-guided, high-throughput Raman microscopy system for blood-based liquid biopsy. This software coordinates optical hardware, motion control, image-based feedback, and data acquisition into an automated workflow with a user-facing interface for diverse technical backgrounds.

**Key Challenge**: Prevent timeout errors during data acquisition by properly managing 3D printer stage communication, thread synchronization, and GUI responsiveness.

---

## UI/UX Design Guidelines

### Visual Design Preferences
- **Theme**: Dark theme (dark background, light text) - easier on eyes during long sessions
- **Color Palette**:
  - Background: Dark gray (#1e1e1e to #2d2d2d)
  - Accent: Cyan/teal for active elements, green for success, orange for warnings, red for errors
  - Text: Light gray (#e0e0e0) for readability
- **Typography**: Clean, monospace fonts for data displays; sans-serif for labels
- **Spacing**: Generous padding, clear visual hierarchy

### UX Principles
- **Real-time feedback**: Always show what the system is doing (status indicators, progress bars)
- **Clear visual hierarchy**: Group related controls, use cards/panels with subtle borders
- **Responsive controls**: Buttons should have hover/pressed states
- **Data visualization**: Use PyQtGraph for real-time plots with dark theme
- **Accessibility**: High contrast, clear labels, keyboard shortcuts

### Widget Styling Guidelines
```python
# Example dark theme stylesheet
DARK_STYLESHEET = '''
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
}
QGroupBox {
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: bold;
}
QPushButton {
    background-color: #3d3d3d;
    border: 1px solid #4d4d4d;
    border-radius: 4px;
    padding: 8px 16px;
}
QPushButton:hover {
    background-color: #4d4d4d;
    border-color: #00a0a0;
}
QPushButton:pressed {
    background-color: #2d2d2d;
}
'''
```

---

## System Architecture

### Hardware Components

#### 1. Motion Control
- **3D Printer Stage (PRUSA-adapted)**
  - Control: G-code commands via serial port
  - Motors: X, Y, Z stepmotors
  - Step sizes: 60 μm (XY), 20 μm (Z)
  - **Critical**: Timeout errors originate here; requires robust serial communication
  - **Commands must be**:
    - Queued (never block main thread)
    - Acknowledged properly
    - Have explicit timeouts
    - Include error recovery

- **Nano Piezo (MadCity Lab)**
  - Z-axis only with 1 μm step size
  - Used for fine z-focusing of objective
  - Typically faster response than stepper motors
  - Can be controlled independently from stage

#### 2. Optical Hardware
- **Wide FOV Camera (Thorlabs)**
  - Purpose: Vision-guided sampling via image segmentation
  - Separate optical path from objective
  - Generates image for segmentation model
  - Continuous acquisition possible

- **Objective Camera (Thorlabs)**
  - Purpose: Live monitoring of sample under objective
  - Shares optical path with spectrometer via beam splitter
  - Part of sample feedback loop

- **Laser + Spectrometer (Wasatch Photonics)**
  - Collects Raman signal from objective
  - Controllable parameters: integration time, laser power
  - Real-time spectrum display needed
  - Critical for autofocus feedback

#### 3. Segmentation & Feedback
- External segmentation model (image → object detection)
- Image coordinates → physical coordinates conversion
- Closed-loop autofocus strategies:
  - Wide FOV autofocus (stepper + piezo feedback)
  - Spectrometer autofocus (piezo + spectrometer feedback)

---

## Software Design Principles

### 1. Threading Architecture (CRITICAL for GUI responsiveness)

**Multi-threaded design with clear responsibilities**:

```
Main GUI Thread
  ↓
  ├─ State Machine (sync coordinator)
  │
  ├─ Camera Thread 1: Wide FOV acquisition
  │   └─ Queue: images, metadata
  │
  ├─ Camera Thread 2: Objective FOV acquisition
  │   └─ Queue: images, metadata
  │
  ├─ Spectrometer Thread: Real-time spectrum acquisition
  │   └─ Queue: spectra, metadata
  │
  ├─ Motion Control Thread: 3D printer + piezo commands
  │   └─ Queue: movement commands (serialized!)
  │
  └─ Analysis/Callback Thread: Segmentation + feedback computation
      └─ Triggers motion commands based on feedback
```

**Thread Safety Rules**:
- Use `Queue.Queue()` (thread-safe) for all inter-thread communication
- Never access shared hardware state from multiple threads
- GUI updates from worker threads only via queues
- One thread "owns" each hardware interface
- State machine coordinates transitions

### 2. 3D Printer Communication (PREVENT TIMEOUTS)

**Problem Signature**: Timeout occurs when:
- Commands sent while printer is busy
- No proper handshake/acknowledgment
- Buffer overflow or command queue blocking
- Serial port state inconsistent

**Solution**:
1. **Dedicated Motion Thread**
   - Single queue for all movement commands
   - Never call stepper commands from GUI/analysis threads
   - Serialize all G-code with proper delays

2. **Robust Serial Protocol**
   - Send command → Wait for ACK → Timeout if no ACK → Error recovery
   - Typical pattern:
     ```python
     # Don't do this (blocks):
     serial_port.write(command)
     response = serial_port.readline()  # Blocks forever if no response!
     
     # Do this:
     serial_port.write(command)
     if wait_for_acknowledgment(timeout=5.0):  # Non-blocking with timeout
         process_response()
     else:
         handle_timeout_error()
     ```

3. **Command Queuing**
   - Queue all movement commands from different sources
   - Process serially with inter-command delays
   - Typical G-code needs 10-50ms between commands

4. **Heartbeat/Status Checking**
   - Periodic status queries to detect hung communication
   - Automatic recovery on timeout

### 3. GUI Framework

**Modern Look + Smooth Updates**:
- Replace old tkinter/PyQt4 with **PyQt6** or **PySimpleGUI** (modern styling)
- Or use **PySimpleGUI** with modern theme for fastest development
- For Qt: Use QThread for hardware threads, signals for safe GUI updates

**Key Pattern**:
```python
# Hardware thread emits signal/puts in queue
# GUI thread listens to queue and updates display
# Never block GUI with hardware waits

# Good:
class SpectrumUpdater(QThread):
    spectrum_ready = pyqtSignal(np.ndarray)
    def run(self):
        while True:
            spectrum = acquire_spectrum()  # Blocking, but in separate thread
            self.spectrum_ready.emit(spectrum)  # Emit signal to GUI

# Bad:
def update_spectrum():
    spectrum = acquire_spectrum()  # BLOCKS GUI!
    display_spectrum(spectrum)
```

---

## Recommended Library Updates

### Current Issues → Modern Replacements

| Aspect | Old | New | Reason |
|--------|-----|-----|--------|
| GUI Framework | tkinter/PyQt4 | PyQt6/PySide6 | Modern look, active development, better threading |
| Serial Communication | `pyserial` simple wrapper | `pyserial` + custom timeout handler | Better timeout handling |
| Threading | Basic threads | `threading` + `Queue` + QThread | Safer inter-thread communication |
| Image Processing | PIL/Pillow (v5) | OpenCV 4.x | Better integration, faster |
| Plotting | Matplotlib | Vispy or PyQtGraph | Real-time spectrum display, fast refresh |
| Data Storage | CSV/NumPy pickle | HDF5 (h5py) or Zarr | Metadata, compression, scalable |
| State Management | Ad-hoc flags | State machine (transitions lib) | Explicit, debuggable state transitions |
| Logging | print() statements | Python logging module | Structured, filterable output |

### Priority Order for Migration
1. **Logging** (minimal effort, huge debugging benefit)
2. **Threading + Queue** (solves timeout issues)
3. **GUI Framework** (PyQt6 or PySimpleGUI)
4. **Spectrum Display** (PyQtGraph for real-time)
5. **Data Storage** (HDF5 for metadata + compression)
6. **State Machine** (transitions library for clarity)

---

## Code Structure

### Recommended Organization

```
raman_control/
├── claude.md                          # This file
├── requirements.txt                   # Dependencies
├── config/
│   ├── __init__.py
│   ├── hardware_config.py            # Serial ports, device IDs, timeouts
│   └── acquisition_config.py         # Laser power, integration time defaults
├── hardware/
│   ├── __init__.py
│   ├── base_device.py                # Abstract hardware interface
│   ├── thorlabs_camera.py            # Wide FOV + Objective FOV
│   ├── wasatch_spectrometer.py       # Spectrometer interface
│   ├── stepper_stage.py              # 3D printer stage (CRITICAL)
│   └── nano_piezo.py                 # Nano piezo control
├── core/
│   ├── __init__.py
│   ├── state_machine.py              # FSM for workflow coordination
│   ├── motion_controller.py          # Queued movement + autofocus
│   ├── image_processor.py            # Segmentation + coordinate conversion
│   └── acquisition_engine.py         # Main data acquisition loop
├── gui/
│   ├── __init__.py
│   ├── main_window.py                # Main UI
│   ├── widgets/
│   │   ├── spectrum_viewer.py        # Real-time spectrum display
│   │   ├── camera_viewer.py          # Live camera feeds
│   │   └── status_panel.py           # System status
│   └── threads/
│       ├── camera_thread.py          # Camera acquisition thread
│       ├── spectrum_thread.py        # Spectrum acquisition thread
│       └── motion_thread.py          # Motion control thread
├── utils/
│   ├── __init__.py
│   ├── logging.py                    # Structured logging
│   ├── file_manager.py               # Data saving with metadata
│   ├── image_utils.py                # Image processing utilities
│   └── time_manager.py               # Timestamps, delays
├── tests/
│   ├── test_stepper_communication.py # CRITICAL: stress test
│   ├── test_threading.py
│   └── test_autofocus.py
└── main.py                           # Entry point
```

### Logging Best Practice

```python
# utils/logging.py

import logging
import logging.handlers

def setup_logging(log_file="raman_control.log"):
    """Configure structured logging"""
    
    logger = logging.getLogger('raman')
    logger.setLevel(logging.DEBUG)
    
    # File handler (detailed)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler (info only)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# Use in code:
logger = logging.getLogger('raman')
logger.info("Movement started: X=100, Y=200")
logger.debug(f"Serial response: {response}")
logger.error(f"Timeout on stepper: {error}")
```

---

## Autofocus Strategies

### Wide FOV Autofocus (Closed-Loop)
- Acquire frame → Detect focus quality → Adjust Z position (stepper/piezo)
- Common metrics: Laplacian variance, contrast, sharpness
- Loop until metric plateau

### Spectrometer Autofocus (Closed-Loop)
- Acquire spectrum → Fit peak intensity → Adjust Z position (piezo fine)
- Use intensity gradient to find optimum
- Faster than image-based due to fewer degrees of freedom


## Checklist for Implementation

- [ ] Serial communication with proper timeout handling
- [ ] Dedicated motion control thread with command queue
- [ ] GUI thread with PyQt6 + signal/slot pattern
- [ ] Camera acquisition thread (separate for each camera)
- [ ] Spectrum acquisition thread with real-time display
- [ ] State machine for workflow coordination
- [ ] Logging configured (file + console)
- [ ] HDF5 data saving with metadata
- [ ] Image-to-physical coordinate conversion
- [ ] Autofocus loops (wide FOV + spectrometer)
- [ ] Flexible acquisition protocol definition
- [ ] Unit tests for stepper communication
- [ ] Stress tests for threading
- [ ] Error recovery mechanisms
- [ ] User-friendly status messages

---

## Common Pitfalls to Avoid

1. **❌ Blocking GUI with hardware waits**
   - ✅ Use threads + signals, never `blocking_hardware_call()` in main thread

2. **❌ Accessing shared hardware from multiple threads**
   - ✅ One thread per hardware interface, use queues for communication

3. **❌ No timeout on serial reads**
   - ✅ Always `read_with_timeout()`, never indefinite `readline()`

4. **❌ Mixing G-code commands from different threads**
   - ✅ Central motion queue, process serially with delays

5. **❌ No error recovery after timeout**
   - ✅ Serial port recovery, command retry logic, state reset

6. **❌ Hardcoded paths/configurations**
   - ✅ Use config files, environment variables

7. **❌ No progress feedback during long operations**
   - ✅ Emit signals/logs, update GUI status bar frequently

8. **❌ Outdated imaging libraries (old OpenCV, old Qt)**
   - ✅ Use modern versions (OpenCV 4.x, PyQt6, Vispy/PyQtGraph)

---

## References & Resources

- **PyQt6 Threading**: https://doc.qt.io/qt-6/qthread.html
- **PySerial Timeout**: https://github.com/pyserial/pyserial/wiki
- **State Machines**: https://github.com/encode/python-transitions
- **Real-time Plotting**: https://www.vispy.io/, https://www.pyqtgraph.org/
- **HDF5 in Python**: https://docs.h5py.org/
- **G-code Reference**: https://marlin.3djake.de/ (Prusa uses Marlin)

---

## Questions for Claude Code Sessions

When starting a session, ask Claude Code to:

1. "Refactor stepper_stage.py with robust timeout handling and queue-based command processing"
2. "Create a PyQt6 main GUI with proper thread separation using signal/slot pattern"
3. "Implement spectrum viewer widget using PyQtGraph for real-time 100Hz display"
4. "Add comprehensive logging throughout the codebase following Python standards"
5. "Create state machine for acquisition workflow with automatic error recovery"
6. "Implement HDF5-based data storage with automatic metadata tracking"
7. "Add unit and stress tests for all hardware interfaces"
8. "Create flexible acquisition protocol system allowing easy workflow modifications"

---

**Last Updated**: January 2026
**Author**: Raman Microscopy Development Team
