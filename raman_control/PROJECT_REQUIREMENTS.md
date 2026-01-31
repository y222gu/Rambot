# Rambot Project Requirements & Implementation Guidelines

## Hardware Coordinate System

### Camera/Stage Offset Configuration
- **Wide FOV Camera** and **Objective Camera** have a physical offset in XYZ
- Objective camera shares optical path with spectrometer via beam splitter
- **Critical**: Z focal plane offset exists between wide cam and obj cam
- Stage movement required when switching camera views
- **DANGER**: Incorrect coordinate conversion when switching views can crash objective

### Coordinate Conversion Requirements
```python
# Example offset configuration (values TBD per hardware calibration)
CAMERA_OFFSETS = {
    'wide_to_obj': {
        'x': 0.0,      # XY should be same optical axis
        'y': 0.0,
        'z': -5.0,     # Example: Obj cam focal plane is 5mm lower
    }
}

# When switching from wide view to obj view:
# new_z = current_z + offset_z
# MUST verify Z is safe before moving!
```

---

## Feature Requirements

### 1. Sample Navigation Panel
**Status**: Partially implemented in protocol_panel.py

**Missing Features:**
- [ ] Autofocus upper Z limit entry (for wide camera PRUSA autofocus)
- [ ] Autofocus lower Z limit entry
- [ ] XYZ offset configuration display
- [ ] Safe Z position indicator

### 2. Live View (Single Camera Mode)
**Status**: Currently shows dual view - needs redesign

**Required Changes:**
- [ ] Show only one camera at a time
- [ ] Camera selector (Wide/Objective toggle)
- [ ] Stage moves when switching cameras
- [ ] Safe Z check before switching to objective view
- [ ] Replace exposure slider with text entry
- [ ] Add label/icon to crosshair toggle button

**Safety Protocol for View Switching:**
1. If switching TO objective view:
   - Check current Z position
   - Calculate target Z with offset
   - Verify Z is above safe minimum
   - Move stage to safe position first
   - Then switch camera view

### 3. Image Segmentation Module
**Status**: Not implemented

**Requirements:**
- [ ] SAM (Segment Anything Model) integration - vit_b checkpoint provided
- [ ] Traditional watershed segmentation option
- [ ] Segmentation parameter controls
- [ ] Sampling pattern generation from masks
- [ ] Position list creation from segmented objects

**Location**: `raman_control/segmentation/`
- `sam_segmenter.py` - SAM model wrapper
- `watershed_segmenter.py` - Traditional segmentation
- `pattern_generator.py` - Sampling pattern from masks

### 4. Autofocus Module
**Status**: Not implemented

**Requirements:**
- [ ] Coarse autofocus using PRUSA stepper (Z axis)
- [ ] Fine autofocus using NanoDrive piezo
- [ ] Focus metric options (contrast, spectrometer intensity)
- [ ] Upper/lower Z limit for autofocus range
- [ ] Step size configuration
- [ ] Autofocus result display

**Location**: `raman_control/core/autofocus.py`

### 5. Data Management
**Status**: Not implemented

**Requirements:**
- [ ] Data save location selector
- [ ] Filename template configuration
- [ ] Auto-naming with timestamp
- [ ] Metadata inclusion options
- [ ] HDF5 or other format options

---

## UI Improvements

### Spectrometer Widget
- [ ] Laser power in mW (e.g., 450mW) instead of percentage
- [ ] Add power presets if needed

### Live View Widget
- [ ] Add text label or icon to crosshair button ("✛" icon exists, add tooltip)
- [ ] Replace exposure slider with QSpinBox entry

### Protocol Panel
- [ ] Remove redundant "Select Row" and "Select Column" buttons
- [ ] Keep "Select All" and "Clear" buttons
- [ ] Implement drag-to-select in sample grid (rubber band selection)
- [ ] Add data save settings section

---

## Safety Considerations

### Stage Movement
1. **Never move to objective view without Z safety check**
2. **Always verify coordinates before large movements**
3. **Emergency stop should halt all motion immediately**
4. **Log all coordinate conversions for debugging**

### Autofocus
1. **Set hard limits on Z travel**
2. **Timeout on autofocus routines**
3. **Abort if focus metric doesn't improve**

---

## File Structure Update

```
raman_control/
├── segmentation/                    # NEW
│   ├── __init__.py
│   ├── sam_segmenter.py            # SAM model integration
│   ├── watershed_segmenter.py      # Traditional segmentation
│   └── pattern_generator.py        # Sampling pattern creation
├── core/
│   ├── autofocus.py                # NEW - Autofocus routines
│   ├── coordinate_manager.py       # NEW - Offset/conversion handling
│   └── ...
├── gui/widgets/
│   ├── live_view.py                # MODIFY - Single camera mode
│   ├── sample_grid.py              # MODIFY - Drag selection
│   ├── autofocus_panel.py          # NEW
│   ├── segmentation_panel.py       # NEW
│   └── data_settings.py            # NEW
└── models/
    └── sam_vit_b_01ec64.pth        # SAM model checkpoint
```

---

## Implementation Priority

1. **High Priority (Safety-Critical)**
   - Coordinate manager with offset handling
   - Single camera live view with safe switching
   - Autofocus Z limits

2. **Medium Priority (Core Functionality)**
   - Image segmentation module
   - Autofocus implementation
   - Data save settings

3. **Lower Priority (UI Polish)**
   - Drag-to-select in grid
   - UI label improvements
   - Exposure entry replacement

---

*Last Updated: 2026-01-30*
