"""
Hardware configuration for the Raman control system.

Contains default settings for all hardware components.
"""

import json
import os
from dataclasses import dataclass, field, fields, asdict


@dataclass
class SerialConfig:
    """Configuration for serial communication."""
    baudrate: int = 115200
    timeout: float = 5.0  # seconds
    write_timeout: float = 5.0  # seconds
    command_delay: float = 0.05  # seconds between commands
    max_retries: int = 3
    ack_timeout: float = 10.0  # seconds to wait for acknowledgment


@dataclass
class StageConfig:
    """Configuration for the PRUSA 3D printer stage."""
    # Serial settings
    serial: SerialConfig = field(default_factory=SerialConfig)

    # Step sizes (mm)
    small_step_xy: float = 0.06
    medium_step_xy: float = 0.6
    large_step_xy: float = 6.0
    small_step_z: float = 0.02
    medium_step_z: float = 0.2
    large_step_z: float = 2.0

    # Speed (mm/min)
    default_speed: int = 3000
    homing_speed: int = 600

    # Safety
    safety_height: float = 20.0  # mm
    # Relative Z raise performed BEFORE homing, to lift the probe away
    # from the sample/stage so that homing X/Y (moving toward 0) cannot
    # drag the probe across the sample. Done with soft endstops disabled
    # (M211 S0) since Z is not yet referenced. See StepperStage.home().
    pre_home_z_lift: float = 20.0  # mm

    # Home position
    home_x: float = 117.24
    home_y: float = 63.36
    home_z: float = 21.3

    # Calibration offsets (widefield to objective) in mm
    calib_wide_to_obj_x: float = -62.48
    calib_wide_to_obj_y: float = 1.32
    calib_wide_to_obj_z: float = -13.26

    # Stabilization time after movement (seconds)
    movement_stabilization_time: float = 0.1

    # Keep-alive interval (seconds)
    keepalive_interval: float = 5.0

    # Stage travel limits (mm)
    x_min: float = 0.0
    x_max: float = 250.0
    y_min: float = 0.0
    y_max: float = 210.0
    z_min: float = 0.0
    z_max: float = 200.0
    z_safe: float = 5.0  # Safe Z for camera switching

    # Autofocus search limits (mm)
    af_z_upper: float = 40.0
    af_z_lower: float = 30.0


@dataclass
class NanoDriveConfig:
    """Configuration for the MadCity Labs nano piezo."""
    # DLL path (relative to raman_control package root)
    dll_path: str = ""  # Set at runtime based on package location
    axis: int = 3  # Z-axis

    initial_position: float = 50.0  # um
    step_size: float = 1.0  # um
    min_position: float = 0.0  # um
    max_position: float = 90.0  # um (hardware safe max)
    stabilization_time: float = 0.025  # seconds (25 ms settling time)


@dataclass
class CameraConfig:
    """Configuration for Thorlabs cameras."""
    serial_number: str = ""  # Camera serial number for identification
    exposure_time: float = 0.003  # seconds (3 ms default)
    gain: float = 1.0
    frame_rate: float = 30.0  # Hz
    image_poll_timeout_ms: int = 0  # 0 = non-blocking poll

    # Pixel size in object plane (micrometers per pixel)
    pixel_size_x_um: float = 1.0
    pixel_size_y_um: float = 1.0

    # Image transformations
    flip_horizontal: bool = False
    rotate_minus90: bool = False
    crop_enabled: bool = False
    crop_center_x: int = 0
    crop_center_y: int = 0
    crop_width: int = 500
    crop_height: int = 375

    # Max re-center attempts for the "Center" button: after centering it
    # re-segments and repeats while the mask touches an image edge, up to
    # this many times (avoids an infinite loop).
    center_max_repeats: int = 3


@dataclass
class SpectrometerConfig:
    """Configuration for Wasatch spectrometer."""
    integration_time: float = 1000.0  # ms
    laser_power: float = 50.0  # percent
    laser_power_mw: float = 450.0  # mW (default laser power)
    averaging: int = 1
    excitation_wavelength_nm: float = 785.0  # laser line, for cm⁻¹ ↔ nm


@dataclass
class LEDConfig:
    """Configuration for the COB illumination LED strip (Arduino over serial).

    The Arduino firmware accepts a brightness value 0-255 as an ASCII line.
    ``port`` empty means auto-detect an Arduino-class board (CH340/FTDI/etc.);
    auto-detect never claims the Prusa stage port.
    """
    port: str = ""           # "" = auto-detect
    baudrate: int = 9600
    timeout: float = 2.0     # seconds (serial read timeout)
    boot_delay: float = 2.0  # seconds to wait after open (Arduino resets)
    default_brightness: int = 255  # 0-255, used when toggling on from 0


@dataclass
class SampleDropConfig:
    """Configuration for 384 well plate sample positions.

    A standard 384 well plate has 16 rows (A-P) x 24 columns (1-24)
    with a uniform 4.5 mm pitch in both directions.  All 384 positions
    exist on the plate at uniform spacing — "gaps" in valid wells are
    simply empty positions on the same grid.

    Well positions are defined RELATIVE TO THE HOME POSITION so that
    recalibrating home (the "Calibrate Home" button updates
    ``StageConfig.home_x/home_y``) moves the entire grid with it.

    Position of well at (row, col) in stage coordinates:
        x = home_x + offset_home_to_a1_x + row * well_pitch_mm
        y = home_y + offset_home_to_a1_y + col * well_pitch_mm

    where row/col are 0-based indices (A=0, B=1, ...; col1=0, col2=1, ...).
    The ROW letter steps the X axis and the COLUMN number steps the Y
    axis (e.g. B3->C3 moves +4.5 mm in X; B3->B4 moves +4.5 mm in Y).
    Distance from home grows with both the row letter and the column
    number. ``offset_home_to_a1_x/y`` is the offset from the home
    position to well A1.
    """
    # Standard 384 well plate pitch (mm) — uniform across entire plate
    well_pitch_mm: float = 4.5

    # Offset from the home position to well A1 (mm) — calibrate per setup.
    # Tuned so that well B3 (row=1, col=2) lands at (home + 12.5, home + 6.5),
    # with row stepping +X and column stepping +Y:
    #   x: 8.0 + 1*4.5 = 12.5,   y: -2.5 + 2*4.5 = 6.5
    offset_home_to_a1_x: float = 8.0
    offset_home_to_a1_y: float = -2.5


def _default_dll_dir() -> str:
    """Return the absolute path to the 64-bit DLL directory."""
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(pkg_root, 'dlls', '64_lib')


def _default_widefield_camera() -> CameraConfig:
    return CameraConfig(
        serial_number="14938",
        exposure_time=0.04,
        pixel_size_x_um=2.82,
        pixel_size_y_um=2.82,
    )


def _default_objective_camera() -> CameraConfig:
    return CameraConfig(
        serial_number="14628",
        exposure_time=0.04,
        pixel_size_x_um=0.191,
        pixel_size_y_um=0.191,
        flip_horizontal=True,
        rotate_minus90=True,
        crop_enabled=True,
        crop_center_x=798,
        crop_center_y=452,
        crop_width=500,
        crop_height=375,
    )


def _default_nanodrive() -> NanoDriveConfig:
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return NanoDriveConfig(
        dll_path=os.path.join(pkg_root, 'dlls', 'Madlib.dll'),
    )


@dataclass
class HardwareConfig:
    """Master configuration containing all hardware settings."""
    # Path to directory containing native DLLs (Thorlabs + MadCity Labs)
    dll_dir: str = field(default_factory=_default_dll_dir)

    stage: StageConfig = field(default_factory=StageConfig)
    nanodrive: NanoDriveConfig = field(default_factory=_default_nanodrive)
    widefield_camera: CameraConfig = field(default_factory=_default_widefield_camera)
    objective_camera: CameraConfig = field(default_factory=_default_objective_camera)
    spectrometer: SpectrometerConfig = field(default_factory=SpectrometerConfig)
    sample_drop: SampleDropConfig = field(default_factory=SampleDropConfig)
    led: LEDConfig = field(default_factory=LEDConfig)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'HardwareConfig':
        """Load configuration from a JSON file, reconstructing nested dataclasses."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> 'HardwareConfig':
        """Reconstruct a HardwareConfig from a plain dict."""
        nested_map = {
            'stage': StageConfig,
            'nanodrive': NanoDriveConfig,
            'widefield_camera': CameraConfig,
            'objective_camera': CameraConfig,
            'spectrometer': SpectrometerConfig,
            'sample_drop': SampleDropConfig,
            'led': LEDConfig,
        }
        kwargs = {}
        for key, value in data.items():
            if key in nested_map and isinstance(value, dict):
                dc_cls = nested_map[key]
                # Handle second-level nesting (StageConfig.serial)
                if key == 'stage' and 'serial' in value and isinstance(value['serial'], dict):
                    value = dict(value)
                    value['serial'] = SerialConfig(**value['serial'])
                # Convert tuple fields stored as lists back to tuples
                for f in fields(dc_cls):
                    if f.name in value and (f.type == 'tuple' or (hasattr(f.type, '__origin__') and f.type is tuple)):
                        value[f.name] = tuple(value[f.name])
                kwargs[key] = dc_cls(**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to a JSON file."""
        data = asdict(self)
        # Convert tuples to lists for JSON serialization (json.dumps handles this,
        # but being explicit helps with round-tripping)
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def update_from_dict(self, data: dict) -> None:
        """Update this config in-place from a flat dict of dotted keys.

        Keys use dot notation, e.g. ``'stage.home_x'`` sets
        ``self.stage.home_x``.
        """
        for dotted_key, value in data.items():
            parts = dotted_key.split('.')
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
