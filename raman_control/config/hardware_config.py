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
    safety_height: float = 25.0  # mm
    # Relative Z raise performed BEFORE homing, to lift the probe away
    # from the sample/stage so that homing X/Y (moving toward 0) cannot
    # drag the probe across the sample. Done with soft endstops disabled
    # (M211 S0) since Z is not yet referenced. See StepperStage.home().
    pre_home_z_lift: float = 20.0  # mm
    # Robotic Deposition (capture-upfront flow): when the objective moves to a
    # droplet on a different slide, Z is first raised to the objective
    # autofocus "Highest" + this clearance so it clears the barrier between
    # slides, then lowered back to "Highest" once over the new slide.
    slide_change_z_clearance: float = 2.0  # mm

    # Home position
    home_x: float = 117.24
    home_y: float = 63.24
    home_z: float = 21.3

    # Calibration offsets (widefield to objective) in mm
    calib_wide_to_obj_x: float = -62.96
    calib_wide_to_obj_y: float = 0.9
    calib_wide_to_obj_z: float = -12.95

    # When True, the wide->objective Y offset is taken from the linear
    # calibration curve fitted to config/offset_calibration.csv, evaluated at
    # the current WIDEFIELD X position (the offset drifts linearly along X).
    # When False, the fixed ``calib_wide_to_obj_y`` above is used everywhere,
    # exactly as before. Falls back to the fixed value if the CSV is missing
    # or unusable. See config/offset_calibration.py.
    use_offset_calibration_curve: bool = True

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
    # Usable travel cap (um). Moves are clamped to this range, the autofocus
    # NanoDrive limit spin boxes are bounded by it, and it is the default
    # NanoDrive autofocus/z-stack range (lower=min, upper=max).
    min_position: float = 0.0    # um
    max_position: float = 100.0  # um
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

    # Camera-autofocus stabilization (seconds). Each Prusa Z step during the
    # autofocus scan shakes the stage; wait this long after every step for the
    # vibration to settle before scoring the (continuously acquired) camera
    # frame, so a blurry mid-vibration frame is never scored. Raise this if the
    # stage vibrates a lot. af_capture_stabilization_time is applied ONCE after
    # the best focus is reached, before grabbing the frame used for
    # segmentation, and is slightly longer so that capture is extra steady.
    af_settle_time: float = 1.0
    af_capture_stabilization_time: float = 1.5


@dataclass
class KymeraConfig:
    """Andor Kymera 328i spectrograph + Newton DU920P CCD (the alternative
    spectrometer). Values mirror the lab's automation notebooks."""
    grating: int = 3                      # turret position (1-based)
    center_wavelength_nm: float = 853.0   # grating centre wavelength
    cooler_on: bool = True
    target_temperature_c: float = -60.0   # CCD set point
    # Keep the set point when the camera is shut down (Andor cooler mode 1),
    # so a disconnect never leaves the sensor warming uncontrolled.
    keep_cooler_on_shutdown: bool = True
    # Extra time allowed for one exposure beyond 2x the exposure itself before
    # it is declared hung and aborted (readout + transfer margin).
    acquisition_timeout_margin_s: float = 5.0
    # Optional folder holding atmcd64d.dll / atspectrograph.dll. Empty = use
    # the DLLs bundled with the pyAndorSDK2 / pyAndorSpectrograph packages,
    # then C:\Program Files\Andor SDK.
    sdk_dir: str = ""


@dataclass
class SpectrometerConfig:
    """Configuration for the Raman spectrometer and excitation laser.

    ``backend`` selects the spectrometer: 'wasatch' (Wasatch Photonics unit),
    'kymera' (Andor Kymera + Newton), or 'auto' (Wasatch if detected, else
    Kymera). The excitation laser is always the one on the Wasatch unit; when
    the Kymera is the active spectrometer the Wasatch is opened for laser
    control only.
    """
    backend: str = 'auto'  # 'auto' | 'wasatch' | 'kymera'
    integration_time: float = 5000.0  # ms
    laser_power: float = 50.0  # percent
    laser_power_mw: float = 450.0  # mW (default laser power)
    averaging: int = 1
    excitation_wavelength_nm: float = 785.0  # laser line, for cm⁻¹ ↔ nm
    kymera: KymeraConfig = field(default_factory=KymeraConfig)

    # Spectrometer autofocus focus metric: 'ratio' (signal band / reference
    # band, the Wasatch default below), 'entropy' (Shannon entropy of the
    # entropy band, the lab's Kymera notebooks), or 'auto' (ratio on the
    # Wasatch, entropy on the Kymera). Both are MINIMISED.
    af_metric: str = 'auto'  # 'auto' | 'ratio' | 'entropy'
    af_entropy_min_nm: float = 830.0
    af_entropy_max_nm: float = 920.0

    # Spectrometer autofocus focus score: the summed intensity of the SIGNAL
    # band divided by that of the REFERENCE band. The scan minimises it, so
    # best focus is where the signal is weakest relative to the reference.
    # Defaults are noodlepy's: quartz peak (808-815 nm) over the reflected
    # laser line (790-795 nm). Bounds are inclusive, in nanometres, and both
    # bands must land on real pixels or every position scores "worst".
    #
    # The coarse (Prusa) and fine (NanoDrive) scans have their own bands: the
    # coarse scan sweeps a wide Z range far from focus, where a broad band is
    # more forgiving, while the fine scan can key on a narrower feature. They
    # start identical, so leaving them alone reproduces a single shared band.
    af_prusa_signal_min_nm: float = 808.0
    af_prusa_signal_max_nm: float = 815.0
    af_prusa_reference_min_nm: float = 790.0
    af_prusa_reference_max_nm: float = 795.0
    af_nano_signal_min_nm: float = 948.0
    af_nano_signal_max_nm: float = 953.0
    af_nano_reference_min_nm: float = 851.0
    af_nano_reference_max_nm: float = 852.0

    def af_signal_band(self, stage: str = 'prusa') -> tuple:
        """(min_nm, max_nm) of the focus-score numerator band for a scan
        stage: 'prusa' (coarse) or 'nano' (fine)."""
        if stage == 'nano':
            return (self.af_nano_signal_min_nm, self.af_nano_signal_max_nm)
        return (self.af_prusa_signal_min_nm, self.af_prusa_signal_max_nm)

    def af_reference_band(self, stage: str = 'prusa') -> tuple:
        """(min_nm, max_nm) of the focus-score denominator band for a scan
        stage: 'prusa' (coarse) or 'nano' (fine)."""
        if stage == 'nano':
            return (self.af_nano_reference_min_nm, self.af_nano_reference_max_nm)
        return (self.af_prusa_reference_min_nm, self.af_prusa_reference_max_nm)

    def af_entropy_band(self) -> tuple:
        """(min_nm, max_nm) of the band the entropy focus metric is taken over."""
        return (self.af_entropy_min_nm, self.af_entropy_max_nm)

    def resolve_af_metric(self, backend_name: str = '') -> str:
        """The focus metric to use ('ratio' / 'entropy') given the spectrometer
        that is actually connected ('wasatch' / 'kymera')."""
        metric = (self.af_metric or 'auto').lower()
        if metric in ('ratio', 'entropy'):
            return metric
        return 'entropy' if backend_name == 'kymera' else 'ratio'


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
    default_brightness: int = 128  # 0-255 (~50%), the initial/startup brightness


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
        pixel_size_x_um=2.825,
        pixel_size_y_um=2.825,
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
        # Centered on the objective FOV, matching the widefield aspect ratio.
        # After flip+rotate(-90) the raw 1440x1080 sensor becomes 1080 wide x
        # 1440 tall. Widefield FOV is 1440x1080 = 4:3, so using the full 1080
        # width gives height = 1080 * 3/4 = 810, centered at (540, 720).
        crop_center_x=780,
        crop_center_y=1150,
        crop_width=1080,
        crop_height=810,
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
                # Handle second-level nesting (StageConfig.serial,
                # SpectrometerConfig.kymera)
                if key == 'stage' and 'serial' in value and isinstance(value['serial'], dict):
                    value = dict(value)
                    value['serial'] = SerialConfig(**value['serial'])
                if key == 'spectrometer' and isinstance(value.get('kymera'), dict):
                    value = dict(value)
                    value['kymera'] = KymeraConfig(**value['kymera'])
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
