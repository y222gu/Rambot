"""
Hardware configuration for the Raman control system.

Contains default settings for all hardware components.
"""

from dataclasses import dataclass, field
from typing import Optional


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
    safety_height: float = 30.0  # mm

    # Home position
    home_x: float = 100.0
    home_y: float = 100.0
    home_z: float = 30.0

    # Calibration offsets (widefield to objective)
    calib_wide_to_obj_x: float = 0.0
    calib_wide_to_obj_y: float = 0.0
    calib_wide_to_obj_z: float = 0.0

    # Stabilization time after movement (seconds)
    movement_stabilization_time: float = 0.1

    # Keep-alive interval (seconds)
    keepalive_interval: float = 5.0


@dataclass
class NanoDriveConfig:
    """Configuration for the MadCity Labs nano piezo."""
    initial_position: float = 50.0  # um
    step_size: float = 1.0  # um
    min_position: float = 0.0  # um
    max_position: float = 100.0  # um
    stabilization_time: float = 0.05  # seconds


@dataclass
class CameraConfig:
    """Configuration for Thorlabs cameras."""
    exposure_time: float = 0.01  # seconds
    gain: float = 1.0
    frame_rate: float = 30.0  # Hz


@dataclass
class SpectrometerConfig:
    """Configuration for Wasatch spectrometer."""
    integration_time: float = 100.0  # ms
    laser_power: float = 50.0  # percent
    averaging: int = 1


@dataclass
class SampleDropConfig:
    """Configuration for sample drop grid layout."""
    circle_radius: int = 15  # pixels
    spacing: int = 40  # pixels
    grid_pattern: tuple = (4, 4, 4, 4, 4)  # circles per column

    # Physical dimensions (mm)
    interval_between_drop_x: float = 5.0
    interval_between_drop_y: float = 5.0
    offset_home_to_p1_x: float = 10.0
    offset_home_to_p1_y: float = 10.0
    interval_between_slides_x: float = 25.0
    interval_between_slides_y: float = 0.0


@dataclass
class HardwareConfig:
    """Master configuration containing all hardware settings."""
    stage: StageConfig = field(default_factory=StageConfig)
    nanodrive: NanoDriveConfig = field(default_factory=NanoDriveConfig)
    widefield_camera: CameraConfig = field(default_factory=CameraConfig)
    objective_camera: CameraConfig = field(default_factory=CameraConfig)
    spectrometer: SpectrometerConfig = field(default_factory=SpectrometerConfig)
    sample_drop: SampleDropConfig = field(default_factory=SampleDropConfig)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'HardwareConfig':
        """Load configuration from a JSON/YAML file."""
        # TODO: Implement file loading
        return cls()

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to a JSON/YAML file."""
        # TODO: Implement file saving
        pass
