"""
Abstract base class for hardware devices.

Provides a common interface for all hardware components.
"""

from abc import ABC, abstractmethod
from typing import Optional
from raman_control.config.logging_config import get_logger


class BaseDevice(ABC):
    """Abstract base class for hardware devices."""

    def __init__(self, name: str):
        """
        Initialize the device.

        Args:
            name: Human-readable device name for logging
        """
        self.name = name
        self.logger = get_logger(name)
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if device is connected."""
        return self._connected

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the device.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the device."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if device is ready for operation.

        Returns:
            True if device is ready, False otherwise
        """
        pass


class MotionDevice(BaseDevice):
    """Abstract base class for motion control devices."""

    @abstractmethod
    def get_position(self) -> tuple:
        """
        Get current position.

        Returns:
            Position tuple (format depends on device)
        """
        pass

    @abstractmethod
    def move_to(self, **kwargs) -> bool:
        """
        Move to absolute position.

        Returns:
            True if move initiated successfully
        """
        pass

    @abstractmethod
    def move_by(self, **kwargs) -> bool:
        """
        Move by relative distance.

        Returns:
            True if move initiated successfully
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Emergency stop motion."""
        pass

    @abstractmethod
    def home(self, axes: Optional[str] = None) -> bool:
        """
        Home specified axes.

        Args:
            axes: Which axes to home (e.g., 'X', 'XY', 'XYZ')

        Returns:
            True if homing initiated successfully
        """
        pass

    @abstractmethod
    def is_moving(self) -> bool:
        """
        Check if device is currently in motion.

        Returns:
            True if moving, False if stationary
        """
        pass
