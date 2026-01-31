"""
Serial protocol handler for robust G-code communication.

This module solves the timeout errors by implementing:
- Thread-safe command queuing
- Separate send/receive threads
- Proper timeout handling with retry logic
- ACK/NAK protocol for G-code commands
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Any
import serial
import serial.tools.list_ports

from raman_control.config.logging_config import get_logger

logger = get_logger('serial_protocol')


class CommandStatus(Enum):
    """Status of a G-code command."""
    PENDING = auto()
    SENT = auto()
    ACKNOWLEDGED = auto()
    TIMEOUT = auto()
    ERROR = auto()


@dataclass
class GCodeCommand:
    """Represents a G-code command with metadata."""
    command: str
    callback: Optional[Callable[[bool, str], None]] = None
    timeout_ms: int = 5000
    retries: int = 3
    status: CommandStatus = CommandStatus.PENDING
    response: str = ""
    event: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self):
        # Ensure command ends with newline
        if not self.command.endswith('\n'):
            self.command = self.command.rstrip('\r\n') + '\n'


class SerialProtocol:
    """
    Thread-safe serial communication protocol for PRUSA 3D printer.

    This class handles all serial communication with the printer in a
    non-blocking manner using dedicated worker threads for sending
    and receiving data.

    Key features:
    - All commands are queued and processed serially
    - Automatic timeout handling with configurable retries
    - ACK detection for G-code commands
    - Thread-safe design - safe to call from any thread

    Usage:
        protocol = SerialProtocol(port='/dev/ttyACM0')
        protocol.start()

        # Non-blocking command dispatch
        protocol.send_command('G28 X', callback=on_homing_complete)

        # Blocking command with result
        success, response = protocol.send_command_sync('M114', timeout_ms=3000)

        protocol.stop()
    """

    # Known ACK patterns from PRUSA/Marlin firmware
    ACK_PATTERNS = ['ok', 'echo:', 'start']
    ERROR_PATTERNS = ['error', 'Error', 'Unknown command']

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        timeout: float = 5.0,
        write_timeout: float = 5.0,
        command_delay: float = 0.05,
        max_retries: int = 3
    ):
        """
        Initialize the serial protocol handler.

        Args:
            port: Serial port name (auto-detected if None)
            baudrate: Baud rate for serial communication
            timeout: Read timeout in seconds
            write_timeout: Write timeout in seconds
            command_delay: Delay between commands in seconds
            max_retries: Maximum retry attempts for failed commands
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.write_timeout = write_timeout
        self.command_delay = command_delay
        self.max_retries = max_retries

        # Serial port instance
        self._serial: Optional[serial.Serial] = None

        # Thread-safe command queue
        self._command_queue: queue.Queue[GCodeCommand] = queue.Queue()

        # Response queue for received data
        self._response_queue: queue.Queue[str] = queue.Queue()

        # Worker threads
        self._send_thread: Optional[threading.Thread] = None
        self._receive_thread: Optional[threading.Thread] = None

        # Control flags
        self._running = False
        self._lock = threading.Lock()
        self._current_command: Optional[GCodeCommand] = None

        # Connection status
        self._connected = False

        # Callbacks for status updates
        self._on_connected: Optional[Callable[[], None]] = None
        self._on_disconnected: Optional[Callable[[], None]] = None
        self._on_response: Optional[Callable[[str], None]] = None

    @property
    def is_connected(self) -> bool:
        """Check if serial port is connected and open."""
        return self._connected and self._serial is not None and self._serial.is_open

    @property
    def is_running(self) -> bool:
        """Check if worker threads are running."""
        return self._running

    def find_prusa_port(self) -> Optional[str]:
        """
        Auto-detect the PRUSA printer serial port.

        Returns:
            Port name if found, None otherwise
        """
        ports = serial.tools.list_ports.comports()
        for port in ports:
            # Match by description or VID/PID
            desc = port.description.lower()
            if 'prusa' in desc or 'marlin' in desc:
                logger.info(f"Found PRUSA printer on {port.device}: {port.description}")
                return port.device
            # PRUSA MK3/MK4 USB VID/PID
            if port.vid == 0x2c99:  # Prusa Research
                logger.info(f"Found PRUSA printer by VID on {port.device}")
                return port.device

        logger.warning("PRUSA printer not found")
        return None

    def connect(self) -> bool:
        """
        Connect to the serial port.

        Returns:
            True if connection successful, False otherwise
        """
        with self._lock:
            if self._connected:
                logger.warning("Already connected")
                return True

            # Auto-detect port if not specified
            port = self.port or self.find_prusa_port()
            if not port:
                logger.error("No serial port specified and auto-detection failed")
                return False

            try:
                self._serial = serial.Serial(
                    port=port,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                    write_timeout=self.write_timeout
                )

                # Clear buffers
                self._serial.reset_input_buffer()
                self._serial.reset_output_buffer()

                # Wait for printer to initialize
                time.sleep(2.0)

                # Flush startup messages
                startup_response = self._serial.read_all().decode('utf-8', errors='replace')
                logger.debug(f"Startup response: {startup_response}")

                self._connected = True
                logger.info(f"Connected to {port} at {self.baudrate} baud")

                if self._on_connected:
                    self._on_connected()

                return True

            except serial.SerialException as e:
                logger.error(f"Failed to connect to {port}: {e}")
                self._serial = None
                return False

    def disconnect(self) -> None:
        """Disconnect from the serial port."""
        with self._lock:
            if self._serial and self._serial.is_open:
                self._serial.close()
                logger.info("Serial port closed")

            self._serial = None
            self._connected = False

            if self._on_disconnected:
                self._on_disconnected()

    def start(self) -> bool:
        """
        Start the worker threads for send/receive.

        Returns:
            True if started successfully, False otherwise
        """
        if not self.is_connected:
            if not self.connect():
                return False

        if self._running:
            logger.warning("Worker threads already running")
            return True

        self._running = True

        # Start receive thread first
        self._receive_thread = threading.Thread(
            target=self._receive_worker,
            name="SerialReceive",
            daemon=True
        )
        self._receive_thread.start()

        # Start send thread
        self._send_thread = threading.Thread(
            target=self._send_worker,
            name="SerialSend",
            daemon=True
        )
        self._send_thread.start()

        logger.info("Worker threads started")
        return True

    def stop(self) -> None:
        """Stop the worker threads and disconnect."""
        self._running = False

        # Wake up any blocking calls
        try:
            self._command_queue.put(None)  # Sentinel to wake up send thread
        except Exception:
            pass

        # Wait for threads to finish
        if self._send_thread and self._send_thread.is_alive():
            self._send_thread.join(timeout=2.0)

        if self._receive_thread and self._receive_thread.is_alive():
            self._receive_thread.join(timeout=2.0)

        self.disconnect()
        logger.info("Serial protocol stopped")

    def send_command(
        self,
        command: str,
        callback: Optional[Callable[[bool, str], None]] = None,
        timeout_ms: int = 5000,
        retries: int = None
    ) -> None:
        """
        Queue a command for execution (non-blocking).

        Args:
            command: G-code command to send
            callback: Function called with (success, response) when complete
            timeout_ms: Timeout in milliseconds for ACK
            retries: Number of retry attempts (uses default if None)
        """
        if retries is None:
            retries = self.max_retries

        cmd = GCodeCommand(
            command=command,
            callback=callback,
            timeout_ms=timeout_ms,
            retries=retries
        )

        self._command_queue.put(cmd)
        logger.debug(f"Queued command: {command.strip()}")

    def send_command_sync(
        self,
        command: str,
        timeout_ms: int = 5000,
        retries: int = None
    ) -> tuple[bool, str]:
        """
        Send a command and wait for result (blocking).

        Args:
            command: G-code command to send
            timeout_ms: Timeout in milliseconds for ACK
            retries: Number of retry attempts

        Returns:
            Tuple of (success, response)
        """
        result = {'success': False, 'response': ''}
        event = threading.Event()

        def on_complete(success: bool, response: str):
            result['success'] = success
            result['response'] = response
            event.set()

        self.send_command(
            command=command,
            callback=on_complete,
            timeout_ms=timeout_ms,
            retries=retries
        )

        # Wait for completion with timeout
        wait_time = (timeout_ms / 1000.0) * ((retries or self.max_retries) + 1) + 1.0
        if event.wait(timeout=wait_time):
            return result['success'], result['response']
        else:
            logger.error(f"Sync command timed out: {command.strip()}")
            return False, "Timeout waiting for command completion"

    def _send_worker(self) -> None:
        """Worker thread that processes the command queue."""
        logger.info("Send worker started")

        while self._running:
            try:
                # Get next command from queue (blocks with timeout)
                cmd = self._command_queue.get(timeout=0.1)

                if cmd is None:  # Sentinel for shutdown
                    continue

                self._current_command = cmd
                success = self._send_with_retry(cmd)
                self._current_command = None

                # Inter-command delay to prevent buffer overflow
                time.sleep(self.command_delay)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Send worker error: {e}")

        logger.info("Send worker stopped")

    def _send_with_retry(self, cmd: GCodeCommand) -> bool:
        """
        Send a command with retry logic.

        Args:
            cmd: GCodeCommand to send

        Returns:
            True if command acknowledged, False otherwise
        """
        for attempt in range(cmd.retries + 1):
            if not self._running:
                return False

            try:
                # Send command
                with self._lock:
                    if not self._serial or not self._serial.is_open:
                        logger.error("Serial port not open")
                        return False

                    self._serial.write(cmd.command.encode('utf-8'))
                    self._serial.flush()

                cmd.status = CommandStatus.SENT
                logger.debug(f"Sent (attempt {attempt + 1}): {cmd.command.strip()}")

                # Wait for acknowledgment
                deadline = time.time() + (cmd.timeout_ms / 1000.0)
                response_lines = []

                while time.time() < deadline:
                    try:
                        response = self._response_queue.get(timeout=0.1)
                        response_lines.append(response)

                        if self._is_ack(response):
                            cmd.status = CommandStatus.ACKNOWLEDGED
                            cmd.response = '\n'.join(response_lines)

                            if cmd.callback:
                                cmd.callback(True, cmd.response)

                            logger.debug(f"ACK received for: {cmd.command.strip()}")
                            return True

                        if self._is_error(response):
                            cmd.status = CommandStatus.ERROR
                            cmd.response = '\n'.join(response_lines)
                            logger.warning(f"Error response: {response}")
                            break

                    except queue.Empty:
                        continue

                # Timeout or error - retry
                if attempt < cmd.retries:
                    logger.warning(
                        f"Retry {attempt + 1}/{cmd.retries} for: {cmd.command.strip()}"
                    )
                    time.sleep(0.5)  # Wait before retry

            except serial.SerialException as e:
                logger.error(f"Serial error: {e}")
                self._handle_serial_error()

        # All retries exhausted
        cmd.status = CommandStatus.TIMEOUT
        cmd.response = "Timeout after all retries"

        if cmd.callback:
            cmd.callback(False, cmd.response)

        logger.error(f"Command failed after {cmd.retries + 1} attempts: {cmd.command.strip()}")
        return False

    def _receive_worker(self) -> None:
        """Worker thread that reads data from serial port."""
        logger.info("Receive worker started")

        while self._running:
            try:
                if not self._serial or not self._serial.is_open:
                    time.sleep(0.1)
                    continue

                # Read line with timeout (handled by serial port timeout)
                if self._serial.in_waiting > 0:
                    line = self._serial.readline().decode('utf-8', errors='replace').strip()

                    if line:
                        logger.debug(f"Received: {line}")
                        self._response_queue.put(line)

                        if self._on_response:
                            self._on_response(line)
                else:
                    time.sleep(0.01)  # Small delay to prevent busy-waiting

            except serial.SerialException as e:
                logger.error(f"Receive error: {e}")
                self._handle_serial_error()
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Receive worker error: {e}")
                time.sleep(0.1)

        logger.info("Receive worker stopped")

    def _is_ack(self, response: str) -> bool:
        """Check if response is an acknowledgment."""
        response_lower = response.lower()
        return any(pattern in response_lower for pattern in self.ACK_PATTERNS)

    def _is_error(self, response: str) -> bool:
        """Check if response is an error."""
        return any(pattern in response for pattern in self.ERROR_PATTERNS)

    def _handle_serial_error(self) -> None:
        """Handle serial communication errors with recovery."""
        logger.warning("Attempting serial recovery...")

        with self._lock:
            if self._serial:
                try:
                    self._serial.reset_input_buffer()
                    self._serial.reset_output_buffer()
                except Exception:
                    pass

                # Try to reconnect
                port = self._serial.port
                try:
                    self._serial.close()
                except Exception:
                    pass

                time.sleep(1.0)

                try:
                    self._serial = serial.Serial(
                        port=port,
                        baudrate=self.baudrate,
                        timeout=self.timeout,
                        write_timeout=self.write_timeout
                    )
                    logger.info("Serial connection recovered")
                except Exception as e:
                    logger.error(f"Serial recovery failed: {e}")
                    self._connected = False

    def get_queue_size(self) -> int:
        """Get the number of pending commands in queue."""
        return self._command_queue.qsize()

    def clear_queue(self) -> None:
        """Clear all pending commands from queue."""
        while not self._command_queue.empty():
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("Command queue cleared")

    def set_callbacks(
        self,
        on_connected: Optional[Callable[[], None]] = None,
        on_disconnected: Optional[Callable[[], None]] = None,
        on_response: Optional[Callable[[str], None]] = None
    ) -> None:
        """Set callback functions for status updates."""
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        self._on_response = on_response
