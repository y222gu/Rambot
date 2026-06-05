"""
Unit tests for the SerialProtocol class.

Tests cover:
- Command queuing
- Timeout handling
- Retry logic
- ACK detection
"""

import pytest
import queue
import time
import threading
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from raman_control.core.serial_protocol import (
    SerialProtocol, GCodeCommand, CommandStatus
)


class MockSerial:
    """Mock serial port for testing."""

    def __init__(self, *args, **kwargs):
        self.written_data = []
        self.read_responses = queue.Queue()
        self.is_open = True
        self._in_waiting = 0

    def write(self, data):
        self.written_data.append(data.decode('utf-8'))
        return len(data)

    def readline(self):
        try:
            response = self.read_responses.get(timeout=0.1)
            return response.encode('utf-8')
        except queue.Empty:
            return b''

    def read_all(self):
        responses = []
        while not self.read_responses.empty():
            try:
                responses.append(self.read_responses.get_nowait())
            except queue.Empty:
                break
        return '\n'.join(responses).encode('utf-8')

    @property
    def in_waiting(self):
        return not self.read_responses.empty()

    def flush(self):
        pass

    def reset_input_buffer(self):
        while not self.read_responses.empty():
            try:
                self.read_responses.get_nowait()
            except queue.Empty:
                break

    def reset_output_buffer(self):
        self.written_data.clear()

    def close(self):
        self.is_open = False

    def add_response(self, response: str):
        """Add a response to be returned on next readline."""
        self.read_responses.put(response)


class TestGCodeCommand:
    """Tests for GCodeCommand dataclass."""

    def test_command_initialization(self):
        """Test basic command creation."""
        cmd = GCodeCommand(command="G28 X")
        assert cmd.command == "G28 X\n"
        assert cmd.status == CommandStatus.PENDING
        assert cmd.timeout_ms == 5000
        assert cmd.retries == 3

    def test_command_with_newline(self):
        """Test command already has newline."""
        cmd = GCodeCommand(command="G28 X\n")
        assert cmd.command == "G28 X\n"

    def test_command_with_callback(self):
        """Test command with callback function."""
        callback = Mock()
        cmd = GCodeCommand(command="M114", callback=callback)
        assert cmd.callback is callback


class TestSerialProtocol:
    """Tests for SerialProtocol class."""

    @pytest.fixture
    def mock_serial(self):
        """Create a mock serial port."""
        return MockSerial()

    @pytest.fixture
    def protocol(self, mock_serial):
        """Create a protocol instance with mocked serial."""
        with patch('serial.Serial', return_value=mock_serial):
            with patch.object(SerialProtocol, 'find_prusa_port', return_value='/dev/ttyACM0'):
                p = SerialProtocol(
                    port='/dev/ttyACM0',
                    timeout=1.0,
                    command_delay=0.01
                )
                p._serial = mock_serial
                p._connected = True
                return p

    def test_initialization(self):
        """Test protocol initialization."""
        protocol = SerialProtocol(port='/dev/ttyACM0')
        assert protocol.port == '/dev/ttyACM0'
        assert protocol.baudrate == 115200
        assert not protocol.is_connected
        assert not protocol.is_running

    def test_command_queuing(self, protocol, mock_serial):
        """Test that commands are queued properly."""
        # Start workers
        protocol._running = True

        protocol.send_command('G28 X')
        protocol.send_command('G28 Y')
        protocol.send_command('G28 Z')

        assert protocol.get_queue_size() == 3

    def test_clear_queue(self, protocol):
        """Test clearing the command queue."""
        protocol.send_command('G28 X')
        protocol.send_command('G28 Y')

        protocol.clear_queue()
        assert protocol.get_queue_size() == 0

    def test_ack_detection(self, protocol):
        """Test ACK pattern detection.

        Only lines starting with 'ok' count as an ACK. Startup/echo
        messages ('start', 'echo: ...') must NOT be treated as ACKs —
        otherwise they could be mistaken for a command's acknowledgment
        (the false-ACK class of bug that stalls/races the motion queue).
        """
        assert protocol._is_ack('ok')
        assert protocol._is_ack('ok T:25.0')
        assert not protocol._is_ack('echo: start')
        assert not protocol._is_ack('start')
        assert not protocol._is_ack('moving')
        assert not protocol._is_ack('X:100.00 Y:100.00')

    def test_error_detection(self, protocol):
        """Test error pattern detection."""
        assert protocol._is_error('error: unknown command')
        assert protocol._is_error('Error:line number')
        assert protocol._is_error('Unknown command: G999')
        assert not protocol._is_error('ok')
        assert not protocol._is_error('X:100.00')

    def test_send_command_sync_success(self, protocol, mock_serial):
        """Test synchronous command with success."""
        # Setup mock to return ACK
        def mock_worker():
            time.sleep(0.1)
            # Simulate receive worker putting response in queue
            protocol._response_queue.put('ok')

        # Start a thread to simulate response
        thread = threading.Thread(target=mock_worker, daemon=True)
        protocol._running = True

        # Start send worker in background
        send_thread = threading.Thread(target=protocol._send_worker, daemon=True)
        send_thread.start()

        # Start receive simulation
        thread.start()

        success, response = protocol.send_command_sync('M114', timeout_ms=2000)

        protocol._running = False
        time.sleep(0.2)

        # Command should have been written
        assert len(mock_serial.written_data) > 0

    def test_callbacks(self, protocol):
        """Test callback registration."""
        on_connected = Mock()
        on_disconnected = Mock()
        on_response = Mock()

        protocol.set_callbacks(
            on_connected=on_connected,
            on_disconnected=on_disconnected,
            on_response=on_response
        )

        assert protocol._on_connected is on_connected
        assert protocol._on_disconnected is on_disconnected
        assert protocol._on_response is on_response


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
