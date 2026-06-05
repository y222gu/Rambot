"""
Pytest configuration for the raman_control test suite.

SAFETY: tests marked ``@pytest.mark.hardware`` talk to REAL hardware and
can physically move the stage (e.g. G28 homing). They are SKIPPED BY
DEFAULT and only run when you explicitly pass ``--run-hardware``.

    pytest raman_control/tests                 # safe: hardware tests skipped
    pytest raman_control/tests --run-hardware  # opt-in: WILL move the stage

This guarantees a normal ``pytest`` run never commands the printer.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-hardware",
        action="store_true",
        default=False,
        help="Run @pytest.mark.hardware tests that drive REAL hardware "
             "and can MOVE the stage. Off by default.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "hardware: test communicates with REAL hardware and may physically "
        "move the stage. Skipped unless --run-hardware is given.",
    )


def pytest_collection_modifyitems(config, items):
    """Skip hardware tests unless explicitly opted in.

    The skip marker is applied at collection time, so pytest skips the
    test BEFORE its fixtures run — meaning the real-hardware
    ``live_protocol`` fixture never even opens the serial port.
    """
    if config.getoption("--run-hardware"):
        return
    skip_hardware = pytest.mark.skip(
        reason="hardware test — skipped by default (pass --run-hardware to "
               "run; this MOVES the stage)"
    )
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip_hardware)
