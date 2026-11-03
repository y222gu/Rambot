"""
Dry-run tests (no hardware) for the Wasatch 2D readout ("area scan") and the
spectrometer switch-over:

* WasatchSpectrometer against a fake ``FeatureIdentificationDevice``: the
  capability probe (FPGA ``has_area_scan``, fixed vs settable vertical ROI),
  acquire_image() <-> acquire() switching, frame geometry / x inversion,
  giving up on a unit that streams no rows, bin rows -> set_vertical_roi
* SpectrumThread.request_reconnect(): spectra only ever come from the
  spectrometer selected in the config; an absent one ends with
  "No spectrometer detected"
* detector temperature of TEC-cooled Wasatch units (poll(), status, metadata,
  status-bar text), the same path the Kymera/Newton uses
* per-backend bin-row storage (WasatchConfig / KymeraConfig)
* SpectrumViewer placeholder / row locking (offscreen Qt)
"""

import os
import threading
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402

from raman_control.config.hardware_config import (  # noqa: E402
    HardwareConfig, SpectrometerConfig, WasatchConfig)
from raman_control.gui.threads.spectrum_thread import SpectrumThread  # noqa: E402
from raman_control.hardware import wasatch_spectrometer as ws_mod  # noqa: E402
from raman_control.hardware.wasatch_spectrometer import WasatchSpectrometer  # noqa: E402
from raman_control.tests.test_spectrometer_backends import (  # noqa: E402
    FakeBackend, FakeImageBackend, patch_candidates, run_thread, stop_thread)


# ---------------------------------------------------------------------------
# Fake wasatch driver objects
# ---------------------------------------------------------------------------

PIXELS = 1024
ROWS = 70          # driver frame buffer (actual_pixels_vertical)
STREAMED = 69      # rows a Hamamatsu unit actually streams


class Resp:
    """Stand-in for wasatch.SpectrometerResponse."""

    def __init__(self, data=None, error_msg='', poison_pill=False, keep_alive=False):
        self.data = data
        self.error_msg = error_msg
        self.poison_pill = poison_pill
        self.keep_alive = keep_alive


class SpectrumAndRow:
    def __init__(self, spectrum):
        self.spectrum = spectrum


class FakeEEPROM:
    def __init__(self):
        self.detector = "S7031-1006"
        self.serial_number = "WP-01000"
        self.model = "WP-785-C2-ER-LMM"
        self.invert_x_axis = False
        self.roi_vertical_region_1_start = 0
        self.roi_vertical_region_1_end = 63
        self.active_pixels_horizontal = PIXELS
        self.active_pixels_vertical = 64
        self.actual_pixels_vertical = ROWS
        # Detector TEC (the lab unit is cooled; the driver applies startup_temp_degC)
        self.has_cooling = True
        self.startup_temp_degC = -15
        self.min_temp_degC = -20
        self.max_temp_degC = 20


class FakeFPGAOptions:
    def __init__(self, has_area_scan=True):
        self.has_area_scan = has_area_scan


class FakeSettings:
    def __init__(self, has_area_scan=True, micro_fw="10.0.0.10", ingaas=False, xs=False):
        self.eeprom = FakeEEPROM()
        self.fpga_options = FakeFPGAOptions(has_area_scan)
        self.microcontroller_firmware_version = micro_fw
        self.fpga_firmware_version = "030-107"
        self.wavelengths = np.linspace(790.0, 1050.0, PIXELS)
        self.wavenumbers = 1e7 / 785.0 - 1e7 / self.wavelengths
        self._ingaas = ingaas
        self._xs = xs

    def is_ingaas(self):
        return self._ingaas

    def is_xs(self):
        return self._xs

    def is_arm(self):
        return False

    def pixels(self):
        return self.eeprom.active_pixels_horizontal

    def supports_feature(self, name):
        assert name == 'hamamatsu_vertical_roi'
        parts = [int(p) for p in self.microcontroller_firmware_version.split('.')]
        return parts >= [10, 0, 0, 47]


def usb_error(msg):
    import usb.core
    return usb.core.USBError(msg)


def row_value(r, c):
    """Pixel value of row r (0-based), column c in the fake frames: a row
    offset plus a column ramp, so row order and x orientation are checkable."""
    return (r + 1) * 100 + c // 16


class FakeUSBDevice:
    def __init__(self):
        self.halts = []

    def clear_halt(self, ep):
        self.halts.append(ep)


class FakeDeviceType:
    """RealUSBDevice stand-in: forwards bulk reads to the FakeFID."""

    def __init__(self, fid):
        self._fid = fid

    def read(self, device, ep, n, timeout=0):
        return self._fid.bulk_read(ep, n, timeout)


class FakeFID:
    """Records the calls the backend makes; models the area-scan protocol at
    the USB level: ACQUIRE (0xad) queues the rows the unit streams, bulk
    reads on 0x82 drain them, an empty pipe raises a libusb timeout.

    ``marker``: newer-firmware rows (0xffff + row index, sent in reverse
    order); default is the lab unit's format (plain rows, in order).
    ``chunked``: bulk reads return at most half a row at a time.
    ``fail_mode='pipe'``: every bulk read raises a non-timeout USB error.
    """

    def __init__(self, settings, streams_rows=True, refuse_area_scan=False,
                 marker=False, chunked=False, fail_mode=None, tec_enabled=True):
        self.settings = settings
        self.streams_rows = streams_rows
        self.refuse_area_scan = refuse_area_scan
        self.marker = marker
        self.chunked = chunked
        self.fail_mode = fail_mode
        self.area_scan_enabled = False
        self.area_scan_frame = [[0] * PIXELS for _ in range(ROWS)]
        self.extra_area_scan_data = [1, 2, 3]   # stale bytes to be dropped
        self.device = FakeUSBDevice()
        self.device_type = FakeDeviceType(self)
        self.pending = bytearray()
        self.acquires = 0
        self.timeouts = 0
        self.calls = []
        self.area_scan_toggles = []
        self.vertical_rois = []
        self.get_line_calls = 0
        self.next_spectra = []   # spectra get_line returns first (FIFO)
        # Detector TEC as the driver leaves it after its own connect.
        self.detector_tec_setpoint_has_been_set = True
        self.tec_setpoint_c = -15.0
        self.tec_enabled = tec_enabled
        self.temperature_c = 19.6
        self.temperature_reads = 0
        self.temperature_error = None

    def generate_timeout_ms(self):
        return 2 * 5 + 1000

    def _send_code(self, bRequest, wValue=0, wIndex=0, data_or_wLength=None, label=""):
        self.calls.append(('code', bRequest, label))
        if bRequest == 0xad:
            self.acquires += 1
            if self.area_scan_enabled and self.streams_rows:
                order = range(STREAMED - 1, -1, -1) if self.marker else range(STREAMED)
                for r in order:
                    row = [row_value(r, c) for c in range(PIXELS)]
                    if self.marker:
                        row[0], row[1] = 0xFFFF, r
                    self.pending.extend(np.array(row, dtype='<u2').tobytes())
        return Resp(keep_alive=True)

    def bulk_read(self, ep, n, timeout):
        assert ep == 0x82
        if self.fail_mode == 'pipe':
            raise usb_error("libusb0-dll:err [_usb_reap_async] reaping request "
                            "failed, win error: A device attached to the system "
                            "is not functioning.")
        if not self.pending:
            self.timeouts += 1
            raise usb_error("libusb0-dll:err [_usb_reap_async] timeout error")
        take = min(n, PIXELS if self.chunked else n)   # half a row when chunked
        chunk = bytes(self.pending[:take])
        del self.pending[:take]
        return chunk

    # --- settings / laser ---------------------------------------------------
    def set_integration_time_ms(self, ms):
        self.calls.append(('integration', ms))

    def set_laser_power_high_resolution(self, flag):
        self.calls.append(('laser_hr', flag))

    def set_laser_power_mW(self, mw):
        self.calls.append(('laser_mw', mw))

    def set_laser_enable(self, on):
        self.calls.append(('laser', on))

    # --- detector TEC -------------------------------------------------------
    def get_detector_temperature_degC(self):
        self.temperature_reads += 1
        if self.temperature_error is not None:
            raise self.temperature_error
        return Resp(data=self.temperature_c)

    def get_detector_tec_setpoint_degC(self):
        return Resp(data=self.tec_setpoint_c)

    def get_tec_enabled(self):
        return Resp(data=self.tec_enabled)

    # --- spectrum -----------------------------------------------------------
    def get_line(self):
        assert not self.area_scan_enabled, "spectrum read while streaming rows"
        self.get_line_calls += 1
        if self.next_spectra:
            return Resp(data=SpectrumAndRow(self.next_spectra.pop(0)))
        return Resp(data=SpectrumAndRow([100 + i % 7 for i in range(PIXELS)]))

    # --- area scan ----------------------------------------------------------
    def set_area_scan_enable(self, flag):
        self.area_scan_toggles.append(bool(flag))
        if self.refuse_area_scan and flag:
            return Resp(error_msg="area scan is not supported on InGaAs detectors")
        self.area_scan_enabled = bool(flag)
        return Resp(keep_alive=True)

    def reset_area_scan_frame(self):
        self.area_scan_frame = [[0] * PIXELS for _ in range(ROWS)]

    def get_area_scan(self):
        raise AssertionError("the backend must not use the driver's parser "
                             "(it drops rows without the 0xffff marker)")

    def set_vertical_roi(self, roi):
        self.vertical_rois.append(tuple(roi))
        self.settings.eeprom.roi_vertical_region_1_start = roi[0]
        self.settings.eeprom.roi_vertical_region_1_end = roi[1]
        return Resp(data=True)


class FakeDevice:
    def __init__(self, settings):
        self.settings = settings
        self.disconnected = False

    def disconnect(self):
        self.disconnected = True


def make_wasatch(monkeypatch, settings=None, cfg=None, **fid_kw):
    settings = settings or FakeSettings()
    fid = FakeFID(settings, **fid_kw)
    dev = FakeDevice(settings)
    monkeypatch.setattr(ws_mod, 'open_wasatch_device',
                        lambda log=None, quiet=False: (dev, fid))
    cfg = cfg or SpectrometerConfig(backend='wasatch', integration_time=5)
    backend = WasatchSpectrometer(cfg)
    assert backend.connect()
    return backend, fid, dev


# ---------------------------------------------------------------------------
# WasatchSpectrometer: probe
# ---------------------------------------------------------------------------

def test_wasatch_probe_reports_2d_sensor_with_fixed_rows(monkeypatch):
    backend, fid, _ = make_wasatch(monkeypatch)
    assert backend.supports_image_readout
    assert backend.detector_rows == ROWS
    # EEPROM vertical ROI (0, 63) 0-based inclusive -> rows 1-64.
    assert backend.bin_rows == (1, 64)
    # Microcontroller 10.0.0.10 < 10.0.0.47: the ROI cannot be changed.
    assert not backend.supports_bin_rows
    assert backend.model == "Wasatch Photonics WP-785-C2-ER-LMM"
    st = backend.status()
    assert st.serial == "WP-01000" and st.pixels == PIXELS
    assert st.details["Detector"] == "S7031-1006"
    assert st.details["Area scan"] == "supported"
    assert st.details["Bin rows"].startswith("1-64 of 70")
    assert "fixed by firmware" in st.details["Bin rows"]
    meta = backend.metadata()
    assert meta["Detector"] == "S7031-1006" and "Firmware" in meta
    # Nothing touched the area scan just by connecting.
    assert fid.area_scan_toggles == []


def test_wasatch_without_area_scan_is_a_1d_spectrometer(monkeypatch):
    backend, fid, _ = make_wasatch(monkeypatch, settings=FakeSettings(has_area_scan=False))
    assert not backend.supports_image_readout and not backend.supports_bin_rows
    assert backend.acquire_image() is None
    assert fid.area_scan_toggles == []
    assert backend.status().details["Area scan"] == "not supported"
    # Spectra still work as before.
    spec = backend.acquire()
    assert spec is not None and spec.shape == (PIXELS,)


def test_wasatch_ingaas_never_area_scans(monkeypatch):
    backend, fid, _ = make_wasatch(monkeypatch, settings=FakeSettings(ingaas=True))
    assert not backend.supports_image_readout
    assert backend.acquire_image() is None and fid.area_scan_toggles == []


# ---------------------------------------------------------------------------
# WasatchSpectrometer: image <-> spectrum readout
# ---------------------------------------------------------------------------

def test_wasatch_acquire_image_streams_rows_then_spectrum_restores(monkeypatch):
    backend, fid, _ = make_wasatch(monkeypatch)
    frame = backend.acquire_image()
    assert frame is not None and frame.dtype == np.float32
    # One stale-row probe (timeout: pipe was clean), one ACQUIRE, every
    # streamed row read in arrival order (no marker / index on this firmware),
    # then one drain timeout ends the frame.
    assert fid.acquires == 1 and fid.timeouts == 2 and not fid.pending
    assert frame.shape == (STREAMED, PIXELS)
    assert np.allclose(frame[:, 0], np.arange(1, STREAMED + 1) * 100)
    assert np.allclose(frame[3, :], [row_value(3, c) for c in range(PIXELS)])
    assert fid.area_scan_toggles == [True]
    assert fid.extra_area_scan_data == []      # stale bytes dropped
    assert backend.detector_rows == STREAMED   # follows the real frame
    assert backend.bin_rows == (1, 64)

    # A second frame does not re-enable the area scan.
    assert backend.acquire_image().shape == (STREAMED, PIXELS)
    assert fid.area_scan_toggles == [True] and fid.acquires == 2

    # Back to spectra: the area scan is switched off BEFORE get_line().
    spec = backend.acquire()
    assert spec is not None and spec.shape == (PIXELS,)
    assert fid.area_scan_toggles == [True, False]
    assert fid.get_line_calls == 1
    # ...and stays off for further spectra.
    backend.acquire()
    assert fid.area_scan_toggles == [True, False] and fid.get_line_calls == 2

    # Switching back and forth again.
    assert backend.acquire_image() is not None
    assert fid.area_scan_toggles == [True, False, True]


def test_wasatch_discards_leftover_area_scan_row_once(monkeypatch):
    backend, fid, _ = make_wasatch(monkeypatch)
    assert backend.acquire_image() is not None
    marker_row = [0xFFFF, 12] + [5] * (PIXELS - 2)
    fid.next_spectra = [marker_row]
    spec = backend.acquire()
    # The row still in the pipe was dropped and a real spectrum read instead.
    assert fid.get_line_calls == 2
    assert spec[0] != 0xFFFF
    # A saturated first pixel later on is NOT discarded (only right after
    # leaving the area scan).
    fid.next_spectra = [[0xFFFF] * PIXELS]
    spec = backend.acquire()
    assert fid.get_line_calls == 3 and spec[0] == 0xFFFF


def test_wasatch_image_follows_x_inversion(monkeypatch):
    settings = FakeSettings()
    settings.eeprom.invert_x_axis = True
    backend, fid, _ = make_wasatch(monkeypatch, settings=settings)
    frame = backend.acquire_image()
    # Column ramp reversed: the last column now holds the smallest value.
    assert frame[0, -1] < frame[0, 0]
    assert frame[0, -1] == row_value(0, 0) and frame[0, 0] == row_value(0, PIXELS - 1)


def test_wasatch_marked_rows_are_placed_by_index(monkeypatch):
    # Newer firmware: 0xffff marker + row index, rows sent in reverse order.
    backend, fid, _ = make_wasatch(monkeypatch, marker=True)
    frame = backend.acquire_image()
    assert frame.shape == (STREAMED, PIXELS)
    # Rows land at their index and the marker / index pixels are stomped.
    assert np.allclose(frame[:, 0], np.arange(1, STREAMED + 1) * 100)
    assert np.allclose(frame[:, 1], frame[:, 2])


def test_wasatch_reads_rows_from_partial_bulk_transfers(monkeypatch):
    backend, fid, _ = make_wasatch(monkeypatch, chunked=True)
    frame = backend.acquire_image()
    assert frame.shape == (STREAMED, PIXELS)
    assert np.allclose(frame[:, 0], np.arange(1, STREAMED + 1) * 100)


def test_wasatch_pipe_error_is_a_failed_frame_and_clears_the_endpoint(monkeypatch):
    backend, fid, _ = make_wasatch(monkeypatch, fail_mode='pipe')
    assert backend.acquire_image() is None
    assert fid.device.halts == [0x82]
    assert backend.supports_image_readout          # one failure only
    fid.fail_mode = None
    assert backend.acquire_image().shape == (STREAMED, PIXELS)


def test_wasatch_gives_up_on_unit_that_streams_no_rows(monkeypatch):
    backend, fid, _ = make_wasatch(monkeypatch, streams_rows=False)
    for _ in range(ws_mod.AREA_SCAN_MAX_FAILURES):
        assert backend.supports_image_readout
        assert backend.acquire_image() is None
    assert not backend.supports_image_readout
    # Area scan was switched off again so spectra keep working.
    assert fid.area_scan_toggles[-1] is False and not fid.area_scan_enabled
    assert backend.acquire_image() is None
    assert backend.acquire() is not None


def test_wasatch_refused_area_scan_enable(monkeypatch):
    backend, fid, _ = make_wasatch(monkeypatch, refuse_area_scan=True)
    assert backend.acquire_image() is None
    assert not backend.supports_image_readout
    assert fid.get_line_calls == 0


def test_wasatch_disconnect_leaves_area_scan(monkeypatch):
    backend, fid, dev = make_wasatch(monkeypatch)
    assert backend.acquire_image() is not None
    backend.disconnect()
    assert fid.area_scan_toggles == [True, False]
    assert ('laser', False) in fid.calls and dev.disconnected
    assert not backend.is_connected and backend.detector_rows == 0
    assert not backend.supports_image_readout


# ---------------------------------------------------------------------------
# WasatchSpectrometer: bin rows (vertical ROI)
# ---------------------------------------------------------------------------

def test_wasatch_bin_rows_fixed_by_old_firmware(monkeypatch):
    cfg = SpectrometerConfig(backend='wasatch', integration_time=5)
    backend, fid, _ = make_wasatch(monkeypatch, cfg=cfg)
    backend.set_bin_rows(10, 40)
    assert fid.vertical_rois == []           # firmware cannot take it
    assert backend.bin_rows == (1, 64)       # factory rows stay in force
    # The request is still recorded for a capable unit.
    assert (cfg.wasatch.bin_row_start, cfg.wasatch.bin_row_end) == (10, 40)


def test_wasatch_bin_rows_settable_on_new_firmware(monkeypatch):
    settings = FakeSettings(micro_fw="10.0.0.47")
    backend, fid, _ = make_wasatch(monkeypatch, settings=settings)
    assert backend.supports_bin_rows
    backend.set_bin_rows(3, 50)
    # 1-based inclusive rows -> 0-based inclusive driver lines.
    assert fid.vertical_rois == [(2, 49)]
    assert backend.bin_rows == (3, 50)
    backend.set_bin_rows(1, 0)               # 0 = last row
    assert fid.vertical_rois[-1] == (0, ROWS - 1)
    assert backend.bin_rows == (1, ROWS)
    backend.set_bin_rows(7, 7)               # single row: driver wants stop > start
    assert fid.vertical_rois[-1] == (6, 7)
    backend.set_bin_rows(60, 20)             # reversed -> ordered
    assert fid.vertical_rois[-1] == (19, 59)


def test_wasatch_configured_rows_applied_at_connect_when_possible(monkeypatch):
    cfg = SpectrometerConfig(backend='wasatch', integration_time=5,
                             wasatch=WasatchConfig(bin_row_start=5, bin_row_end=60))
    backend, fid, _ = make_wasatch(monkeypatch, cfg=cfg,
                                   settings=FakeSettings(micro_fw="10.0.0.50"))
    assert fid.vertical_rois == [(4, 59)] and backend.bin_rows == (5, 60)
    # Old firmware: configured rows are ignored (factory ROI reported).
    backend2, fid2, _ = make_wasatch(monkeypatch, cfg=cfg)
    assert fid2.vertical_rois == [] and backend2.bin_rows == (1, 64)



# ---------------------------------------------------------------------------
# WasatchSpectrometer: detector temperature (TEC-cooled units)
# ---------------------------------------------------------------------------

def test_wasatch_reads_detector_temperature_like_the_kymera(monkeypatch):
    backend, fid, _dev = make_wasatch(monkeypatch)
    # Read once at connect: warm detector, TEC on, set point from the driver.
    assert backend.has_cooling
    assert backend.temperature_c == pytest.approx(19.6)
    assert backend.tec_setpoint_c == -15.0
    assert backend.temperature_stable is False          # still cooling
    st = backend.status()
    assert st.detector_temperature_c == pytest.approx(19.6)
    assert st.temperature_stable is False
    assert st.details["Detector TEC"] == "set point -15 C"
    assert backend.metadata()["Detector temperature (C)"] == "19.6"

    # poll() is rate limited: a second call right away does not hit the USB...
    reads = fid.temperature_reads
    fid.temperature_c = -15.3
    backend.poll()
    assert fid.temperature_reads == reads
    assert backend.temperature_c == pytest.approx(19.6)

    # ...but after the poll interval it does, and the detector is now stable.
    backend._last_temp_poll -= 10.0
    backend.poll()
    assert fid.temperature_reads == reads + 1
    assert backend.temperature_c == pytest.approx(-15.3)
    assert backend.temperature_stable is True
    assert backend.status().temperature_stable is True

    backend.disconnect()
    assert backend.status().detector_temperature_c is None


def test_wasatch_uncooled_unit_has_no_temperature(monkeypatch):
    settings = FakeSettings()
    settings.eeprom.has_cooling = False
    backend, fid, _dev = make_wasatch(monkeypatch, settings=settings)
    assert not backend.has_cooling
    backend._last_temp_poll -= 10.0
    backend.poll()
    assert fid.temperature_reads == 0
    st = backend.status()
    assert st.detector_temperature_c is None
    assert st.temperature_stable is None
    assert "Detector TEC" not in st.details
    assert "Detector temperature (C)" not in backend.metadata()


def test_wasatch_temperature_read_failure_keeps_last_value(monkeypatch):
    backend, fid, _dev = make_wasatch(monkeypatch)
    fid.temperature_error = usb_error("libusb0-dll:err [control_msg] timeout")
    backend._last_temp_poll -= 10.0
    backend.poll()                        # must never raise on the spectrum thread
    assert backend.temperature_c == pytest.approx(19.6)
    fid.temperature_error = None
    fid.temperature_c = 5.0
    backend._last_temp_poll -= 10.0
    backend.poll()
    assert backend.temperature_c == pytest.approx(5.0)


def test_wasatch_tec_off_reports_no_cooling_state(monkeypatch):
    backend, fid, _dev = make_wasatch(monkeypatch, tec_enabled=False)
    assert backend.temperature_c == pytest.approx(19.6)
    assert backend.temperature_stable is None           # nothing is cooling
    assert backend.status().details["Detector TEC"] == "set point -15 C, off"


def test_status_bar_detail_text_for_cooled_and_uncooled_spectrometers():
    from raman_control.gui.main_window import spectrometer_detail_text
    from raman_control.hardware.spectrometer_base import SpectrometerStatus
    label = "Wasatch Photonics"
    assert spectrometer_detail_text(label, SpectrometerStatus()) == label
    cooling = SpectrometerStatus(detector_temperature_c=12.4, temperature_stable=False)
    assert spectrometer_detail_text(label, cooling) == "Wasatch Photonics 12°C (cooling)"
    stable = SpectrometerStatus(detector_temperature_c=-15.2, temperature_stable=True)
    assert spectrometer_detail_text(label, stable) == "Wasatch Photonics -15°C"
    unknown = SpectrometerStatus(detector_temperature_c=-15.2, temperature_stable=None)
    assert spectrometer_detail_text(label, unknown) == "Wasatch Photonics -15°C"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_round_trips_wasatch_rows_and_readout_for(tmp_path):
    cfg = HardwareConfig()
    assert isinstance(cfg.spectrometer.wasatch, WasatchConfig)
    assert (cfg.spectrometer.wasatch.bin_row_start, cfg.spectrometer.wasatch.bin_row_end) == (0, 0)
    assert cfg.spectrometer.readout_for('wasatch') is cfg.spectrometer.wasatch
    assert cfg.spectrometer.readout_for('kymera') is cfg.spectrometer.kymera
    assert cfg.spectrometer.readout_for('') is cfg.spectrometer.kymera
    cfg.spectrometer.wasatch.bin_row_start = 3
    cfg.spectrometer.wasatch.bin_row_end = 40
    path = tmp_path / "hw.json"
    cfg.save_to_file(str(path))
    loaded = HardwareConfig.load_from_file(str(path))
    assert isinstance(loaded.spectrometer.wasatch, WasatchConfig)
    assert (loaded.spectrometer.wasatch.bin_row_start,
            loaded.spectrometer.wasatch.bin_row_end) == (3, 40)
    loaded.update_from_dict({'spectrometer.wasatch.bin_row_end': 0})
    assert loaded.spectrometer.wasatch.bin_row_end == 0


# ---------------------------------------------------------------------------
# SpectrumThread: switching the spectrometer, per-backend bin rows
# ---------------------------------------------------------------------------

class FakeFixedRowsBackend(FakeImageBackend):
    """2D detector whose rows cannot be changed (old Wasatch firmware)."""

    @property
    def supports_bin_rows(self):
        return False


def _wait(pred, timeout=2.0):
    deadline = time.time() + timeout
    while not pred() and time.time() < deadline:
        time.sleep(0.01)
    return pred()


def test_thread_reconnect_switches_to_selected_backend(monkeypatch):
    cfg = SpectrometerConfig(backend='wasatch', integration_time=1)
    spec = SpectrumThread(cfg)
    wasatch = FakeBackend('wasatch', has_laser=True)
    kymera = FakeBackend('kymera')
    # Candidates follow the config, as the real _candidate_backends() does.
    monkeypatch.setattr(spec, '_candidate_backends',
                        lambda: [wasatch] if cfg.backend == 'wasatch' else [kymera])
    events = []
    spec.connected.connect(lambda: events.append('connected'),
                           Qt.ConnectionType.DirectConnection)
    spec.disconnected.connect(lambda: events.append('disconnected'),
                              Qt.ConnectionType.DirectConnection)
    spectra = []
    spec.spectrum_ready.connect(lambda w, i, ts: spectra.append(i),
                                Qt.ConnectionType.DirectConnection)
    t = run_thread(spec)
    try:
        assert spec.backend_name == 'wasatch'
        spec.start_acquisition()
        assert _wait(lambda: wasatch.acquired >= 2)
        cfg.backend = 'kymera'
        spec.request_reconnect()
        assert _wait(lambda: spec.backend_name == 'kymera' and spec.is_connected)
        assert events == ['connected', 'disconnected', 'connected']
        assert not wasatch.is_connected and kymera.is_connected
        # Acquisition keeps running on the new unit and the old one is idle.
        n = wasatch.acquired
        assert _wait(lambda: kymera.acquired >= 2)
        time.sleep(0.05)
        assert wasatch.acquired == n
    finally:
        stop_thread(spec, t)


def test_thread_reconnect_to_absent_backend_shows_no_spectrometer(monkeypatch):
    cfg = SpectrometerConfig(backend='wasatch', integration_time=1)
    spec = SpectrumThread(cfg)
    wasatch = FakeBackend('wasatch', has_laser=True)
    kymera = FakeBackend('kymera', available=False)
    monkeypatch.setattr(spec, '_candidate_backends',
                        lambda: [wasatch] if cfg.backend == 'wasatch' else [kymera])
    events, errors, spectra = [], [], []
    spec.connected.connect(lambda: events.append('connected'),
                           Qt.ConnectionType.DirectConnection)
    spec.disconnected.connect(lambda: events.append('disconnected'),
                              Qt.ConnectionType.DirectConnection)
    spec.error.connect(lambda m: errors.append(m), Qt.ConnectionType.DirectConnection)
    spec.spectrum_ready.connect(lambda w, i, ts: spectra.append(i),
                                Qt.ConnectionType.DirectConnection)
    t = run_thread(spec)
    try:
        spec.start_acquisition()
        assert _wait(lambda: len(spectra) >= 2)
        cfg.backend = 'kymera'
        spec.request_reconnect()
        assert _wait(lambda: errors and errors[-1].startswith("No spectrometer detected"))
        assert not spec.is_connected and not wasatch.is_connected
        assert events == ['connected', 'disconnected']
        # No Wasatch spectrum sneaks through after the switch...
        n = len(spectra)
        time.sleep(0.1)
        assert len(spectra) == n
        # ...but the thread stays alive, idle, so the user can switch back.
        assert t.is_alive() and spec._running
        cfg.backend = 'wasatch'
        spec.request_reconnect()
        assert _wait(lambda: spec.is_connected and spec.backend_name == 'wasatch')
        assert events == ['connected', 'disconnected', 'connected']
        assert _wait(lambda: len(spectra) > n)   # live view resumes
        # A retry towards a still-absent unit keeps idling (no crash, no exit).
        cfg.backend = 'kymera'
        spec.request_reconnect()
        assert _wait(lambda: len(errors) >= 2)
        assert t.is_alive() and not spec.is_connected
        assert events == ['connected', 'disconnected', 'connected', 'disconnected']
    finally:
        stop_thread(spec, t)
        assert not t.is_alive()


def test_thread_retry_from_idle_connects_when_unit_appears(monkeypatch):
    spec = SpectrumThread(SpectrometerConfig(backend='kymera', integration_time=1))
    kymera = FakeBackend('kymera', available=False)
    patch_candidates(monkeypatch, spec, [kymera])
    events, errors = [], []
    spec.connected.connect(lambda: events.append('connected'),
                           Qt.ConnectionType.DirectConnection)
    spec.disconnected.connect(lambda: events.append('disconnected'),
                              Qt.ConnectionType.DirectConnection)
    spec.error.connect(lambda m: errors.append(m), Qt.ConnectionType.DirectConnection)
    t = threading.Thread(target=spec.run, daemon=True)
    t.start()
    try:
        assert _wait(lambda: len(errors) == 1)
        assert errors[0].startswith("No spectrometer detected")
        assert events == ['disconnected'] and t.is_alive() and not spec.is_connected
        # Connect pressed again while the unit is still absent: idles on.
        spec.request_reconnect()
        assert _wait(lambda: len(errors) == 2)
        assert events == ['disconnected'] and t.is_alive()
        # The unit appears and Connect (or Settings OK) retries: connected.
        kymera._available = True
        spec.request_reconnect()
        assert _wait(lambda: spec.is_connected)
        assert events == ['disconnected', 'connected']
    finally:
        stop_thread(spec, t)
    assert events == ['disconnected', 'connected', 'disconnected']


def test_thread_initial_connect_failure_idles_and_stops_cleanly(monkeypatch):
    spec = SpectrumThread(SpectrometerConfig(backend='kymera', integration_time=1))
    patch_candidates(monkeypatch, spec, [FakeBackend('kymera', available=False)])
    events, errors = [], []
    spec.disconnected.connect(lambda: events.append('disconnected'),
                              Qt.ConnectionType.DirectConnection)
    spec.error.connect(lambda m: errors.append(m), Qt.ConnectionType.DirectConnection)
    th = threading.Thread(target=spec.run, daemon=True)
    th.start()
    assert _wait(lambda: len(errors) == 1)
    assert events == ['disconnected'] and errors[0].startswith("No spectrometer detected")
    time.sleep(0.2)
    assert th.is_alive()                       # idle, not dead
    stop_thread(spec, th)
    assert not th.is_alive()
    assert events == ['disconnected']          # not announced twice


def test_thread_stores_bin_rows_in_the_connected_backends_config(monkeypatch):
    cfg = SpectrometerConfig(backend='wasatch', integration_time=1)
    spec = SpectrumThread(cfg)
    backend = FakeFixedRowsBackend('wasatch', rows=70, pixels=16)
    patch_candidates(monkeypatch, spec, [backend])
    t = run_thread(spec)
    try:
        assert spec.supports_image_readout and not spec.supports_bin_rows
        spec.set_bin_rows(2, 30)
        assert (cfg.wasatch.bin_row_start, cfg.wasatch.bin_row_end) == (2, 30)
        assert (cfg.kymera.bin_row_start, cfg.kymera.bin_row_end) == (1, 0)
        assert _wait(lambda: backend.bin_rows == (2, 30))
        cfg.wasatch.bin_row_start = 5
        spec.apply_readout_config()
        assert _wait(lambda: backend.bin_rows == (5, 30))
    finally:
        stop_thread(spec, t)


# ---------------------------------------------------------------------------
# SpectrumViewer: placeholder and row locking (offscreen Qt)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


class StubThread:
    """Just enough of SpectrumThread for the viewer's slots."""

    def __init__(self, image=True, rows=69, settable=False):
        self.is_connected = True
        self.laser_available = True
        self.backend_label = "Wasatch Photonics"
        self.backend_name = "wasatch"
        self.supports_image_readout = image
        self.supports_bin_rows = settable
        self.detector_rows = rows
        self.bin_rows = (1, 64)
        self.rows_set = []
        self.image_mode_set = []

    def start_acquisition(self):
        pass

    def set_bin_rows(self, s, e):
        self.rows_set.append((s, e))

    def set_image_mode(self, on):
        self.image_mode_set.append(on)


def test_viewer_placeholder_and_fixed_rows(qapp):
    from raman_control.gui.widgets.spectrum_viewer import SpectrumViewerWidget as SpectrumViewer
    cfg = SpectrometerConfig(backend='wasatch')
    viewer = SpectrumViewer(spectrum_thread=None, config=cfg)
    # Start-up: empty plot with the "not connected" text.
    assert viewer._placeholder.isVisible()
    assert viewer._placeholder.textItem.toPlainText() == "Spectrometer not connected"

    stub = StubThread()
    viewer._spectrum_thread = stub
    viewer._on_connected()
    assert not viewer._placeholder.isVisible()
    assert viewer._mode_box.isVisibleTo(viewer)
    # Rows fixed by firmware: boxes / lines locked, label shown, rows = ROI.
    assert not viewer._row_start_spin.isEnabled() and not viewer._rows_all_btn.isEnabled()
    assert not viewer._row_start_line.movable
    assert not viewer._rows_fixed_label.isHidden()
    assert (viewer._row_start_spin.value(), viewer._row_end_spin.value()) == (1, 64)
    assert viewer._row_start_spin.maximum() == 69

    # A spectrum shows; a disconnect empties the plot and says so.
    viewer._last_plot_update = 0.0
    viewer._on_spectrum_ready(np.arange(16.0), np.ones(16, dtype=np.float32), 0.0)
    assert len(viewer._spectrum_curve.yData) == 16
    stub.is_connected = False
    viewer._on_disconnected()
    assert viewer._placeholder.isVisible()
    assert viewer._placeholder.textItem.toPlainText() == "No spectrometer detected"
    assert viewer._spectrum_curve.yData is None or len(viewer._spectrum_curve.yData) == 0
    assert viewer._current_spectrum is None
    assert not viewer._mode_box.isVisibleTo(viewer)

    # A detector whose rows CAN be set unlocks the controls again.
    stub2 = StubThread(settable=True)
    viewer._spectrum_thread = stub2
    viewer._on_connected()
    assert viewer._row_start_spin.isEnabled() and viewer._row_start_line.movable
    assert viewer._rows_fixed_label.isHidden()
    viewer._apply_rows(3, 20)
    assert stub2.rows_set == [(3, 20)]
    viewer.deleteLater()
