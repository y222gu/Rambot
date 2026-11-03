"""
Dry-run tests for the spectrometer backend layer (no hardware).

Covers:
* backend selection and the 'auto' Wasatch -> Kymera fallback in SpectrumThread
* laser routing: Wasatch backend applies laser settings itself, the Kymera
  backend forwards them to the laser thread
* KymeraSpectrometer against fake ``pyAndorSDK2`` / ``pyAndorSpectrograph``
  modules: configuration sequence, ms -> s exposure, bounded wait + abort on
  timeout, calibration axis, temperature polling
* SpectrometerConfig focus-metric resolution and the entropy metric
* HardwareConfig JSON round trip of the nested KymeraConfig
"""

import sys
import threading
import time
import types

import numpy as np
import pytest
from PyQt6.QtCore import Qt

from raman_control.config.hardware_config import (
    HardwareConfig, KymeraConfig, SpectrometerConfig)
from raman_control.core.autofocus import AutofocusEngine
from raman_control.hardware.spectrometer_base import (
    SpectrometerBackend, SpectrometerStatus, wavelength_nm_to_wavenumber,
    wavenumber_to_wavelength_nm)
from raman_control.gui.threads import spectrum_thread as st_mod
from raman_control.gui.threads.spectrum_thread import SpectrumThread


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeBackend(SpectrometerBackend):
    """Minimal in-memory backend; ``available`` controls connect()."""
    name = "fake"
    model = "Fake"

    def __init__(self, name, available=True, has_laser=False, pixels=16):
        self.name = name
        self.model = f"Fake {name}"
        self._available = available
        self._has_laser = has_laser
        self._pixels = pixels
        self._connected = False
        self.integration = []
        self.laser_enable = []
        self.laser_power = []
        self.acquired = 0

    def connect(self):
        self._connected = self._available
        return self._available

    def disconnect(self):
        self._connected = False

    @property
    def is_connected(self):
        return self._connected

    @property
    def wavelengths_nm(self):
        return np.linspace(790, 950, self._pixels)

    @property
    def pixels(self):
        return self._pixels

    def set_integration_time_ms(self, ms):
        self.integration.append(ms)

    def acquire(self):
        self.acquired += 1
        return np.ones(self._pixels, dtype=np.float32)

    @property
    def has_laser(self):
        return self._has_laser

    @property
    def laser_available(self):
        return self._has_laser

    def set_laser_enable(self, on):
        self.laser_enable.append(on)
        return True

    def set_laser_power_mw(self, mw):
        self.laser_power.append(mw)
        return True


class FakeLaserThread:
    """Stands in for LaserThread: records forwarded commands."""

    def __init__(self):
        self.enable = []
        self.power = []
        self.started = False
        self.is_available = True
        self.available_changed = _Signal()
        self.laser_state_changed = _Signal()
        self.error = _Signal()

    def isRunning(self):
        return self.started

    def start(self):
        self.started = True

    def stop(self, wait_ms=0):
        self.started = False

    def request_enable(self, on):
        self.enable.append(on)

    def request_power_mw(self, mw):
        self.power.append(mw)


class _Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *a):
        for s in self.slots:
            s(*a)


def run_thread(spec, wait=2.0):
    """Run SpectrumThread.run() on a plain thread (no Qt event loop)."""
    t = threading.Thread(target=spec.run, daemon=True)
    t.start()
    deadline = time.time() + wait
    while not spec.is_connected and t.is_alive() and time.time() < deadline:
        time.sleep(0.01)
    return t


def stop_thread(spec, t):
    spec._running = False
    t.join(2)


def patch_candidates(monkeypatch, spec, backends):
    monkeypatch.setattr(spec, '_candidate_backends', lambda: backends)


# ---------------------------------------------------------------------------
# SpectrumThread: selection, fallback, laser routing
# ---------------------------------------------------------------------------

def test_auto_falls_back_to_second_backend(monkeypatch):
    spec = SpectrumThread(SpectrometerConfig(backend='auto', integration_time=1))
    wasatch = FakeBackend('wasatch', available=False, has_laser=True)
    kymera = FakeBackend('kymera', available=True)
    patch_candidates(monkeypatch, spec, [wasatch, kymera])
    t = run_thread(spec)
    try:
        assert spec.is_connected
        assert spec.backend_name == 'kymera'
        assert spec.backend_label == 'Kymera'
        # Axis from the backend's nm calibration, converted to cm^-1.
        assert len(spec.wavelengths) == 16
        np.testing.assert_allclose(spec.wavelengths_nm, kymera.wavelengths_nm)
        assert spec.wavelengths[0] == pytest.approx(
            1e7 / 785 - 1e7 / 790, rel=1e-6)
    finally:
        stop_thread(spec, t)


def test_connect_fails_when_no_backend_available(monkeypatch):
    spec = SpectrumThread(SpectrometerConfig(backend='auto'))
    patch_candidates(monkeypatch, spec, [FakeBackend('wasatch', False),
                                         FakeBackend('kymera', False)])
    errors = []
    spec.error.connect(lambda m: errors.append(m), Qt.ConnectionType.DirectConnection)
    t = threading.Thread(target=spec.run, daemon=True)
    t.start()
    deadline = time.time() + 2.0
    while not errors and time.time() < deadline:
        time.sleep(0.01)
    # Nothing connects: no backend, "No spectrometer detected", and the
    # thread idles (stays alive) so a later selection / Connect can retry.
    assert errors and errors[0].startswith("No spectrometer detected")
    assert t.is_alive()
    assert not spec.is_connected
    assert spec._backend is None
    assert spec.backend_name == ''
    stop_thread(spec, t)
    assert not t.is_alive()


def test_wasatch_backend_handles_laser_itself(monkeypatch):
    spec = SpectrumThread(SpectrometerConfig(backend='wasatch', integration_time=1))
    wasatch = FakeBackend('wasatch', has_laser=True)
    patch_candidates(monkeypatch, spec, [wasatch])
    t = run_thread(spec)
    try:
        assert spec.laser_available
        spec.set_laser_power_mw(120)
        spec.set_laser(True)
        spec.set_integration_time(50)
        deadline = time.time() + 2
        while (not wasatch.laser_enable or not wasatch.integration) \
                and time.time() < deadline:
            time.sleep(0.01)
        assert wasatch.laser_power[-1] == 120
        assert wasatch.laser_enable[-1] is True
        assert wasatch.integration[-1] == 50
        assert spec.is_laser_on
    finally:
        stop_thread(spec, t)


def test_kymera_backend_forwards_laser_to_laser_thread(monkeypatch):
    laser = FakeLaserThread()
    spec = SpectrumThread(SpectrometerConfig(backend='kymera', integration_time=1),
                          laser_thread=laser)
    kymera = FakeBackend('kymera')
    patch_candidates(monkeypatch, spec, [kymera])
    t = run_thread(spec)
    try:
        assert laser.started, "laser-only thread must start for a laser-less backend"
        assert spec.laser_available
        spec.set_laser_power_mw(300)
        spec.set_laser(True)
        # Forwarded immediately (not queued behind an exposure).
        assert laser.power == [300]
        assert laser.enable == [True]
        assert kymera.laser_enable == []
        spec.set_integration_time(25)
        deadline = time.time() + 2
        while not kymera.integration and time.time() < deadline:
            time.sleep(0.01)
        assert kymera.integration[-1] == 25
    finally:
        stop_thread(spec, t)
    assert not laser.started, "laser thread stops with the spectrum thread"


def test_laser_on_refused_when_laser_unavailable(monkeypatch):
    laser = FakeLaserThread()
    laser.is_available = False
    spec = SpectrumThread(SpectrometerConfig(backend='kymera', integration_time=1),
                          laser_thread=laser)
    patch_candidates(monkeypatch, spec, [FakeBackend('kymera')])
    errors, states = [], []
    spec.error.connect(errors.append)
    spec.laser_state_changed.connect(states.append)
    t = run_thread(spec)
    try:
        assert not spec.laser_available
        spec.set_laser(True)
        assert laser.enable == []
        assert states[-1] is False
        assert any("not available" in e for e in errors)
        # The laser becoming available later is reported through the thread.
        laser.available_changed.emit(True)
        assert spec.laser_available
    finally:
        stop_thread(spec, t)


def test_focus_metric_follows_backend(monkeypatch):
    cfg = SpectrometerConfig(backend='auto', integration_time=1)
    spec = SpectrumThread(cfg)
    patch_candidates(monkeypatch, spec, [FakeBackend('wasatch', False, True),
                                         FakeBackend('kymera')])
    t = run_thread(spec)
    try:
        assert spec.focus_metric == 'entropy'
        cfg.af_metric = 'ratio'
        assert spec.focus_metric == 'ratio'
    finally:
        stop_thread(spec, t)


def test_acquire_frames_uses_backend(monkeypatch):
    spec = SpectrumThread(SpectrometerConfig(backend='kymera', integration_time=1))
    kymera = FakeBackend('kymera')
    patch_candidates(monkeypatch, spec, [kymera])
    t = run_thread(spec)
    try:
        frames = spec.acquire_frames(3, timeout=2.0)
        assert frames is not None and frames.shape == (3, 16)
        assert kymera.acquired == 3
        meta = spec.metadata()
        assert meta["Spectrometer"] == "Fake kymera"
    finally:
        stop_thread(spec, t)


# ---------------------------------------------------------------------------
# KymeraSpectrometer against fake Andor SDK modules
# ---------------------------------------------------------------------------

DRV_SUCCESS = 20002
DRV_NO_NEW_DATA = 20024
DRV_TEMP_STABILIZED = 20036
DRV_TEMP_NOT_REACHED = 20037
ATS_OK = 20202


class FakeAtmcd:
    """Records the SDK2 call sequence; behaviour knobs on the class."""
    instances = []
    init_ret = DRV_SUCCESS
    wait_ret = DRV_SUCCESS
    wait_delay = 0.0
    temperature = (DRV_TEMP_NOT_REACHED, -35.0)
    xpixels = 1024
    ypixels = 255

    def __init__(self, userPath=None):
        self.calls = []
        self.exposure = None
        self.cancelled = threading.Event()
        FakeAtmcd.instances.append(self)

    def _rec(self, name, *args):
        self.calls.append((name, args))
        return DRV_SUCCESS

    def Initialize(self, d):
        self._rec('Initialize', d)
        return self.init_ret

    def GetCameraSerialNumber(self):
        return DRV_SUCCESS, 12345

    def SetTemperature(self, t):
        return self._rec('SetTemperature', t)

    def SetCoolerMode(self, m):
        return self._rec('SetCoolerMode', m)

    def CoolerON(self):
        return self._rec('CoolerON')

    def CoolerOFF(self):
        return self._rec('CoolerOFF')

    def GetTemperature(self):
        return self.temperature

    def SetAcquisitionMode(self, m):
        return self._rec('SetAcquisitionMode', int(m))

    def SetReadMode(self, m):
        return self._rec('SetReadMode', int(m))

    def SetTriggerMode(self, m):
        return self._rec('SetTriggerMode', int(m))

    def GetDetector(self):
        return DRV_SUCCESS, self.xpixels, self.ypixels

    def SetImage(self, *a):
        return self._rec('SetImage', *a)

    def SetSingleTrack(self, centre, height):
        return self._rec('SetSingleTrack', centre, height)

    def SetExposureTime(self, s):
        self.exposure = s
        return self._rec('SetExposureTime', s)

    def GetAcquisitionTimings(self):
        return DRV_SUCCESS, self.exposure or 0.0, 0.0, 0.0

    def PrepareAcquisition(self):
        return self._rec('PrepareAcquisition')

    def GetPixelSize(self):
        return DRV_SUCCESS, 26.0, 26.0

    def StartAcquisition(self):
        self.cancelled.clear()
        return self._rec('StartAcquisition')

    def WaitForAcquisitionTimeOut(self, ms):
        self._rec('WaitForAcquisitionTimeOut', ms)
        if self.wait_delay:
            if self.cancelled.wait(self.wait_delay):
                return DRV_NO_NEW_DATA
        return self.wait_ret

    def CancelWait(self):
        self.cancelled.set()
        return self._rec('CancelWait')

    def AbortAcquisition(self):
        return self._rec('AbortAcquisition')

    def GetImages16(self, first, last, size):
        self._rec('GetImages16', first, last, size)
        return DRV_SUCCESS, np.arange(size, dtype=np.uint16), first, last

    def ShutDown(self):
        return self._rec('ShutDown')


class FakeSpectrograph:
    instances = []
    grating = 1
    wavelength = 700.0

    def __init__(self, userPath=None):
        self.calls = []
        self.pixels = None
        FakeSpectrograph.instances.append(self)

    def _rec(self, name, *args):
        self.calls.append((name, args))
        return ATS_OK

    def Initialize(self, p):
        return self._rec('Initialize', p)

    def Close(self):
        return self._rec('Close')

    def GetFunctionReturnDescription(self, code, n):
        return ATS_OK, f"code {code}"

    def GetSerialNumber(self, dev, n):
        return ATS_OK, "KY-0001"

    def IsGratingPresent(self, dev):
        return ATS_OK, 1

    def GetGrating(self, dev):
        return ATS_OK, FakeSpectrograph.grating

    def SetGrating(self, dev, g):
        FakeSpectrograph.grating = g
        return self._rec('SetGrating', g)

    def IsWavelengthPresent(self, dev):
        return ATS_OK, 1

    def SetWavelength(self, dev, nm):
        FakeSpectrograph.wavelength = nm
        return self._rec('SetWavelength', nm)

    def GetWavelength(self, dev):
        return ATS_OK, FakeSpectrograph.wavelength

    def GetWavelengthLimits(self, dev, g):
        return ATS_OK, 300.0, 1100.0

    def SetNumberPixels(self, dev, n):
        self.pixels = n
        return self._rec('SetNumberPixels', n)

    def SetPixelWidth(self, dev, w):
        return self._rec('SetPixelWidth', w)

    def GetCalibration(self, dev, n):
        self._rec('GetCalibration', n)
        half = 50.0
        return ATS_OK, list(np.linspace(FakeSpectrograph.wavelength - half,
                                        FakeSpectrograph.wavelength + half, n))


@pytest.fixture
def fake_andor(monkeypatch):
    """Install fake pyAndorSDK2 / pyAndorSpectrograph modules."""
    FakeAtmcd.instances.clear()
    FakeSpectrograph.instances.clear()
    FakeAtmcd.init_ret = DRV_SUCCESS
    FakeAtmcd.wait_ret = DRV_SUCCESS
    FakeAtmcd.wait_delay = 0.0
    FakeAtmcd.temperature = (DRV_TEMP_NOT_REACHED, -35.0)
    FakeSpectrograph.grating = 1
    FakeSpectrograph.wavelength = 700.0

    codes = types.SimpleNamespace(
        Acquisition_Mode=types.SimpleNamespace(SINGLE_SCAN=1),
        Read_Mode=types.SimpleNamespace(FULL_VERTICAL_BINNING=0),
        Trigger_Mode=types.SimpleNamespace(INTERNAL=0),
    )
    sdk2 = types.ModuleType('pyAndorSDK2')
    sdk2.atmcd = FakeAtmcd
    sdk2.atmcd_codes = codes
    sdk2.__file__ = __file__
    spg_pkg = types.ModuleType('pyAndorSpectrograph')
    spg_pkg.__file__ = __file__
    spg = types.ModuleType('pyAndorSpectrograph.spectrograph')
    spg.ATSpectrograph = FakeSpectrograph
    spg_pkg.spectrograph = spg
    monkeypatch.setitem(sys.modules, 'pyAndorSDK2', sdk2)
    monkeypatch.setitem(sys.modules, 'pyAndorSpectrograph', spg_pkg)
    monkeypatch.setitem(sys.modules, 'pyAndorSpectrograph.spectrograph', spg)
    yield


def make_kymera(**kcfg):
    from raman_control.hardware.kymera_spectrometer import KymeraSpectrometer
    cfg = SpectrometerConfig(backend='kymera', integration_time=100.0,
                             kymera=KymeraConfig(**kcfg))
    return KymeraSpectrometer(cfg)


def test_kymera_connect_configures_fvb_single_scan_and_calibration(fake_andor):
    k = make_kymera(grating=3, center_wavelength_nm=853.0, target_temperature_c=-60)
    assert k.connect()
    cam = FakeAtmcd.instances[0]
    spc = FakeSpectrograph.instances[0]
    names = [c[0] for c in cam.calls]
    assert names.index('Initialize') < names.index('SetAcquisitionMode')
    assert ('SetAcquisitionMode', (1,)) in cam.calls      # SINGLE_SCAN
    assert ('SetReadMode', (0,)) in cam.calls             # FVB
    assert ('SetTriggerMode', (0,)) in cam.calls          # INTERNAL
    assert ('SetTemperature', (-60,)) in cam.calls
    assert ('CoolerON', ()) in cam.calls
    assert ('SetCoolerMode', (1,)) in cam.calls
    assert ('SetExposureTime', (0.1,)) in cam.calls       # 100 ms -> 0.1 s
    assert ('SetGrating', (3,)) in spc.calls
    assert ('SetWavelength', (853.0,)) in spc.calls
    assert spc.pixels == 1024
    assert k.pixels == 1024
    wl = k.wavelengths_nm
    assert wl.shape == (1024,)
    assert wl[0] == pytest.approx(803.0) and wl[-1] == pytest.approx(903.0)
    assert k.grating == 3 and k.center_wavelength_nm == 853.0
    st = k.status()
    assert st.serial == "12345"
    assert st.detector_temperature_c == -35.0
    assert st.temperature_stable is False
    meta = k.metadata()
    assert meta["Grating"] == 3
    assert meta["Center wavelength (nm)"] == "853.0"
    assert "Newton" in meta["Spectrometer"]


def test_kymera_connect_fails_when_camera_init_fails(fake_andor):
    FakeAtmcd.init_ret = 20992  # DRV_NOT_AVAILABLE
    k = make_kymera()
    assert not k.connect()
    assert not k.is_connected
    # Spectrograph was opened first and must be closed again.
    assert ('Close', ()) in FakeSpectrograph.instances[0].calls


def test_kymera_acquire_sets_exposure_and_reads_one_spectrum(fake_andor):
    k = make_kymera(acquisition_timeout_margin_s=2.0)
    assert k.connect()
    cam = FakeAtmcd.instances[0]
    k.set_integration_time_ms(250)
    spec = k.acquire()
    assert spec is not None and spec.shape == (1024,) and spec.dtype == np.float32
    assert spec[5] == 5.0
    assert cam.exposure == pytest.approx(0.25)
    names = [c[0] for c in cam.calls]
    i = names.index('StartAcquisition')
    assert names[i - 1] == 'SetExposureTime'          # applied before the start
    assert names[i + 1] == 'WaitForAcquisitionTimeOut'
    # Bounded wait: 2 x exposure + margin, in ms.
    assert cam.calls[i + 1][1] == (int((0.25 * 2 + 2.0) * 1000),)
    assert names[i + 2] == 'GetImages16'
    # Same exposure again -> SetExposureTime is NOT re-sent.
    n_set = names.count('SetExposureTime')
    k.acquire()
    assert [c[0] for c in cam.calls].count('SetExposureTime') == n_set


def test_kymera_acquire_timeout_aborts_and_returns_none(fake_andor):
    FakeAtmcd.wait_ret = DRV_NO_NEW_DATA
    k = make_kymera()
    assert k.connect()
    cam = FakeAtmcd.instances[0]
    assert k.acquire() is None
    assert ('AbortAcquisition', ()) in cam.calls
    assert k.is_connected  # a timeout does not drop the connection


def test_kymera_abort_interrupts_long_exposure(fake_andor):
    FakeAtmcd.wait_delay = 5.0  # "exposure" blocks until cancelled
    k = make_kymera()
    assert k.connect()
    cam = FakeAtmcd.instances[0]
    result = {}

    def worker():
        result['spec'] = k.acquire()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    deadline = time.time() + 2
    while 'WaitForAcquisitionTimeOut' not in [c[0] for c in cam.calls] \
            and time.time() < deadline:
        time.sleep(0.01)
    t0 = time.time()
    k.abort()
    t.join(2)
    assert not t.is_alive()
    assert time.time() - t0 < 1.5
    assert result['spec'] is None
    assert ('CancelWait', ()) in cam.calls
    assert ('AbortAcquisition', ()) in cam.calls


def test_kymera_temperature_poll_and_stable_flag(fake_andor):
    k = make_kymera()
    assert k.connect()
    FakeAtmcd.temperature = (DRV_TEMP_STABILIZED, -60.0)
    k._last_temp_poll = 0.0
    k.poll()
    assert k.temperature_c == -60.0
    assert k.temperature_stable is True
    assert k.status().temperature_stable is True


def test_kymera_disconnect_shuts_down_and_closes(fake_andor):
    k = make_kymera(keep_cooler_on_shutdown=True)
    assert k.connect()
    k.disconnect()
    cam = FakeAtmcd.instances[0]
    assert ('ShutDown', ()) in cam.calls
    assert ('CoolerOFF', ()) not in cam.calls          # cooler kept on
    assert ('Close', ()) in FakeSpectrograph.instances[0].calls
    assert not k.is_connected
    k.disconnect()  # idempotent


def test_kymera_missing_sdk_fails_cleanly(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyAndorSDK2', None)
    monkeypatch.setitem(sys.modules, 'pyAndorSpectrograph', None)
    monkeypatch.setitem(sys.modules, 'pyAndorSpectrograph.spectrograph', None)
    k = make_kymera()
    assert not k.connect()


# ---------------------------------------------------------------------------
# Config + metrics
# ---------------------------------------------------------------------------

def test_config_resolves_focus_metric_per_backend():
    cfg = SpectrometerConfig()
    assert cfg.resolve_af_metric('wasatch') == 'ratio'
    assert cfg.resolve_af_metric('kymera') == 'entropy'
    cfg.af_metric = 'entropy'
    assert cfg.resolve_af_metric('wasatch') == 'entropy'
    cfg.af_metric = 'ratio'
    assert cfg.resolve_af_metric('kymera') == 'ratio'


def test_hardware_config_round_trips_kymera_settings(tmp_path):
    cfg = HardwareConfig()
    cfg.spectrometer.backend = 'kymera'
    cfg.spectrometer.kymera.grating = 2
    cfg.spectrometer.kymera.center_wavelength_nm = 912.0
    cfg.spectrometer.af_metric = 'entropy'
    path = tmp_path / "hw.json"
    cfg.save_to_file(str(path))
    loaded = HardwareConfig.load_from_file(str(path))
    assert isinstance(loaded.spectrometer.kymera, KymeraConfig)
    assert loaded.spectrometer.backend == 'kymera'
    assert loaded.spectrometer.kymera.grating == 2
    assert loaded.spectrometer.kymera.center_wavelength_nm == 912.0
    assert loaded.spectrometer.af_metric == 'entropy'
    # Dotted update reaches the nested dataclass (Settings dialog path).
    loaded.update_from_dict({'spectrometer.kymera.grating': 3,
                             'spectrometer.backend': 'auto'})
    assert loaded.spectrometer.kymera.grating == 3
    assert loaded.spectrometer.backend == 'auto'


def test_axis_conversions_round_trip():
    nm = np.linspace(800, 950, 11)
    wn = wavelength_nm_to_wavenumber(nm, 785.0)
    assert wn[0] == pytest.approx(1e7 / 785 - 1e7 / 800)
    np.testing.assert_allclose(wavenumber_to_wavelength_nm(wn, 785.0), nm)


def test_entropy_metric_prefers_peaked_spectrum():
    wl = np.linspace(800, 950, 1024)
    flat = np.ones_like(wl) * 100
    peaked = flat + 5000 * np.exp(-((wl - 870) ** 2) / 2)
    band = (830.0, 920.0)
    e_flat = AutofocusEngine.calculate_entropy(wl, flat, band)
    e_peak = AutofocusEngine.calculate_entropy(wl, peaked, band)
    assert e_peak < e_flat
    # Dispatcher
    assert AutofocusEngine.spectrum_focus_score('entropy', wl, peaked,
                                                entropy_band=band) == e_peak
    assert AutofocusEngine.spectrum_focus_score(
        'ratio', wl, peaked, signal_band=(860, 880),
        reference_band=(900, 910)) > 1.0
    # Empty band -> worst score, never a false "best".
    assert AutofocusEngine.calculate_entropy(wl, peaked, (1000, 1100)) == float('inf')


# ---------------------------------------------------------------------------
# Kymera readout geometry: bin rows + 2D image readout
# ---------------------------------------------------------------------------

def _calls_named(cam, name):
    return [c for c in cam.calls if c[0] == name]


def test_kymera_default_rows_use_fvb_and_whole_chip(fake_andor):
    k = make_kymera()
    assert k.connect()
    cam = FakeAtmcd.instances[0]
    assert k.supports_image_readout and k.detector_rows == 255
    assert k.bin_rows == (1, 255)
    assert ('SetReadMode', (0,)) in cam.calls
    assert ('SetImage', (1, 1, 1, 1024, 1, 255)) in cam.calls
    st = k.status()
    assert st.details["Spectrum readout"] == "FVB"
    assert st.details["Bin rows"] == "1-255"


def test_kymera_partial_rows_read_as_single_track(fake_andor):
    k = make_kymera(bin_row_start=100, bin_row_end=140)
    assert k.connect()
    cam = FakeAtmcd.instances[0]
    assert k.bin_rows == (100, 140)
    assert ('SetReadMode', (3,)) in cam.calls                    # SINGLE_TRACK
    assert ('SetSingleTrack', (120, 41)) in cam.calls            # centre, height
    spec = k.acquire()
    assert spec is not None and spec.shape == (1024,)
    # One row of x pixels is read.
    assert _calls_named(cam, 'GetImages16')[-1][1] == (1, 1, 1024)
    assert k.status().details["Spectrum readout"] == "single track"


def test_kymera_set_bin_rows_applies_before_next_exposure(fake_andor):
    k = make_kymera()
    assert k.connect()
    cam = FakeAtmcd.instances[0]
    k.acquire()
    n_mode = len(_calls_named(cam, 'SetReadMode'))
    k.set_bin_rows(50, 60)
    assert k.bin_rows == (50, 60)
    assert len(_calls_named(cam, 'SetReadMode')) == n_mode  # deferred
    k.acquire()
    names = [c[0] for c in cam.calls]
    i_start = len(names) - 1 - names[::-1].index('StartAcquisition')
    i_mode = len(names) - 1 - names[::-1].index('SetReadMode')
    assert i_mode < i_start
    assert ('SetSingleTrack', (55, 11)) in cam.calls
    # Unchanged rows -> not re-sent.
    n_mode = len(_calls_named(cam, 'SetReadMode'))
    k.acquire()
    assert len(_calls_named(cam, 'SetReadMode')) == n_mode
    # Back to the full range -> FVB again.
    k.set_bin_rows(1, 0)
    k.acquire()
    assert _calls_named(cam, 'SetReadMode')[-1][1] == (0,)


def test_kymera_rows_are_clamped_and_ordered(fake_andor):
    k = make_kymera()
    assert k.connect()
    k.set_bin_rows(0, 0)
    assert k.bin_rows == (1, 255)
    k.set_bin_rows(300, 10)       # reversed + beyond the sensor
    assert k.bin_rows == (10, 255)
    k.set_bin_rows(200, 150)      # reversed
    assert k.bin_rows == (150, 200)
    k.set_bin_rows(-5, 3)
    assert k.bin_rows == (1, 3)


def test_kymera_acquire_image_reads_whole_sensor_then_spectrum_restores(fake_andor):
    k = make_kymera(bin_row_start=10, bin_row_end=12)
    assert k.connect()
    cam = FakeAtmcd.instances[0]
    frame = k.acquire_image()
    assert ('SetReadMode', (4,)) in cam.calls                     # IMAGE
    assert ('SetImage', (1, 1, 1, 1024, 1, 255)) in cam.calls
    assert _calls_named(cam, 'GetImages16')[-1][1] == (1, 1, 1024 * 255)
    assert frame is not None and frame.shape == (255, 1024)
    assert frame.dtype == np.float32
    assert frame[3, 7] == 3 * 1024 + 7     # fake returns arange(size)
    # The next spectrum goes back to the single-track readout of the rows.
    n_track = len(_calls_named(cam, 'SetSingleTrack'))
    spec = k.acquire()
    assert spec.shape == (1024,) and np.array_equal(spec, np.arange(1024))
    assert _calls_named(cam, 'SetReadMode')[-1][1] == (3,)
    assert len(_calls_named(cam, 'SetSingleTrack')) == n_track + 1
    assert _calls_named(cam, 'SetSingleTrack')[-1][1] == (11, 3)
    assert _calls_named(cam, 'GetImages16')[-1][1] == (1, 1, 1024)


def test_kymera_acquire_image_timeout_returns_none(fake_andor):
    FakeAtmcd.wait_ret = DRV_NO_NEW_DATA
    k = make_kymera()
    assert k.connect()
    assert k.acquire_image() is None
    assert ('AbortAcquisition', ()) in FakeAtmcd.instances[0].calls


def test_to_counts_unwraps_signed_16bit_readout():
    from raman_control.hardware.kymera_spectrometer import _to_counts
    raw = np.array([0, 100, -1, -32768], dtype=np.int16)  # c_short buffer
    out = _to_counts(raw)
    assert out.dtype == np.float32
    assert out.tolist() == [0.0, 100.0, 65535.0, 32768.0]


class FakeImageBackend(FakeBackend):
    """FakeBackend with a 2D detector (rows x pixels)."""

    def __init__(self, name, rows=8, **kw):
        super().__init__(name, **kw)
        self.rows = rows
        self._bin = (1, rows)
        self.images = 0

    @property
    def supports_image_readout(self):
        return True

    @property
    def detector_rows(self):
        return self.rows

    def set_bin_rows(self, start, end):
        end = self.rows if end <= 0 else end
        self._bin = (start, end)

    @property
    def bin_rows(self):
        return self._bin

    def acquire_image(self):
        self.images += 1
        return np.tile(np.arange(self._pixels, dtype=np.float32), (self.rows, 1))


def test_thread_image_mode_reads_frames_instead_of_spectra(monkeypatch):
    spec = SpectrumThread(SpectrometerConfig(backend='kymera', integration_time=1))
    backend = FakeImageBackend('kymera', rows=8, pixels=16)
    patch_candidates(monkeypatch, spec, [backend])
    frames, spectra = [], []
    # Direct connections: the emitting thread has no Qt event loop here.
    spec.image_ready.connect(lambda f, ts: frames.append(f),
                             Qt.ConnectionType.DirectConnection)
    spec.spectrum_ready.connect(lambda w, i, ts: spectra.append(i),
                                Qt.ConnectionType.DirectConnection)
    t = run_thread(spec)
    try:
        assert spec.supports_image_readout and spec.detector_rows == 8
        spec.set_bin_rows(2, 3)
        spec.set_image_mode(True)
        spec.start_acquisition()
        deadline = time.time() + 2.0
        while len(frames) < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert len(frames) >= 2 and frames[0].shape == (8, 16)
        assert backend.images >= 2
        # Rows were recorded on the backend but the live loop took no spectra.
        assert backend.bin_rows == (2, 3) and spec.bin_rows == (2, 3)
        assert spectra == [] and backend.acquired == 0
        # On-demand frames (autofocus / acquisition) still take real spectra.
        out = spec.acquire_frames(2, timeout=2.0)
        assert out is not None and out.shape == (2, 16)
        assert backend.acquired == 2 and len(spectra) == 1
        # Back to spectrum display: spectra flow, frames stop.
        spec.set_image_mode(False)
        n_frames, n_images = len(frames), backend.images
        deadline = time.time() + 2.0
        while len(spectra) < 3 and time.time() < deadline:
            time.sleep(0.01)
        assert len(spectra) >= 3
        assert backend.images in (n_images, n_images + 1)  # at most one in flight
        assert len(frames) in (n_frames, n_frames + 1)
    finally:
        stop_thread(spec, t)


def test_hardware_config_round_trips_bin_rows(tmp_path):
    cfg = HardwareConfig()
    assert cfg.spectrometer.kymera.bin_row_start == 1
    assert cfg.spectrometer.kymera.bin_row_end == 0
    cfg.spectrometer.kymera.bin_row_start = 90
    cfg.spectrometer.kymera.bin_row_end = 160
    path = tmp_path / "hw.json"
    cfg.save_to_file(str(path))
    loaded = HardwareConfig.load_from_file(str(path))
    assert loaded.spectrometer.kymera.bin_row_start == 90
    assert loaded.spectrometer.kymera.bin_row_end == 160
    loaded.update_from_dict({'spectrometer.kymera.bin_row_end': 0})
    assert loaded.spectrometer.kymera.bin_row_end == 0
