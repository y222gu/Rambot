import numpy as np
from ttkbootstrap.constants import *
from wasatch.WasatchBus import WasatchBus
from wasatch.WasatchDevice import WasatchDevice
from wasatch.RealUSBDevice import RealUSBDevice

class WasatchManager():
    def __init__(self, integ_time_ms, laser_power_mW):
        self.integ_time_ms = integ_time_ms
        self.laser_power_mW = laser_power_mW
        self.connect()

    def connect(self):
        bus = WasatchBus(use_sim=False)
        if not bus.device_ids:
            print("No Wasatch USB spectrometers found.")
            return False

        self.device_id = bus.device_ids[0]
        print(f"Connecting to {self.device_id}")
        self.device_id.device_type = RealUSBDevice(self.device_id)
        self.device = WasatchDevice(self.device_id)
        ok = self.device.connect()
        if not ok:
            print("Can't connect to %s", self.device_id)
            return False

        self.settings = self.device.settings
        self.fid = self.device.hardware
        self.fid.set_integration_time_ms(self.integ_time_ms)
        self.fid.set_laser_power_mW(self.laser_power_mW)

        if self.settings.wavelengths is None:
            print("Script requires Raman spectrometer")
            return False

        print("connected to %s %s with %d pixels (%.2f, %.2fnm) (%.2f, %.2fcm¹)" % (
            self.settings.eeprom.model,
            self.settings.eeprom.serial_number,
            self.settings.pixels(),
            self.settings.wavelengths[0],
            self.settings.wavelengths[-1],
            self.settings.wavenumbers[0],
            self.settings.wavenumbers[-1]))
        
        self.fid.set_laser_power_high_resolution(True)
        return True

    def turn_laser_on(self):
        self.fid.set_laser_enable(True)
        print("Turning laser on. Please wait for laser to stabilize.")
    
    def turn_laser_off(self):
        self.fid.set_laser_enable(False)
        print("Turning laser off.")

    def set_integration_time(self, time_ms):
        self.fid.set_integration_time_ms(time_ms)
        self.integ_time_ms = time_ms
        print(f"Integration time set to {time_ms}ms")

    def set_laser_power(self, power_mW):
        self.fid.set_laser_power_mW(power_mW)
        self.laser_power_mW = power_mW
        print(f"Laser power set to {power_mW}mW")

    def get_spectrum(self):
        response = self.fid.get_line()
        if response and response.data:
            spectrum = response.data.spectrum
            return np.asarray(spectrum)