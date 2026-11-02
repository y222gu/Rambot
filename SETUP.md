# Setting up the raman_control environment

These steps reproduce the lab environment (`Rambot_venv`) on a new Windows PC.
Tested with Python 3.12.3, 64-bit, Windows 11.

## 1. Python and virtual environment

Install Python 3.12 (64-bit) from python.org, then from the repository root:

```powershell
python -m venv Rambot_venv
.\Rambot_venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins every package to the versions in the working lab
environment. PyTorch comes from the CUDA 12.4 index (see the comment in the
file if you want the smaller CPU-only build).

## 2. Thorlabs camera SDK (not on PyPI)

The Python package `thorlabs_tsi_sdk` ships inside the Thorlabs Scientific
Camera SDK download, not on PyPI.

1. Download the "Scientific Camera Interfaces" package from the Thorlabs
   website (Software tab of any Zelux / Kiralux / CS-series camera page).
2. Inside it, locate `thorlabs_tsi_camera_python_sdk_package.zip` and install
   it into the activated venv:

   ```powershell
   pip install path\to\thorlabs_tsi_camera_python_sdk_package.zip
   ```

   The lab copy of this zip is kept at
   `C:\Users\yifei\Documents\thorlabs_tsi_camera_python_sdk_package.zip`.
   Installed version in the lab: 0.0.8.
3. Copy the 64-bit native DLLs from the same SDK download
   (`Native_64_lib` folder, files such as `thorlabs_tsi_camera_sdk.dll`)
   into `raman_control/dlls/64_lib/`. The camera thread adds this folder to
   the DLL search path at startup.
4. Install the Thorlabs camera USB driver (ThorCam installer) so Windows
   recognises the cameras.

## 3. MadCity Labs NanoDrive (piezo)

No pip package. The software loads `raman_control/dlls/Madlib.dll` via
ctypes. Copy `Madlib.dll` (64-bit) from the MadCity Labs NanoDrive software
installation into `raman_control/dlls/`. If the DLL is missing the piezo
runs in simulated mode.

Note: the whole `raman_control/dlls/` folder is listed in `.gitignore`, so
these DLLs must be copied by hand on each new machine.

## 4. Wasatch spectrometer

The `wasatch` package is installed from PyPI by `requirements.txt`. The
spectrometer also needs the libusb-based driver installed by the Wasatch
ENLIGHTEN installer (or Zadig) so `pyusb` can open the device.

## 4b. Andor Kymera 328i + Newton DU920P (alternative spectrometer)

The software can collect spectra with the Andor Kymera spectrograph and Newton
CCD instead of the Wasatch (Settings -> Spectrometer -> "Spectrometer":
Auto / Wasatch Photonics / Andor Kymera + Newton; Auto uses the Wasatch when
one is detected and falls back to the Kymera). The excitation laser is always
the one on the Wasatch spectrometer-laser unit: with the Kymera active, the
Wasatch is opened for laser control only.

Andor's Python wrappers are not on PyPI. They ship with the Andor SDK
installer (also installed by Andor SOLIS) and are installed from there into
the activated venv:

```powershell
pip install "C:\Program Files\Andor SDK\Python\pyAndorSDK2"
pip install "C:\Program Files\Andor SDK\Python\pyAndorSpectrograph"
```

Both packages bundle the 64-bit DLLs they need (`atmcd64d.dll`,
`atspectrograph.dll`); the Andor USB drivers come with the SDK / SOLIS
installer. Installed version in the lab: 1.2.0 for both.

Notes:
- Only one program can hold the Newton camera. **Close Andor SOLIS** before
  connecting from raman_control (Initialize fails otherwise).
- Connecting takes 10 to 30 s: the SDK initialisation is about 10 s and a
  grating change (turret move) another 15 to 20 s.
- The CCD cooler is switched on at connect (set point in Settings, default
  -60 C) and by default is kept on when the software disconnects.

## 5. Prusa stage and other serial devices

Install the USB-serial drivers for the Prusa board and any Arduino-based
accessories, then set the COM ports in `raman_control/config/hardware_config.py`.
If the LED controller uses a CH340 chip, keep the CH340 driver at version
3.7.2022.1; newer drivers break re-opening the port.

## 6. Segmentation model weights

Model checkpoints (`*.pth`) are excluded from git. Copy them into
`raman_control/segmentation/models/`:

- `sam_vit_h_4b8939.pth` (Segment Anything, ViT-H, about 2.5 GB)
- `unet_trained_20260205.pth` (lab-trained U-Net)

## 7. Run

```powershell
.\Rambot_venv\Scripts\Activate.ps1
python -m raman_control.main
```

Run it as a module from the repository root. `python raman_control/main.py`
fails because the package uses absolute imports.

Tests:

```powershell
python -m pytest raman_control/tests -v
```
