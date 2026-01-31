# noodlepy/gui/settingmodule.py

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from dataclasses import dataclass, field
import os
from PIL import Image, ImageTk
from noodlepy.gui.publisher_subscriber import Publisher, Subscriber


# ------------ STAGE CONTROL SETTINGS ------------

@dataclass
class StageSettings():
    def __init__(self):
        # Basic geometry / safety
        self.safety_height: float = 15.0  # mm
        self.home_x: float = 118.90
        self.home_y: float = 173.88
        self.home_z: float = 15.0

        # Widefield ↔ objective calibration (mm)
        self.calib_wide_to_obj_x: float = -62.48
        self.calib_wide_to_obj_y: float = 1.32
        self.calib_wide_to_obj_z: float = -13.26

        # Nanodrive & step sizes
        self.initial_nanodrive_position: float = 50.0
        self.small_step_xy_mm: float = 0.06
        self.medium_step_xy_mm: float = 0.5
        self.large_step_xy_mm: float = 4.5
        self.small_step_z_mm: float = 0.02
        self.medium_step_z_mm: float = 0.1
        self.large_step_z_mm: float = 1.0
        self.nanodrive_step_size_um: float = 1.0

        # Autofocus / stabilization
        self.nanodrive_movement_stabilization_time: float = 0.02  # s
        self.prusa_movement_stabilization_time: float = 1.0       # s
        # Sample drop panel geometry (mm / px)
        self.interval_between_drop_x: float = 4.5
        self.interval_between_drop_y: float = 4.5
        self.offset_home_to_p1_x: float = 6.22
        self.offset_home_to_p1_y: float = -10.0
        self.interval_between_quartz_slides_x: float = 27.5
        self.interval_between_quartz_slides_y: float = 27.5



# ------------ LIVE VIEW / CAMERA SETTINGS ------------

@dataclass
class LiveViewSettings():
    def __init__(self):
        # Canvas size
        self.width: int = 500
        self.height: int = 375

        # Pixel sizes (µm)
        self.x_pixel_size_objective_camera: float = 0.191
        self.y_pixel_size_objective_camera: float = 0.191
        self.x_pixel_size_widefield_camera: float = 2.82
        self.y_pixel_size_widefield_camera: float = 2.82

        # Crop center (pixels)
        self.objective_crop_center_x: int = 798
        self.objective_crop_center_y: int = 452

        # Crop size (pixels)
        self.objective_crop_width: int = 500
        self.objective_crop_height: int = 375
        # Sampling method
        self.sampling_offset_from_the_edge: int = 40

        # Camera serials (so you can change camera HW without touching code)
        self.objective_camera_serial: str = "14628"
        self.widefield_camera_serial: str = "14938"
        self.flip_objective_camera = True
        self.crop_objective_camera = True
        self.rotate_objective_camera = True

# ------------ APP-WIDE SETTINGS CONTAINER ------------

@dataclass
class AppSettings:
    stage: StageSettings = field(default_factory=StageSettings)
    liveview: LiveViewSettings = field(default_factory=LiveViewSettings)


# ------------ SETTINGS MODULE FRAME (HAS THE BUTTON + POPUP METHOD) ------------

class SettingsModule(ttk.Frame, Publisher, Subscriber):
    """
    Small module frame that just has one button.
    Clicking the button opens a settings window (Toplevel).
    """
    def __init__(self, parent, app_settings: AppSettings):
        super().__init__(parent)
        Subscriber.__init__(self)
        Publisher.__init__(self, ['update_settings'])

        self.app_settings = app_settings       # shared dataclasses
        self.name = "SettingsModule_observableobserver"

        self.setting_icon = Image.open(
            os.path.join(os.getcwd(), "noodlepy", "assets", "setting_icon.png")
        ).resize((20, 20))
        self._create_widgets()

    def _create_widgets(self):
        # Keep this frame compact; no expansion
        lf = ttk.Frame(self)
        lf.grid(row=0, column=0)  # no sticky here

        # Use the loaded setting icon instead of text
        self.setting_icon_tk = ImageTk.PhotoImage(self.setting_icon)  # keep reference to avoid GC
        btn = ttk.Button(
            lf,
            image=self.setting_icon_tk,
            bootstyle="dark",
            command=self.open_settings_window,
        )
        btn.grid(row=0, column=0, padx=5, pady=5)

    # ------------ POPUP WINDOW CREATION ------------

    def open_settings_window(self):
        """
        Build the settings window as a Toplevel, not a separate class.
        """
        parent = self.winfo_toplevel()
        win = ttk.Toplevel(parent)
        win.title("System Settings")
        win.resizable(False, False)

        # local dict to store entry widgets
        entries: dict[tuple[str, str], tk.Entry] = {}

        nb = ttk.Notebook(win)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        # =====================================================================
        # STAGE TAB
        # =====================================================================
        stage_frame = ttk.Frame(nb, padding=10)
        nb.add(stage_frame, text="Stage")

        # --- Basic geometry / safety (col 0, row 0) ---
        geom_lf = ttk.Labelframe(stage_frame, text="Basic geometry / safety", padding=8)
        geom_lf.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self._add_float_entry(geom_lf, "Safety height (mm)", "stage", "safety_height", 0, entries)
        self._add_float_entry(geom_lf, "Home X (mm)",        "stage", "home_x",        1, entries)
        self._add_float_entry(geom_lf, "Home Y (mm)",        "stage", "home_y",        2, entries)
        self._add_float_entry(geom_lf, "Home Z (mm)",        "stage", "home_z",        3, entries)

        # --- Calibration (col 1, row 0) ---
        calib_lf = ttk.Labelframe(stage_frame, text="Widefield → objective calibration (mm)", padding=8)
        calib_lf.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self._add_float_entry(calib_lf, "ΔX (mm)", "stage", "calib_wide_to_obj_x", 0, entries)
        self._add_float_entry(calib_lf, "ΔY (mm)", "stage", "calib_wide_to_obj_y", 1, entries)
        self._add_float_entry(calib_lf, "ΔZ (mm)", "stage", "calib_wide_to_obj_z", 2, entries)

        # --- Step sizes (col 0, row 1) ---
        step_lf = ttk.Labelframe(stage_frame, text="Nanodrive & step sizes", padding=8)
        step_lf.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self._add_float_entry(step_lf, "Initial nanodrive (µm)", "stage", "initial_nanodrive_position", 0, entries)

        self._add_float_entry(step_lf, "Small step XY (mm)",  "stage", "small_step_xy_mm",   1, entries)
        self._add_float_entry(step_lf, "Medium step XY (mm)", "stage", "medium_step_xy_mm",  2, entries)
        self._add_float_entry(step_lf, "Large step XY (mm)",  "stage", "large_step_xy_mm",   3, entries)

        self._add_float_entry(step_lf, "Small step Z (mm)",   "stage", "small_step_z_mm",    4, entries)
        self._add_float_entry(step_lf, "Medium step Z (mm)",  "stage", "medium_step_z_mm",   5, entries)
        self._add_float_entry(step_lf, "Large step Z (mm)",   "stage", "large_step_z_mm",    6, entries)

        self._add_float_entry(step_lf, "Nanodrive step (µm)", "stage", "nanodrive_step_size_um", 7, entries)

        # --- Stabilization (col 1, row 1) ---
        stab_lf = ttk.Labelframe(stage_frame, text="Stabilization (s)", padding=8)
        stab_lf.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self._add_float_entry(stab_lf, "Nanodrive move stabilization (s)",
                              "stage", "nanodrive_movement_stabilization_time", 0, entries)
        self._add_float_entry(stab_lf, "Prusa move stabilization (s)",
                              "stage", "prusa_movement_stabilization_time",    1, entries)

        # --- Sample drop geometry (span both columns, row 2) ---
        drop_lf = ttk.Labelframe(stage_frame, text="Sample drop panel geometry", padding=8)
        drop_lf.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        self._add_float_entry(drop_lf, "Drop interval X (mm)", "stage", "interval_between_drop_x", 0, entries)
        self._add_float_entry(drop_lf, "Drop interval Y (mm)", "stage", "interval_between_drop_y", 1, entries)

        self._add_float_entry(drop_lf, "Offset home→P1 X (mm)", "stage", "offset_home_to_p1_x", 2, entries)
        self._add_float_entry(drop_lf, "Offset home→P1 Y (mm)", "stage", "offset_home_to_p1_y", 3, entries)

        self._add_float_entry(drop_lf, "Quartz slide interval X (mm)", "stage", "interval_between_quartz_slides_x", 4, entries)
        self._add_float_entry(drop_lf, "Quartz slide interval Y (mm)", "stage", "interval_between_quartz_slides_y", 5, entries)

        # Layout: 2 columns that share space nicely
        stage_frame.grid_columnconfigure(0, weight=1)
        stage_frame.grid_columnconfigure(1, weight=1)

        # =====================================================================
        # LIVE VIEW TAB
        # =====================================================================
        live_frame = ttk.Frame(nb, padding=10)
        nb.add(live_frame, text="Live View")

        # --- Canvas size (col 0, row 0) ---
        canvas_lf = ttk.Labelframe(live_frame, text="Canvas size (px)", padding=8)
        canvas_lf.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self._add_int_entry(canvas_lf, "Width",  "liveview", "width",  0, entries)
        self._add_int_entry(canvas_lf, "Height", "liveview", "height", 1, entries)

        # --- Pixel sizes (col 1, row 0) ---
        pixel_lf = ttk.Labelframe(live_frame, text="Pixel size (µm)", padding=8)
        pixel_lf.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self._add_float_entry(pixel_lf, "Objective X", "liveview", "x_pixel_size_objective_camera", 0, entries)
        self._add_float_entry(pixel_lf, "Objective Y", "liveview", "y_pixel_size_objective_camera", 1, entries)
        self._add_float_entry(pixel_lf, "Widefield X", "liveview", "x_pixel_size_widefield_camera", 2, entries)
        self._add_float_entry(pixel_lf, "Widefield Y", "liveview", "y_pixel_size_widefield_camera", 3, entries)

        # --- Crop (col 0, row 1) ---
        crop_lf = ttk.Labelframe(live_frame, text="Crop (px)", padding=8)
        crop_lf.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self._add_int_entry(crop_lf, "Center X", "liveview", "objective_crop_center_x", 0, entries)
        self._add_int_entry(crop_lf, "Center Y", "liveview", "objective_crop_center_y", 1, entries)
        self._add_int_entry(crop_lf, "Width",    "liveview", "objective_crop_width",    2, entries)
        self._add_int_entry(crop_lf, "Height",   "liveview", "objective_crop_height",   3, entries)

        # --- Sampling (col 1, row 1) ---
        sampling_lf = ttk.Labelframe(live_frame, text="Sampling", padding=8)
        sampling_lf.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self._add_int_entry(sampling_lf, "Offset from edge", "liveview", "sampling_offset_from_the_edge", 0, entries)

        # --- Camera serials (span 2 columns, row 5) ---
        cam_lf = ttk.Labelframe(live_frame, text="Camera serials", padding=8)
        cam_lf.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        self._add_str_entry(cam_lf, "Widefield camera", "liveview", "widefield_camera_serial", 0, entries)
        self._add_str_entry(cam_lf, "Objective camera", "liveview", "objective_camera_serial", 1, entries)
        self._add_str_entry(cam_lf, "Flip objective camera", "liveview", "flip_objective_camera", 2, entries)
        self._add_str_entry(cam_lf, "Crop objective camera", "liveview", "crop_objective_camera", 3, entries)
        self._add_str_entry(cam_lf, "Rotate objective camera", "liveview", "rotate_objective_camera", 4, entries)


        live_frame.grid_columnconfigure(0, weight=1)
        live_frame.grid_columnconfigure(1, weight=1)

        # =====================================================================
        # BUTTONS + CENTERING
        # =====================================================================
        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))

        # Center the settings window over the parent application window
        parent.update_idletasks()
        win.update_idletasks()
        win_w = win.winfo_reqwidth()
        win_h = win.winfo_reqheight()

        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        p_w = parent.winfo_width() or parent.winfo_reqwidth()
        p_h = parent.winfo_height() or parent.winfo_reqheight()

        x = int(px + (p_w - win_w) / 2)
        y = int(py + (p_h - win_h) / 2)
        win.geometry(f"+{x}+{y}")

        def on_cancel():
            win.destroy()

        def on_save():
            # update dataclasses from entries
            for (group, attr), entry in entries.items():
                text = entry.get()
                target_obj = getattr(self.app_settings, group)
                current = getattr(target_obj, attr)

                if isinstance(current, int):
                    try:
                        value = int(text)
                    except ValueError:
                        continue
                elif isinstance(current, float):
                    try:
                        value = float(text)
                    except ValueError:
                        continue
                else:
                    # strings, etc.
                    value = text

                setattr(target_obj, attr, value)

            self.dispatch('update_settings', self.app_settings)

            win.destroy()

        ttk.Button(btn_frame, text="Cancel",
                   command=on_cancel,
                   bootstyle="danger").pack(side="right", padx=5)

        ttk.Button(btn_frame, text="Save",
                   command=on_save,
                   bootstyle="info").pack(side="right", padx=5)


    # ------------ helper methods to build entries ------------

    def _add_entry(self, parent, label_text, group, attr, row, entries_dict):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", pady=2, padx=2)
        e = ttk.Entry(parent, width=15)
        e.grid(row=row, column=1, sticky="ew", pady=2, padx=2)
        parent.grid_columnconfigure(1, weight=1)

        current_value = getattr(getattr(self.app_settings, group), attr)
        e.insert(0, str(current_value))

        entries_dict[(group, attr)] = e

    def _add_float_entry(self, parent, label_text, group, attr, row, entries_dict):
        self._add_entry(parent, label_text, group, attr, row, entries_dict)

    def _add_int_entry(self, parent, label_text, group, attr, row, entries_dict):
        self._add_entry(parent, label_text, group, attr, row, entries_dict)

    def _add_str_entry(self, parent, label_text, group, attr, row, entries_dict):
        self._add_entry(parent, label_text, group, attr, row, entries_dict)

    def handle_update_home_position(self, new_home_position):
        self.app_settings.stage.home_x = new_home_position[0]
        self.app_settings.stage.home_y = new_home_position[1]
        self.app_settings.stage.home_z = new_home_position[2]
