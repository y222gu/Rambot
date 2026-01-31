import os
import time
from pyparsing import col
import serial
import serial.tools.list_ports
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from PIL import Image, ImageTk
from tkinter import StringVar, messagebox
import tkinter as tk
import numpy as np
from threading import Thread, Lock
from noodlepy.gui.publisher_subscriber import Subscriber, Publisher
from noodlepy.gui.calibrationmodule import CalibrationModule

class StageControlModule(ttk.Frame, Publisher, Subscriber):
    def __init__(self, parent, settings):
        ttk.Frame.__init__(self, parent)
        Publisher.__init__(self, ['move_nanodrive_by', 
                                  'move_nanodrive_to', 
                                  'calculate_focus_score', 
                                  'get_nanodrive_position', 
                                  'activate_camera_autofocus_button', 
                                  'activate_switch_view_button', 
                                  'activate_protocol_module_state',
                                  'get_nanodrive_min_max',
                                  'task_completed',
                                  'abort_aquisition',
                                  'update_prusa_focus_range_for_wasatch',
                                  'update_home_position'])
        Subscriber.__init__(self)
        self.name = 'StageControlModule_obserableobserver'
        self.parent = parent
        self.settings = settings
        self.apply_settings()

        self.img_small_step = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","small_step.png")).resize((20, 20))
        self.img_medium_step = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","medium_step.png")).resize((20, 20))
        self.img_large_step = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","large_step.png")).resize((20, 20))
        self.circle_black = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","circle-black.png")).resize((20, 20))
        self.nanodrive_icon = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","nanodrive_arrow_1.png")).resize((20, 20))
        self.home_icon = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","home_icon.png")).resize((40, 40))

        self.img_up_low = ImageTk.PhotoImage(self.img_small_step.rotate(90))
        self.img_up_medium = ImageTk.PhotoImage(self.img_medium_step.rotate(-90))
        self.img_up_high = ImageTk.PhotoImage(self.img_large_step.rotate(90))
        self.img_down_low = ImageTk.PhotoImage(self.img_small_step.rotate(-90))
        self.img_down_medium = ImageTk.PhotoImage(self.img_medium_step.rotate(90))
        self.img_down_high = ImageTk.PhotoImage(self.img_large_step.rotate(-90))
        self.img_left_low = ImageTk.PhotoImage(self.img_small_step.rotate(180))
        self.img_left_medium = ImageTk.PhotoImage(self.img_medium_step)
        self.img_left_high = ImageTk.PhotoImage(self.img_large_step.rotate(180))
        self.img_right_low = ImageTk.PhotoImage(self.img_small_step)
        self.img_right_medium = ImageTk.PhotoImage(self.img_medium_step.rotate(180))
        self.img_right_high = ImageTk.PhotoImage(self.img_large_step)
        self.img_up_nano = ImageTk.PhotoImage(self.nanodrive_icon.rotate(180))
        self.img_down_nano = ImageTk.PhotoImage(self.nanodrive_icon)
        self.img_home_icon = ImageTk.PhotoImage(self.home_icon)

        self.widefield_focus_prusa_upper_limit = None
        self.widefield_focus_prusa_lower_limit = None
        self.focus_score_at_current_z_position = None
        self.current_nanodrive_position = None #um
        self.prusa_x_referenced = False
        self.prusa_y_referenced = False
        self.prusa_z_referenced = False
        self.capture_frame_center_widefield_x= None
        self.capture_frame_center_widefield_y = None
        self.capture_frame_center_objective_x= None
        self.capture_frame_center_objective_y = None
        self.ser = None

        self.sample_drop_panel_circle_radius = 15
        self.sample_drop_panel_spacing = 40
        self.sample_drop_panel_grid_pattern = [4, 4, 4, 4, 4]  # Column-wise circle counts for symmetry
        self.sample_drop_panel_max_rows = max(self.sample_drop_panel_grid_pattern)
        self.sample_drop_panel_total_width = len(self.sample_drop_panel_grid_pattern) * self.sample_drop_panel_spacing
        self.sample_drop_panel_total_height = self.sample_drop_panel_max_rows * self.sample_drop_panel_spacing
        self.sample_drop_panel_current_sample_row_column = None
        self.sample_drop_panel_circle_pursa_coordinates_in_widefield = {}  # Store physical coordinates of circles
        self.sample_drop_panel_circle_pursa_coordinates_in_objective = {}  # Store physical coordinates of circles in objective view
        self.sample_drop_panel_calculate_circle_pursa_coordinates()
        self.create_widgets()
        self.connect_prusa_device()

        # parameters for the protocol module
        self.selected_circle_positions = []

    def apply_settings(self):
        s = self.settings

        # Safety / home
        self.safety_height = s.safety_height
        self.home_position = [s.home_x, s.home_y, s.home_z]

        # Calibration
        self.calibration_from_widefield_to_objective_x = s.calib_wide_to_obj_x
        self.calibration_from_widefield_to_objective_y = s.calib_wide_to_obj_y
        self.calibration_from_widefield_to_objective_z = s.calib_wide_to_obj_z

        # Steps / nanodrive
        self.initial_nanodrive_position = s.initial_nanodrive_position
        self.small_step_size_xy_mm = s.small_step_xy_mm
        self.medium_step_size_xy_mm = s.medium_step_xy_mm
        self.large_step_size_xy_mm = s.large_step_xy_mm
        self.small_step_size_z_mm = s.small_step_z_mm
        self.medium_step_size_z_mm = s.medium_step_z_mm
        self.large_step_size_z_mm = s.large_step_z_mm
        self.nanodrive_step_size_um = s.nanodrive_step_size_um

        # Focus & stabilization
        self.nanodrive_movement_stabilization_time = s.nanodrive_movement_stabilization_time
        self.prusa_movement_stabilization_time = s.prusa_movement_stabilization_time

        # Sample drop panel
        self.interval_between_drop_x = s.interval_between_drop_x
        self.interval_between_drop_y = s.interval_between_drop_y
        self.offset_home_to_p1_x = s.offset_home_to_p1_x
        self.offset_home_to_p1_y = s.offset_home_to_p1_y
        self.interval_between_quartz_slides_x = s.interval_between_quartz_slides_x
        self.interval_between_quartz_slides_y = s.interval_between_quartz_slides_y

    def handle_update_settings(self, new_settings):
        self.settings = new_settings.stage
        self.apply_settings()
        
        ttk.Label(self.direction_frame, text=f"{self.large_step_size_xy_mm} mm").grid(row=1, column=3, pady=3)
        ttk.Label(self.direction_frame, text=f"{self.medium_step_size_xy_mm} mm").grid(row=2, column=3, pady=3)
        ttk.Label(self.direction_frame, text=f"{self.small_step_size_xy_mm} mm").grid(row=3, column=3,  pady=3)
        ttk.Label(self.direction_frame, text=f"{self.large_step_size_z_mm} mm").grid(row=1, column=9, pady=3)
        ttk.Label(self.direction_frame, text=f"{self.medium_step_size_z_mm} mm").grid(row=2, column=9, pady=3)
        ttk.Label(self.direction_frame, text=f"{self.small_step_size_z_mm} mm").grid(row=3, column=9, pady=3)
        ttk.Label(self.direction_frame, text=f"{self.nanodrive_step_size_um} um").grid(row=4, column=11, pady=3)

        if hasattr(self, 'sample_drop_panel_grid_pattern') and self.sample_drop_panel_grid_pattern is not None:
            self.sample_drop_panel_calculate_circle_pursa_coordinates()

    def create_widgets(self):

        main_frame = ttk.Labelframe(self, text='Stage Control', padding=5)
        main_frame.grid(row=0, column=0, columnspan=7, sticky="nsew", padx=5, pady=5)

        top_frame = ttk.Frame(main_frame)
        top_frame.grid(row=0, column=0, columnspan=7, sticky="nsew")

        # Reference buttons frame
        reference_frame = ttk.Labelframe(top_frame, text="Reference", padding=5)
        reference_frame.grid(row=0, column=0,  columnspan=4, rowspan=2, padx=5, sticky='nsew', pady=5)
        ttk.Button(reference_frame, text="Ref X", command=lambda: self.ref("X"), bootstyle="info-outline", width=8).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(reference_frame, text="Ref Y", command=lambda: self.ref("Y"), bootstyle="info-outline", width=8).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(reference_frame, text="Ref Z", command=lambda: self.ref("Z"), bootstyle="info-outline", width=8).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(reference_frame, text="Ref NP", command=lambda: self.ref("nanodrive"), bootstyle="info-outline", width=12).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(reference_frame, text="Ref All", command=lambda: self.ref("ALL"), bootstyle="info").grid(row=1, column=0, columnspan=4, sticky='nesw',padx=5, pady=5)

        # Home frame
        self.home_button = ttk.Button(top_frame, image=self.img_home_icon, command=self.prusa_go_to_home_position, bootstyle="info", state=DISABLED)
        self.home_button.grid(row=0, column=4, columnspan=2, sticky='ew',padx=5, pady=5)

        self.calibrate_home_position_button = ttk.Button(top_frame, text="Calibrate Home Position", command = self.update_home_position, bootstyle ="info-outline", state=DISABLED)
        self.calibrate_home_position_button.grid(row=1, column=4, columnspan=2, padx=5, pady=5)

        # Frame for movement
        movement_frame = ttk.Labelframe(main_frame, text='Move' , padding=5)
        movement_frame.grid(row=1, column=0, columnspan=6, sticky="ew", padx=5, pady=5)
        ttk.Label(movement_frame, text="X [mm]").grid(row=0, column=1)
        ttk.Label(movement_frame, text="Y [mm]").grid(row=0, column=2)
        ttk.Label(movement_frame, text="Z [mm]").grid(row=0, column=3)
        ttk.Label(movement_frame, text="ND Z [um]").grid(row=0, column=4)
        ttk.Label(movement_frame, text="Current").grid(row=1, column=0, padx=5, pady=5)
        self.current_x_entry = ttk.Label(movement_frame, text= 'N/A')
        self.current_y_entry = ttk.Label(movement_frame, text= 'N/A')
        self.current_z_entry = ttk.Label(movement_frame, text= 'N/A')
        self.current_nanodrive_z_entry = ttk.Label(movement_frame, text= 'N/A')
        self.current_x_entry.grid(row=1, column=1, padx=5, pady=5)
        self.current_y_entry.grid(row=1, column=2, padx=5, pady=5)
        self.current_z_entry.grid(row=1, column=3, padx=5, pady=5)
        self.current_nanodrive_z_entry.grid(row=1, column=4, padx=5, pady=5)

        ttk.Label(movement_frame, text="Move To").grid(row=2, column=0, padx=5, pady=5)
        self.target_x_entry = ttk.Entry(movement_frame, width=8)
        self.target_y_entry = ttk.Entry(movement_frame, width=8)
        self.target_z_entry = ttk.Entry(movement_frame, width=8)
        self.target_nanodrive_entry = ttk.Spinbox(movement_frame, from_=0, to=100, width=8)
        self.target_x_entry.grid(row=2, column=1, padx=5, pady=5)
        self.target_y_entry.grid(row=2, column=2, padx=5, pady=5)
        self.target_z_entry.grid(row=2, column=3, padx=5, pady=5)
        self.target_nanodrive_entry.grid(row=2, column=4, padx=5, pady=5)
        self.target_x_entry.insert(0, self.home_position[0])
        self.target_y_entry.insert(0, self.home_position[1])
        self.target_z_entry.insert(0, self.home_position[2])
        self.target_nanodrive_entry.insert(0, "0")
        self.go_button = ttk.Button(movement_frame, text="Go", command=self.move_both_prusa_nanodrive, bootstyle="info", state=DISABLED)
        self.go_button.grid(row=2, column=5, padx=10, pady=5)
        self.go_button.config(width=6)
        self.update_position_button = ttk.Button(movement_frame, text="Update", command=self.update_prusa_position, bootstyle="info", state=DISABLED)
        self.update_position_button.grid(row=1, column=5, padx=10, pady=5)
        self.update_position_button.config(width=6)

        # Speed slider
        self.slider_value = StringVar()
        self.slider_value.set(3000)
        self.speed_slider = ttk.Scale(
            movement_frame,
            from_=10,
            to=3000,
            orient=HORIZONTAL,
            bootstyle="info",
            variable=self.slider_value
        )
        self.speed_slider.grid(row=3, column=2, columnspan=2, pady=15)
        self.speed_slider.config(length=200)
        self.speed = self.speed_slider.get()
        ttk.Label(movement_frame, text="Speed").grid(row=3, column=0, padx=5, pady=5)
        ttk.Label(movement_frame, text="10 mm/s").grid(row=3, column=1, padx=5, pady=5)
        ttk.Label(movement_frame, text="3000 mm/s").grid(row=3, column=4, padx=5, pady=5)
        self.speed_slider.bind("<ButtonRelease-1>", lambda e: self.update_prusa_speed())

        # Frame for direction buttons
        self.direction_frame = ttk.Frame(movement_frame, padding=3)
        self.direction_frame.grid(row=5, column=0, columnspan=10, pady=3, padx=3)
        ttk.Label(self.direction_frame, text="Y").grid(row=0, column=4, pady=3)
        ttk.Label(self.direction_frame, text="X").grid(row=4, column=0, padx=3)
        ttk.Label(self.direction_frame, text="Z (up)").grid(row=0, column=10, pady=3)
        ttk.Label(self.direction_frame, text="Z (down)").grid(row=8, column=10, pady=3)
        ttk.Label(self.direction_frame, text="ND Z (up)").grid(row=0, column=11, pady=3)
        ttk.Label(self.direction_frame, text="ND Z (down)").grid(row=8, column=11, pady=3)
        ttk.Label(self.direction_frame, text="").grid(row=2, column=9,columnspan=2, padx=50)
        ttk.Label(self.direction_frame, text=f"{self.large_step_size_xy_mm} mm").grid(row=1, column=3, pady=3)
        ttk.Label(self.direction_frame, text=f"{self.medium_step_size_xy_mm} mm").grid(row=2, column=3, pady=3)
        ttk.Label(self.direction_frame, text=f"{self.small_step_size_xy_mm} mm").grid(row=3, column=3,  pady=3)

        ttk.Label(self.direction_frame, text=f"{self.large_step_size_z_mm} mm").grid(row=1, column=9, pady=3)
        ttk.Label(self.direction_frame, text=f"{self.medium_step_size_z_mm} mm").grid(row=2, column=9, pady=3)
        ttk.Label(self.direction_frame, text=f"{self.small_step_size_z_mm} mm").grid(row=3, column=9, pady=3)
        ttk.Label(self.direction_frame, text=f"{self.nanodrive_step_size_um} um").grid(row=4, column=11, pady=3)


        ttk.Button(self.direction_frame, image=self.img_up_low, command=lambda: self.move_prusa("BACK SMALL"), bootstyle="light").grid(row=3, column=4, pady=3)
        ttk.Button(self.direction_frame, image=self.img_up_medium, command=lambda: self.move_prusa("BACK MEDIUM"), bootstyle="secondary").grid(row=2, column=4, pady=3)
        ttk.Button(self.direction_frame, image=self.img_up_high, command=lambda: self.move_prusa("BACK LARGE"), bootstyle="dark").grid(row=1, column=4, pady=3)
        ttk.Button(self.direction_frame, image=self.img_left_low, command=lambda: self.move_prusa("LEFT SMALL"), bootstyle="light").grid(row=4, column=3, padx=3)
        ttk.Button(self.direction_frame, image=self.img_left_medium, command=lambda: self.move_prusa("LEFT MEDIUM"), bootstyle="secondary").grid(row=4, column=2, padx=3)
        ttk.Button(self.direction_frame, image=self.img_left_high, command=lambda: self.move_prusa("LEFT LARGE"), bootstyle="dark").grid(row=4, column=1, padx=3)
        ttk.Button(self.direction_frame, image=self.img_right_low, command=lambda: self.move_prusa("RIGHT SMALL"),bootstyle="light").grid(row=4, column=5, padx=3)
        ttk.Button(self.direction_frame, image=self.img_right_medium, command=lambda: self.move_prusa("RIGHT MEDIUM"), bootstyle="secondary").grid(row=4, column=6, padx=3)
        ttk.Button(self.direction_frame, image=self.img_right_high, command=lambda: self.move_prusa("RIGHT LARGE"), bootstyle="dark").grid(row=4, column=7, padx=3, pady=3)
        ttk.Button(self.direction_frame, image=self.img_down_low, command=lambda: self.move_prusa("FRONT SMALL"), bootstyle="light").grid(row=5, column=4, pady=3)
        ttk.Button(self.direction_frame, image=self.img_down_medium, command=lambda: self.move_prusa("FRONT MEDIUM"), bootstyle="secondary").grid(row=6, column=4, pady=3)
        ttk.Button(self.direction_frame, image=self.img_down_high, command=lambda: self.move_prusa("FRONT LARGE"), bootstyle="dark").grid(row=7, column=4, pady=3)
        ttk.Button(self.direction_frame, image=self.img_up_low, command=lambda: self.move_prusa("UP SMALL"), bootstyle="light").grid(row=3, column=10)
        ttk.Button(self.direction_frame, image=self.img_up_medium, command=lambda: self.move_prusa("UP MEDIUM"), bootstyle="secondary").grid(row=2, column=10, pady=3)
        ttk.Button(self.direction_frame, image=self.img_up_high, command=lambda: self.move_prusa("UP LARGE"), bootstyle="dark").grid(row=1, column=10, pady=3)
        ttk.Button(self.direction_frame, image=self.img_down_low, command=lambda: self.move_prusa("DOWN SMALL"), bootstyle="light").grid(row=5, column=10)
        ttk.Button(self.direction_frame, image=self.img_down_medium, command=lambda: self.move_prusa("DOWN MEDIUM"), bootstyle="secondary").grid(row=6, column=10, pady=3)
        ttk.Button(self.direction_frame, image=self.img_down_high, command=lambda: self.move_prusa("DOWN LARGE"), bootstyle="dark").grid(row=7, column=10, pady=3)
        ttk.Button(self.direction_frame, image=self.img_up_nano, command=lambda: self.move_nanodrive("UP"), bootstyle="light").grid(row=3, column=11)
        ttk.Button(self.direction_frame, image=self.img_down_nano, command=lambda: self.move_nanodrive("DOWN"), bootstyle="light").grid(row=5, column=11)

        # Combined frame for drop position and camera autofocus
        quartz_slides_and_sample_drops_frame = ttk.Labelframe(main_frame, text="Sample Drop & Camera Autofocus", padding=5)
        quartz_slides_and_sample_drops_frame.grid(row=3, column=0, columnspan=5, rowspan=6, sticky='nsew', padx=5, pady=5)

        # Sample drop position section
        self.sample_drop_panel_canvas = tk.Canvas(quartz_slides_and_sample_drops_frame, width=self.sample_drop_panel_total_width, height=self.sample_drop_panel_total_height, state=DISABLED)
        self.sample_drop_panel_canvas.grid(row=0, column=0, rowspan=5, padx=5, pady=5)
        self.sample_drop_panel_tooltip = tk.Label(quartz_slides_and_sample_drops_frame, text="", relief=tk.SOLID, bd=1, state=DISABLED)
        self.sample_drop_panel_create_circle_grid()

        # Camera autofocus section
        register_camera_focus_range_frame = ttk.Frame(quartz_slides_and_sample_drops_frame, padding=5)
        register_camera_focus_range_frame.grid(row=0, column=1, rowspan=5, columnspan=4, sticky='nsew', padx=5, pady=5)

        ttk.Label(register_camera_focus_range_frame, text="Upper Z Limit [mm]:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.upper_limit_entry = ttk.Entry(register_camera_focus_range_frame, width=3)
        self.upper_limit_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(register_camera_focus_range_frame, text="Lower Z Limit [mm]:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.lower_limit_entry = ttk.Entry(register_camera_focus_range_frame, width=3)
        self.lower_limit_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(register_camera_focus_range_frame, text="Widefield # of Steps:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.widefield_focus_steps_entry = ttk.Entry(register_camera_focus_range_frame, width=3)
        self.widefield_focus_steps_entry.grid(row=2, column=1, padx=5, pady=5)
        self.widefield_focus_steps_entry.insert(0, 20)

        self.register_widefield_focus_range_button = ttk.Button(register_camera_focus_range_frame, text="Register", command=self.register_prusa_focus_range, bootstyle="info-outline", state= DISABLED)
        self.register_widefield_focus_range_button.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky='news')

        self.widefield_focus_z_label = ttk.Label(register_camera_focus_range_frame, text="Widefield f Z[mm]: N/A", foreground="grey", anchor="center")
        self.widefield_focus_z_label.grid(row=2, column=2,  padx=5, pady=5, sticky="nsew")
        self.objective_focus_z_label = ttk.Label(register_camera_focus_range_frame, text="Obj f Z[mm]: N/A", foreground="grey", anchor="center")
        self.objective_focus_z_label.grid(row=3, column=2,  padx=5, pady=5, sticky="nsew")
        self.objective_focus_ND_z_label = ttk.Label(register_camera_focus_range_frame, text="Obj f ND Z[um]: N/A", foreground="grey", anchor="center")
        self.objective_focus_ND_z_label.grid(row=4, column=2, padx=5, pady=5, sticky="nsew")

        # quartz slide navigation section
        quartz_slide_navigation_frame = ttk.Frame(quartz_slides_and_sample_drops_frame, padding=5)
        quartz_slide_navigation_frame.grid(row=5, column=0, columnspan=5, sticky='nsew', padx=5, pady=5)
        self.previous_quartz_slide_button = ttk.Button(quartz_slide_navigation_frame, text="Previous Slide", command=self.move_to_previous_quartz_slide, bootstyle="info", state=DISABLED)
        self.previous_quartz_slide_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.quartz_slide_label = ttk.Label(quartz_slide_navigation_frame, text="Quartz Slide: 1")
        self.quartz_slide_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.next_quartz_slide_button = ttk.Button(quartz_slide_navigation_frame, text="Next Quartz Slide", command=self.move_to_next_quartz_slide, bootstyle="info", state=DISABLED)
        self.next_quartz_slide_button.grid(row=0, column=2, padx=5, pady=5, sticky="e")


    def run_in_thread(self, func, *args):
        thread = Thread(target=func, args=args, daemon=True)
        thread.start()

    def run_in_thread_with_callback(self, func, callback, *args):
        def wrapper():
            # Execute the target function and store the result
            result = func(*args)
            # Pass the result to the callback function
            callback(result)

        # Run the wrapper function in a separate thread
        thread = Thread(target=wrapper, daemon=True)
        thread.start()

    def prusa_go_to_home_position(self):
        self.prusa_go_to_xyz(z=self.safety_height) # raise to a safe height first
        self.prusa_go_to_xyz(x=float(self.home_position[0]), y=float(self.home_position[1]))
        self.prusa_go_to_xyz(z=float(self.home_position[2]))

    def move_both_prusa_nanodrive(self):
        print("Moving both the stage and the nanodrive")
        self.move_prusa("GO")
        self.move_nanodrive("GO")

    def move_nanodrive(self, direction):
        if direction == "UP":
            self.dispatch('move_nanodrive_by', -self.nanodrive_step_size_um)
        elif direction == "DOWN":
            self.dispatch('move_nanodrive_by', self.nanodrive_step_size_um)
        elif direction == "GO":
            self.dispatch('move_nanodrive_to', self.target_nanodrive_entry.get())

    def get_nanodrive_position(self):
        self.current_nanodrive_position = None
        self.dispatch('get_nanodrive_position')
        # wait until nanodrive position is updated
        while self.current_nanodrive_position is None:
            time.sleep(0.1)
        return self.current_nanodrive_position

    def move_prusa(self, option):
        if self.ser is None:
            print("Please connect to the printer first")
            return
        if option == "UP SMALL":
            self.prusa_go_by_xyz(z=self.small_step_size_z_mm)
        elif option == "UP MEDIUM":
            self.prusa_go_by_xyz(z=self.medium_step_size_z_mm)
        elif option == "UP LARGE":
            self.prusa_go_by_xyz(z=self.large_step_size_z_mm)

        elif option == "DOWN SMALL":
            self.prusa_go_by_xyz(z=-self.small_step_size_z_mm)
        elif option == "DOWN MEDIUM":
            self.prusa_go_by_xyz(z=-self.medium_step_size_z_mm)
        elif option == "DOWN LARGE":
            self.prusa_go_by_xyz(z=-self.large_step_size_z_mm)

        elif option == "LEFT SMALL":
            self.prusa_go_by_xyz(x=-self.small_step_size_xy_mm)
        elif option == "LEFT MEDIUM":
            self.prusa_go_by_xyz(x=-self.medium_step_size_xy_mm)
        elif option == "LEFT LARGE":
            self.prusa_go_by_xyz(x=-self.large_step_size_xy_mm)

        elif option == "RIGHT SMALL":
            self.prusa_go_by_xyz(x=self.small_step_size_xy_mm)
        elif option == "RIGHT MEDIUM":
            self.prusa_go_by_xyz(x=self.medium_step_size_xy_mm)
        elif option == "RIGHT LARGE":
            self.prusa_go_by_xyz(x=self.large_step_size_xy_mm)

        elif option == "BACK SMALL":
            self.prusa_go_by_xyz(y=self.small_step_size_xy_mm)
        elif option == "BACK MEDIUM":
            self.prusa_go_by_xyz(y=self.medium_step_size_xy_mm)
        elif option == "BACK LARGE":
            self.prusa_go_by_xyz(y=self.large_step_size_xy_mm)

        elif option == "FRONT SMALL":
            self.prusa_go_by_xyz(y=-self.small_step_size_xy_mm)
        elif option == "FRONT MEDIUM":
            self.prusa_go_by_xyz(y=-self.medium_step_size_xy_mm)
        elif option == "FRONT LARGE":
            self.prusa_go_by_xyz(y=-self.large_step_size_xy_mm)
        elif option == "GO":
            p2_x = self.target_x_entry.get()
            p2_y = self.target_y_entry.get()
            p2_z = self.target_z_entry.get()

            if p2_x and p2_y and p2_z:
                # # raise to a safe height
                # self.prusa_go_to_xyz(z=30)
                self.prusa_go_to_xyz(x=p2_x, y=p2_y, z=p2_z)
            else:
                print("Please enter all the coordinates")
                return
        else:
            print("Invalid option")
    
    def ref(self, option):
        self.run_in_thread(self._ref, option)

    def _ref(self, option):
        if self.ser is None:
            print("Please connect to the printer first")
            return
        if option == "X":
            self.ser.write(str.encode("G28 X F600\r\n"))
            self.prusa_x_referenced = True
        elif option == "Y":
            self.ser.write(str.encode("G28 Y F600\r\n"))
            self.prusa_y_referenced = True
        elif option == "Z":
            self.ser.write(str.encode("G28 Z F600\r\n"))
            self.prusa_z_referenced = True
        elif option == "nanodrive":
            self.dispatch('move_nanodrive_to', self.initial_nanodrive_position)
            self.nanodrive_min, self.nanodrive_max = self.dispatch('get_nanodrive_min_max')
            self.target_nanodrive_entry.config(from_=self.nanodrive_min, to=self.nanodrive_max)
        elif option == "ALL":
            # Disable software endstops to allow Z-axis movement before homing
            self.ser.write(b'M211 S0\n')
            time.sleep(1)

            # Move the Z-axis up by 10mm to avoid collisions
            self.ser.write(b'G91\n')         # Set to relative positioning
            self.ser.write(b'G1 Z20 F600\n')  # Move Z up 10mm at 600mm/min
            time.sleep(2)

            # Re-enable software endstops
            self.ser.write(b'M211 S1\n')

            time.sleep(1)

            self.ser.write(str.encode("G28 X Y Z F600\r\n"))

            self.prusa_x_referenced = True
            self.prusa_y_referenced = True
            self.prusa_z_referenced = True

            self.prusa_go_to_xyz(z=self.safety_height)
            self.update_prusa_position()
            self.dispatch('move_nanodrive_to', self.initial_nanodrive_position)
            results = self.dispatch('get_nanodrive_min_max')
            self.nanodrive_min, self.nanodrive_max = results[0]
            self.target_nanodrive_entry.config(from_=self.nanodrive_min, to=self.nanodrive_max)

            self.home_button.configure(state=NORMAL)
            self.calibrate_home_position_button.configure(state=NORMAL)
            self.sample_drop_panel_canvas.configure(state=NORMAL)
            self.sample_drop_panel_activate_circles()
            self.previous_quartz_slide_button.configure(state=NORMAL)
            self.next_quartz_slide_button.configure(state=NORMAL)
            self.sample_drop_panel_tooltip.configure(state=NORMAL)
            self.go_button.configure(state=NORMAL)
            self.update_position_button.configure(state=NORMAL)
            self.register_widefield_focus_range_button.configure(state=NORMAL)
            self.dispatch('activate_switch_view_button')
            self.dispatch('activate_protocol_module_state', 'stage_reference_is_setup')

        else:
            print("Invalid option")
            return

    def prusa_go_to_xyz(self, x=None, y=None, z=None):
        self.ser.write(str.encode("G90\r\n"))
        gcode = "G0"
        if x is not None:
            if self.prusa_x_referenced:
                gcode += f" X{x}"
        if y is not None:
            if self.prusa_y_referenced:
                gcode += f" Y{y}"
        if z is not None:
            if self.prusa_z_referenced:
                gcode += f" Z{z}"

        if x is not None or y is not None or z is not None:
            gcode += f" F{self.speed}\r\n"
            self.ser.write(str.encode(gcode))
            self.update_prusa_position()

    def prusa_go_by_xyz(self, x=None, y=None, z=None):
        self.ser.write(str.encode("G91\r\n"))
        gcode = "G0"
        if x is not None:
            if self.prusa_x_referenced:
                gcode += f" X{x}"
        if y is not None:
            if self.prusa_y_referenced:
                gcode += f" Y{y}"
        if z is not None:
            if self.prusa_z_referenced:
                gcode += f" Z{z}"

        if x is not None or y is not None or z is not None:
            gcode += f" F{self.speed}\r\n"
            self.ser.write(str.encode(gcode))
            self.update_prusa_position()

    def find_prusa_com_ports(self):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if port.description == "Original Prusa i3 MK3 (COM3)":
                return port.name

    def wait_for_prusa_process_complete(self, process_name, critiria):
        """Waits for a signal from the printer that homing is complete."""
        while True:
            line = self.ser.readline().decode('utf-8').strip()
            print(line)
            if critiria in line:
                return line
            elif line:
                print(f"Received during {process_name}: {line}")
                
    def update_prusa_speed(self):
        self.speed = self.speed_slider.get()
        print(f"Speed of stage is updated to: {self.speed} mm/s")

    def connect_prusa_device(self):
        self.port = self.find_prusa_com_ports()
        self.ser = serial.Serial(self.port, 115200, timeout=1000000, write_timeout=1000000)
        # time.sleep(3)
        printer_status = self.is_prusa_on()
        if printer_status:
            print("Connected to the PRUSA " + self.ser.name)
            # Keep the PRUSA engaged by sending periodic keep-alive commands
            def keep_prusa_alive():
                while True:
                    try:
                        self.ser.write(b'M105\n')  # Send a keep-alive command
                        time.sleep(5)  # Wait for 5 seconds before sending the next command
                    except Exception as e:
                        print(f"Error keeping PRUSA alive: {e}")
                        break

            self.run_in_thread(keep_prusa_alive)
        else:
            self.ser.close()
            print("Couldn't connect to the PRUSA")

    def is_prusa_on(self):
        try:
            self.ser.flushInput()
            self.ser.flushOutput()
            self.ser.write(b'M105\n')
            print('write M105')
            time.sleep(3)
            response = self.ser.read_all().decode('utf-8')
            print(response)

            if 'start\necho:' in response:
                return True
            else:
                return False
        except Exception as e:
            print(f"Unexpected Error: {e}")
            return False

    def get_current_prusa_position(self, dim = 'XYZ'):
        self.ser.flushInput()
        self.ser.flushOutput()
        self.ser.write(b'M114\n')
        line = self.wait_for_prusa_process_complete(process_name = "get_current_position", critiria='X')
        if dim == 'XYZ':
            X = np.round(float(line.split(' ')[0].split(':')[1]),2)
            Y = np.round(float(line.split(' ')[1].split(':')[1]),2)
            Z = np.round(float(line.split(' ')[2].split(':')[1]),2)
            return X, Y, Z
        elif dim == 'X':
            X = np.round(float(line.split(' ')[0].split(':')[1]),2)
            return X
        elif dim == 'Y':
            Y = np.round(float(line.split(' ')[1].split(':')[1]),2)
            return Y
        elif dim == 'Z':
            Z = np.round(float(line.split(' ')[2].split(':')[1]),2)
            return Z

    def update_home_position(self):
        self.home_position = self.get_current_prusa_position('XYZ')

        # recalculate the circle positions
        self.sample_drop_panel_calculate_circle_pursa_coordinates()
        self.dispatch('update_home_position', self.home_position)
        print("Home position updated")

    def handle_switch_view(self, view):
        if view == "TO_OBJECTIVE":
            # send g code to move the stage to the right
            print("Switching to objective view")
            self.prusa_go_by_xyz(x=self.calibration_from_widefield_to_objective_x, y=self.calibration_from_widefield_to_objective_y)
            self.prusa_go_by_xyz(z=self.calibration_from_widefield_to_objective_z)

        elif view == "TO_WIDE":
            # send g code to move the stage to the left
            print("Switching to wide view")
            self.prusa_go_by_xyz(z=-self.calibration_from_widefield_to_objective_z)
            self.prusa_go_by_xyz(x=-self.calibration_from_widefield_to_objective_x, y=-self.calibration_from_widefield_to_objective_y)
        else:
            print("Error Happened", view)

        return True

    def handle_update_nanodrive_position(self, nanodrive_position):
        self.current_nanodrive_position = nanodrive_position
        self.current_nanodrive_z_entry.config(text=nanodrive_position)
        return None

    def handle_test_sampling_points(self, view_to_inspect_in, relative_distance_for_sampling_points):
        # Display a confirmation dialog
        response = Messagebox.show_question(
            title="Confirm Operation",
            message="Is the objective at a safe height?",
            alert=True,
            buttons=["Yes", "No"],
            bootstyle="danger"
        )
        if response == "Yes":  # If the user clicks "Yes"
            print("Operation started!")
            self.run_in_thread(self._handle_test_sampling_points, view_to_inspect_in, relative_distance_for_sampling_points)
        else:  # If the user clicks "No"
            print("Operation canceled.")
        return None

    def _handle_test_sampling_points(self, view_to_inspect_in, relative_distance_for_sampling_points):
        # calculate the absolute position of the sampling points
        sampling_position_x, sampling_position_y = self.calculate_sampling_point_position_in_absolute_coordinate(view_to_inspect_in, relative_distance_for_sampling_points)

        # move to the test sample spot one by one
        for i in range(len(sampling_position_x)):
            print(f'Moving to the test sample spot {sampling_position_x[i]}, {sampling_position_y[i]}')
            self.prusa_go_to_xyz(x=sampling_position_x[i], y=sampling_position_y[i])
            time.sleep(0.1)
        
        # Move back to the center of the captured frame
        if view_to_inspect_in == "OBJECTIVE":
            self.prusa_go_to_xyz(x=self.capture_frame_center_objective_x, y=self.capture_frame_center_objective_y)
        elif view_to_inspect_in == "WIDEFIELD":
            self.prusa_go_to_xyz(x=self.capture_frame_center_widefield_x, y=self.capture_frame_center_widefield_y)
        else:
            raise ValueError("The center of the frame is not captured yet")
        
    def handle_move_to_a_single_sampling_point_during_acquisition(self, sample_drop_row_column, relative_distance_for_sampling_point, ):
        row, column = sample_drop_row_column[0], sample_drop_row_column[1]
        sample_drop_panel_current_sample_pursa_coordinates = self.sample_drop_panel_circle_pursa_coordinates_in_objective[(row, column)]
        fliped_relative_distance_for_sampling_point_y = relative_distance_for_sampling_point[1] * -1
        sample_position_x = np.round(relative_distance_for_sampling_point[0]/1000 + sample_drop_panel_current_sample_pursa_coordinates[0], 2)
        sample_position_y = np.round(fliped_relative_distance_for_sampling_point_y/1000 + sample_drop_panel_current_sample_pursa_coordinates[1], 2)
        self.prusa_go_to_xyz(x=sample_position_x, y=sample_position_y)
        
        while True:
            current_x = float(self.get_current_prusa_position('XYZ')[0])
            current_y = float(self.get_current_prusa_position('XYZ')[1])
            if current_x == sample_position_x and current_y == sample_position_y:
                break

        self.dispatch('task_completed')
        return True

    def calculate_sampling_point_position_in_absolute_coordinate(self, view_to_inspect_in, relative_distance_for_sampling_points):
        if view_to_inspect_in == "OBJECTIVE":
            if self.capture_frame_center_objective_x and self.capture_frame_center_objective_y:
                # printer's y axis is flipped
                filped_sampling_position_y = relative_distance_for_sampling_points[1] * -1

                sampling_position_x = relative_distance_for_sampling_points[0] / 1000 + self.capture_frame_center_objective_x
                sampling_position_y = filped_sampling_position_y / 1000 + self.capture_frame_center_objective_y
        elif view_to_inspect_in == "WIDEFIELD":
            if self.capture_frame_center_widefield_x and self.capture_frame_center_widefield_y:
                # printer's y axis is flipped
                filped_sampling_position_y = relative_distance_for_sampling_points[1] * -1

                sampling_position_x = relative_distance_for_sampling_points[0] / 1000 + self.capture_frame_center_widefield_x
                sampling_position_y = filped_sampling_position_y / 1000 + self.capture_frame_center_widefield_y

                print('the sampling_position_x: ', sampling_position_x)
                print('the sampling_position_y: ', sampling_position_y)
        else:
            raise ValueError("The center of the frame is not captured yet")

        # round up the sampling position to 2 decimal places
        sampling_position_x = np.round(sampling_position_x, 2)
        sampling_position_y = np.round(sampling_position_y, 2)

        return sampling_position_x, sampling_position_y

    def handle_updated_captured_frame_center(self, field_of_view):
        print("Captured the center of the frame")
        if field_of_view == 'WIDEFIELD':
            self.capture_frame_center_widefield_x = float(self.get_current_prusa_position('XYZ')[0])
            self.capture_frame_center_widefield_y = float(self.get_current_prusa_position('XYZ')[1])

            print(f"Captured the center of the frame in the physical space: {self.capture_frame_center_widefield_x}, {self.capture_frame_center_widefield_y}")

            self.capture_frame_center_objective_x = self.capture_frame_center_widefield_x + self.calibration_from_widefield_to_objective_x
            self.capture_frame_center_objective_y = self.capture_frame_center_widefield_y + self.calibration_from_widefield_to_objective_y
        elif field_of_view == 'OBJECTIVE':
            self.capture_frame_center_objective_x = float(self.get_current_prusa_position('XYZ')[0])
            self.capture_frame_center_objective_y = float(self.get_current_prusa_position('XYZ')[1])

            self.capture_frame_center_widefield_x = self.capture_frame_center_objective_x - self.calibration_from_widefield_to_objective_x
            self.capture_frame_center_widefield_y = self.capture_frame_center_objective_y - self.calibration_from_widefield_to_objective_y
        return None

    def handle_move_stage_to_target_sample_drop_during_aquisition(self, view_to_move_in, sample_drop_row_column):
        if view_to_move_in == "OBJECTIVE":
            self.prusa_go_to_xyz(z=self.objective_focus_prusa_upper_limit)
            print('moving to the upper limit')
            # check if the stage is at the target position
            while True:
                current_z_position = np.round(float(self.get_current_prusa_position('Z')),2)
                print('stuck when trying to move to the upper limit')
                print(f"current z position: {current_z_position}")
                print(f"upper limit: {self.objective_focus_prusa_upper_limit}")
                if current_z_position == self.objective_focus_prusa_upper_limit:
                    break

            # re-position nanodrive to the initial position
            self.dispatch('move_nanodrive_to', self.initial_nanodrive_position)
            print('moving nanodrive to the initial position')

            # move to the target sample drop position
            self.sample_drop_panel_move_stage('OBJECTIVE', sample_drop_row_column[0], sample_drop_row_column[1])

            self.dispatch('task_completed')
            return None
        
        elif view_to_move_in == "WIDEFIELD":
            self.prusa_go_to_xyz(z=self.widefield_focus_prusa_upper_limit)
            print('moving to the upper limit')
            # check if the stage is at the target position
            while True:
                current_z_position = np.round(float(self.get_current_prusa_position('Z')),2)
                print('stuck when trying to move to the upper limit')
                print(f"current z position: {current_z_position}")
                print(f"upper limit: {self.widefield_focus_prusa_upper_limit}")
                if current_z_position == self.widefield_focus_prusa_upper_limit:
                    break

            # re-position nanodrive to the initial position
            self.dispatch('move_nanodrive_to', self.initial_nanodrive_position)
            print('moving nanodrive to the initial position')

            # move to the target sample drop position
            self.sample_drop_panel_move_stage('WIDEFIELD', sample_drop_row_column[0], sample_drop_row_column[1])

            self.dispatch('task_completed')
            return None

    def handle_reposition_stage_and_nanodrive_in_objective_view_during_acquisition(self):
        # move the stage to the best focus position in widefield view
        self.prusa_go_to_xyz(z=self.best_widefield_focus_prusa_position)

        # check if the stage is at the target position
        while True:
            current_position = self.get_current_prusa_position('Z')
            if current_position == self.best_widefield_focus_prusa_position:
                break

        # re-position nanodrive to the initial position
        self.dispatch('move_nanodrive_to', self.initial_nanodrive_position)
        self.dispatch('task_completed')
        return True

    def handle_focus_widefield_camera(self):
        self.run_in_thread(self._handle_focus_widefield_camera)

    def _handle_focus_widefield_camera(self):
        print("Focusing the widefield camera")
        # move nanodrive to the middle position
        self.dispatch('move_nanodrive_to', self.initial_nanodrive_position)

        # wait until the nanodrive position is at the target position
        while True:
            current_nanodrive_position = self.get_nanodrive_position()
            print(f"current nanodrive position: {current_nanodrive_position}")
            if current_nanodrive_position == self.initial_nanodrive_position:
                break

        if self.widefield_focus_prusa_lower_limit and self.widefield_focus_prusa_upper_limit:

            self.best_widefield_focus_prusa_position = self.focus_prusa(self.widefield_focus_prusa_lower_limit, self.widefield_focus_prusa_upper_limit, self.widefield_focus_prusa_step_number)
            print(f"Widefield pursa focus [{self.widefield_focus_prusa_lower_limit}, {self.widefield_focus_prusa_upper_limit}] with {self.widefield_focus_prusa_step_number} steps")
            print(f"Best focus position for widefield camera: {self.best_widefield_focus_prusa_position} mm")
            self.widefield_focus_z_label.config(text=f"Widefield Focus Z[mm]: {self.best_widefield_focus_prusa_position}")
            return True
        else:
            print("Please interpolate the focus range first")
            return False
        
    def focus_prusa(self, lower_limit, upper_limit, step_number):
        focus_score = []
        self.focus_score_at_current_z_position = None

        z_range = np.linspace(float(upper_limit), float(lower_limit), int(step_number))
        # round up the z range to 2 decimal places
        z_range = np.round(z_range, 2)

        print(f"Pursa z focus within [{lower_limit}, {upper_limit}]mm with {step_number} steps")
        print(f"z positions: {z_range}")

        for z in z_range:
            self.prusa_go_to_xyz(z=z)

            # wait until the stage position is at the target position
            while True:
                current_z_position = float(self.get_current_prusa_position('Z'))
                if current_z_position == z:
                    break

            # wait until the robot is stable
            time.sleep(self.prusa_movement_stabilization_time)

            self.dispatch('calculate_focus_score')

            # wait until the score is different from the previous one
            while True:
                if self.focus_score_at_current_z_position:
                    if len(focus_score) > 0:
                        if self.focus_score_at_current_z_position != focus_score[-1]:
                            break
                    else:
                        break

            focus_score.append(self.focus_score_at_current_z_position)

        best_prusa_focus_position = self.find_best_focus_position(focus_score, z_range)
        print(f"Best focus position for prusa: {best_prusa_focus_position}")

        self.prusa_go_to_xyz(z=best_prusa_focus_position)
        time.sleep(self.prusa_movement_stabilization_time)
        return best_prusa_focus_position

    def find_best_focus_position(self, focus_score, z_axis_range):
        k = np.argmax(focus_score)
        best_focus_position = z_axis_range[k]
        return best_focus_position

    def handle_focus_widefield_camera_during_acquisition(self):
        def on_focus_complete(result):
            if result:
                self.dispatch('task_completed')
            else:
                self.dispatch('abort_aquisition')
        self.run_in_thread_with_callback(self._handle_focus_widefield_camera, on_focus_complete)

    def handle_update_focus_score(self, focus_score):
        self.focus_score_at_current_z_position = focus_score
        return None

    def handle_move_prusa_to(self, z_position):
        if self.prusa_z_referenced:
            self.prusa_go_to_xyz(z=z_position)
            # wait until the stage position is at the target position
            while True:
                current_z_position = float(self.get_current_prusa_position('Z'))
                if current_z_position == z_position:
                    break
            return True
        else:
            print("Prusa Z-axis is not referenced")
            return False

    def update_prusa_position(self):
        current_prusa_position = self.get_current_prusa_position('XYZ')

        if self.prusa_x_referenced:
            current_prusa_position_x = str(current_prusa_position[0])
            self.current_x_entry.config(text=current_prusa_position_x)
        
        if self.prusa_y_referenced:
            current_prusa_position_y = str(current_prusa_position[1])
            self.current_y_entry.config(text=current_prusa_position_y)

        if self.prusa_z_referenced:
            current_prusa_position_z = str(current_prusa_position[2])
            self.current_z_entry.config(text=current_prusa_position_z)

    def register_prusa_focus_range(self):
        try:
            self.widefield_focus_prusa_upper_limit = float(self.upper_limit_entry.get())
            self.widefield_focus_prusa_lower_limit = float(self.lower_limit_entry.get())
            self.widefield_focus_prusa_step_number = int(self.widefield_focus_steps_entry.get())

            self.objective_focus_prusa_upper_limit = round(self.widefield_focus_prusa_upper_limit + self.calibration_from_widefield_to_objective_z, 2)
            self.objective_focus_prusa_lower_limit = round(self.widefield_focus_prusa_lower_limit + self.calibration_from_widefield_to_objective_z, 2)  

            print(f"Prusa focus range for widefield camera registered: {self.widefield_focus_prusa_lower_limit} - {self.widefield_focus_prusa_upper_limit} mm")
            print(f'The offset from widefield to objective Z: {self.calibration_from_widefield_to_objective_z} mm')
            print(f'Prusa focus range for objective camera registered: {self.objective_focus_prusa_lower_limit} - {self.objective_focus_prusa_upper_limit} mm')
 
            self.dispatch('activate_camera_autofocus_button')
            self.dispatch('activate_protocol_module_state','focus_range_is_setup')
            self.dispatch('update_prusa_focus_range_for_wasatch', self.objective_focus_prusa_lower_limit, self.objective_focus_prusa_upper_limit)

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for the focus range.")

    def sample_drop_panel_create_circle_grid(self): 
        skip_positions = [(0, 0), (3, 0), (0, 4), (3, 4)]  # Positions to skip

        for col_index, num_circles in enumerate(self.sample_drop_panel_grid_pattern):
            vertical_offset = (max(self.sample_drop_panel_grid_pattern) - num_circles) * self.sample_drop_panel_spacing // 2
            for row_index in range(num_circles):
                true_row_index = row_index + vertical_offset // self.sample_drop_panel_spacing
                
                # Skip the specified corner positions
                if (true_row_index, col_index) in skip_positions:
                    continue
                
                x = col_index * self.sample_drop_panel_spacing + self.sample_drop_panel_spacing // 2
                y = vertical_offset + row_index * self.sample_drop_panel_spacing + self.sample_drop_panel_spacing // 2
                
                circle = self.sample_drop_panel_canvas.create_oval(
                    x - self.sample_drop_panel_circle_radius, y - self.sample_drop_panel_circle_radius,
                    x + self.sample_drop_panel_circle_radius, y + self.sample_drop_panel_circle_radius,
                    fill='grey', outline='black', tags=f"circle_{true_row_index}_{col_index}"
                )
                
                # Bind click event to each circle
                self.sample_drop_panel_canvas.tag_bind(circle, '<Button-1>',
                    lambda event, row=true_row_index, col=col_index: self.sample_drop_panel_move_stage('WIDEFIELD', row, col))
                # Bind mouse over and leave events for hover effect and tooltip
                self.sample_drop_panel_canvas.tag_bind(circle, '<Enter>',
                    lambda event, row=true_row_index, col=col_index: self.sample_drop_panel_on_hover(event, row, col))
                self.sample_drop_panel_canvas.tag_bind(circle, '<Leave>', self.sample_drop_panel_on_leave)

    def sample_drop_panel_calculate_circle_pursa_coordinates(self):
        home_x, home_y = self.home_position[0], self.home_position[1]

        for col_index, num_circles in enumerate(self.sample_drop_panel_grid_pattern):
            for row_index in range(num_circles):
                prusa_coordinate_x = np.round(home_x + self.offset_home_to_p1_x + col_index * self.interval_between_drop_x, 2)
                prusa_coordinate_y = np.round(home_y + self.offset_home_to_p1_y - row_index * self.interval_between_drop_y, 2)
                self.sample_drop_panel_circle_pursa_coordinates_in_widefield[(row_index, col_index)] = (prusa_coordinate_x, prusa_coordinate_y)

                prusa_coordinate_x = np.round(prusa_coordinate_x + self.calibration_from_widefield_to_objective_x, 2)
                prusa_coordinate_y = np.round(prusa_coordinate_y + self.calibration_from_widefield_to_objective_y, 2)
                self.sample_drop_panel_circle_pursa_coordinates_in_objective[(row_index, col_index)] = (prusa_coordinate_x, prusa_coordinate_y)

    def sample_drop_panel_move_stage(self, view_to_move_in, row, col):

            if not self.prusa_x_referenced or not self.prusa_y_referenced or not self.prusa_z_referenced:
                print("Stage is not referenced. Action disabled.")
                return  # Disable action if stage is not referenced

            # Reset the previous circle color
            if self.sample_drop_panel_current_sample_row_column:
                prev_row, prev_col = self.sample_drop_panel_current_sample_row_column
                self.sample_drop_panel_canvas.itemconfig(f"circle_{prev_row}_{prev_col}", fill='skyblue')

            # Highlight the selected circle
            self.sample_drop_panel_canvas.itemconfig(f"circle_{row}_{col}", fill='yellow')
            self.sample_drop_panel_current_sample_row_column = (row, col)
            print(f"Moving XY stage to position: Row {row}, Column {col}")
            if view_to_move_in == "OBJECTIVE":
                sample_drop_panel_current_sample_pursa_coordinates = self.sample_drop_panel_circle_pursa_coordinates_in_objective[(row, col)]
            elif view_to_move_in == "WIDEFIELD":
                sample_drop_panel_current_sample_pursa_coordinates = self.sample_drop_panel_circle_pursa_coordinates_in_widefield[(row, col)]
            print(f"Pursa coordinates: {sample_drop_panel_current_sample_pursa_coordinates}")
            self.prusa_go_to_xyz(x=sample_drop_panel_current_sample_pursa_coordinates[0], y=sample_drop_panel_current_sample_pursa_coordinates[1])

    def sample_drop_panel_on_hover(self, event, row, col):
        # Prevent changing color if hovering over the current position
        if self.sample_drop_panel_current_sample_row_column != (row, col):
            self.sample_drop_panel_canvas.itemconfig(f"circle_{row}_{col}", fill='lightgreen')
        self.sample_drop_panel_tooltip.config(text="Move to this drop")
        self.sample_drop_panel_tooltip.place(x=event.x_root - self.master.winfo_rootx() + 10, y=event.y_root - self.master.winfo_rooty() + 10)

    def sample_drop_panel_on_leave(self, event):
        for col_index, num_circles in enumerate(self.sample_drop_panel_grid_pattern):
            for row_index in range(num_circles):
                if self.sample_drop_panel_current_sample_row_column != (row_index, col_index):
                    self.sample_drop_panel_canvas.itemconfig(f"circle_{row_index}_{col_index}", fill='skyblue')
        self.sample_drop_panel_tooltip.place_forget()

    def sample_drop_panel_activate_circles(self):
        if self.prusa_x_referenced and self.prusa_y_referenced and self.prusa_z_referenced:
            for col_index, num_circles in enumerate(self.sample_drop_panel_grid_pattern):
                for row_index in range(num_circles):
                    self.sample_drop_panel_canvas.itemconfig(f"circle_{row_index}_{col_index}", fill='skyblue')
        else:
            print("Stage is not referenced. Action disabled.")

    def move_to_previous_quartz_slide(self):
        pass

    def move_to_next_quartz_slide(self):
        pass

if __name__ == '__main__':
    root = ttk.Window()
    root.style.theme_use('noodlepy')
    app = StageControlModule(root)
    app.grid(row=0, column=0, sticky="nsew")
    root.mainloop()