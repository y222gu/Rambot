import os
import time
import serial
import serial.tools.list_ports
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
from tkinter import StringVar, messagebox
import tkinter as tk
import numpy as np
from threading import Thread
from noodlepy.gui.publisher_subscriber import Subscriber, Publisher


# Helper class to manage images
class ImageManager:
    def __init__(self, assets_dir):
        self.assets_dir = assets_dir
        self.images = self.load_images()

    def load_images(self):
        images = {
            "small_step": Image.open(os.path.join(self.assets_dir, "small_step.png")).resize((20, 20)),
            "medium_step": Image.open(os.path.join(self.assets_dir, "medium_step.png")).resize((20, 20)),
            "large_step": Image.open(os.path.join(self.assets_dir, "large_step.png")).resize((20, 20)),
            "circle_black": Image.open(os.path.join(self.assets_dir, "circle-black.png")).resize((20, 20)),
            "nanodrive_icon": Image.open(os.path.join(self.assets_dir, "nanodrive_arrow_1.png")).resize((20, 20)),
            "home_icon": Image.open(os.path.join(self.assets_dir, "home_icon.png")).resize((40, 40)),
        }
        return images

    def get_tk_images(self):
        return {
            "up_low": ImageTk.PhotoImage(self.images["small_step"].rotate(90)),
            "up_medium": ImageTk.PhotoImage(self.images["medium_step"].rotate(-90)),
            "up_high": ImageTk.PhotoImage(self.images["large_step"].rotate(90)),
            "down_low": ImageTk.PhotoImage(self.images["small_step"].rotate(-90)),
            "down_medium": ImageTk.PhotoImage(self.images["medium_step"].rotate(90)),
            "down_high": ImageTk.PhotoImage(self.images["large_step"].rotate(-90)),
            "left_low": ImageTk.PhotoImage(self.images["small_step"].rotate(180)),
            "left_medium": ImageTk.PhotoImage(self.images["medium_step"]),
            "left_high": ImageTk.PhotoImage(self.images["large_step"].rotate(180)),
            "right_low": ImageTk.PhotoImage(self.images["small_step"]),
            "right_medium": ImageTk.PhotoImage(self.images["medium_step"].rotate(180)),
            "right_high": ImageTk.PhotoImage(self.images["large_step"]),
            "up_nano": ImageTk.PhotoImage(self.images["nanodrive_icon"].rotate(180)),
            "down_nano": ImageTk.PhotoImage(self.images["nanodrive_icon"]),
            "home_icon": ImageTk.PhotoImage(self.images["home_icon"]),
        }


# Handles communication with the Prusa device
class PrusaDeviceManager:
    def __init__(self):
        self.ser = None
        self.port = None
        self.speed = 300

    def find_prusa_com_ports(self):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if port.description == "Original Prusa i3 MK3 (COM3)":
                return port.name
        return None

    def connect_device(self):
        self.port = self.find_prusa_com_ports()
        if not self.port:
            print("No Prusa device found.")
            return False

        self.ser = serial.Serial(self.port, 115200)
        if self.is_device_ready():
            print(f"Connected to Prusa device on port {self.port}")
            return True
        else:
            self.ser.close()
            print("Failed to connect to Prusa device.")
            return False

    def is_device_ready(self):
        try:
            self.ser.flushInput()
            self.ser.flushOutput()
            self.ser.write(b'M105\n')
            time.sleep(2)
            response = self.ser.read_all().decode('utf-8')
            return 'start\necho:' in response
        except Exception as e:
            print(f"Error checking device readiness: {e}")
            return False

    def send_gcode(self, gcode):
        if self.ser:
            self.ser.write(str.encode(gcode + "\n"))
            print(f"Sent G-code: {gcode}")
        else:
            print("Prusa device not connected.")

    def get_current_position(self):
        if not self.ser:
            return None
        self.ser.flushInput()
        self.ser.flushOutput()
        self.ser.write(b'M114\n')
        response = self.ser.readline().decode('utf-8').strip()
        print(f"Position response: {response}")
        if response.startswith("X:"):
            parts = response.split()
            x = float(parts[0][2:])
            y = float(parts[1][2:])
            z = float(parts[2][2:])
            return x, y, z
        return None

    def set_speed(self, speed):
        self.speed = speed
        print(f"Prusa speed updated to {speed} mm/s")


# Handles UI for movement controls
class MovementControlUI(ttk.Frame):
    def __init__(self, parent, image_manager, prusa_manager):
        super().__init__(parent)
        self.image_manager = image_manager
        self.prusa_manager = prusa_manager
        self.current_position_labels = {}
        self.target_entries = {}
        self.speed_slider = None
        self.create_ui()

    def create_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        movement_frame = ttk.Labelframe(self, text="Move", padding=5)
        movement_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ttk.Label(movement_frame, text="X [mm]").grid(row=0, column=1)
        ttk.Label(movement_frame, text="Y [mm]").grid(row=0, column=2)
        ttk.Label(movement_frame, text="Z [mm]").grid(row=0, column=3)

        self.current_position_labels = {
            "X": ttk.Label(movement_frame, text="N/A"),
            "Y": ttk.Label(movement_frame, text="N/A"),
            "Z": ttk.Label(movement_frame, text="N/A"),
        }
        self.current_position_labels["X"].grid(row=1, column=1, padx=5, pady=5)
        self.current_position_labels["Y"].grid(row=1, column=2, padx=5, pady=5)
        self.current_position_labels["Z"].grid(row=1, column=3, padx=5, pady=5)

        ttk.Label(movement_frame, text="Target").grid(row=2, column=0, padx=5, pady=5)
        self.target_entries = {
            "X": ttk.Entry(movement_frame, width=8),
            "Y": ttk.Entry(movement_frame, width=8),
            "Z": ttk.Entry(movement_frame, width=8),
        }
        self.target_entries["X"].grid(row=2, column=1, padx=5, pady=5)
        self.target_entries["Y"].grid(row=2, column=2, padx=5, pady=5)
        self.target_entries["Z"].grid(row=2, column=3, padx=5, pady=5)

        self.target_entries["X"].insert(0, "0")
        self.target_entries["Y"].insert(0, "0")
        self.target_entries["Z"].insert(0, "0")

        go_button = ttk.Button(movement_frame, text="Go", command=self.move_to_target)
        go_button.grid(row=3, column=0, columnspan=4, padx=10, pady=10)

        self.speed_slider = ttk.Scale(
            movement_frame, from_=10, to=3000, orient="horizontal", command=self.update_speed
        )
        self.speed_slider.grid(row=4, column=0, columnspan=4, padx=5, pady=5)

    def move_to_target(self):
        x = self.target_entries["X"].get()
        y = self.target_entries["Y"].get()
        z = self.target_entries["Z"].get()
        gcode = f"G0 X{x} Y{y} Z{z} F{self.prusa_manager.speed}"
        self.prusa_manager.send_gcode(gcode)

    def update_speed(self, value):
        self.prusa_manager.set_speed(int(float(value)))


# Main class
class StageControlModule(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.prusa_manager = PrusaDeviceManager()
        self.image_manager = ImageManager(os.path.join(os.getcwd(), "noodlepy", "assets"))
        self.movement_ui = MovementControlUI(self, self.image_manager, self.prusa_manager)
        self.setup_ui()

    def setup_ui(self):
        self.movement_ui.grid(row=0, column=0, sticky="nsew")
        ttk.Button(self, text="Connect to Prusa", command=self.connect_prusa).grid(row=1, column=0, padx=10, pady=10)

    def connect_prusa(self):
        if self.prusa_manager.connect_device():
            print("Prusa connected successfully")
        else:
            print("Failed to connect to Prusa")


# Entry point
if __name__ == "__main__":
    root = ttk.Window()
    app = StageControlModule(root)
    app.grid(row=0, column=0, sticky="nsew")
    root.mainloop()
