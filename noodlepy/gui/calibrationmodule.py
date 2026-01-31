import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import StringVar

class CalibrationModule():
    def __init__(self, root):
        # create a new window on top of the root window
        self.window = ttk.Toplevel(root)
        self.window.style.theme_use('noodlepy')
        self.window.title("Calibration")
        self.window.geometry("400x300")
        self.create_widgets()

    def create_widgets(self):

        # create a frame to hold the widgets
        frame = ttk.Frame(self.window)
        frame.grid(row=0, column=0, sticky="nsew")

        # create a label widget
        label = ttk.Label(frame, text="Calibration")
        label.grid(row=0, column=0, sticky="nsew")

        # create a button widget
        button = ttk.Button(frame, text="Calibrate", style="info", command=self.calibrate)
        button.grid(row=1, column=0, sticky="nsew")

        # create a button widget
        button = ttk.Button(frame, text="Close", style="info", command=self.window.destroy)
        button.grid(row=2, column=0, sticky="nsew")



    def calibrate(self):
        print("Calibrating...")
