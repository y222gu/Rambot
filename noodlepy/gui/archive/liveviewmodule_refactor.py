import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import numpy as np
from threading import Thread
import queue



import tkinter as tk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk, ImageDraw
import typing
import threading
import queue
import ttkbootstrap as ttk
import numpy as np
from noodlepy.gui.edgedetectorSAM import EdgeDetectorSAM
import os
from threading import Thread

try:
    from noodlepy.utils.windows_setup import configure_path
    print("Configuring path...")
    configure_path()
except ImportError:
    configure_path = None

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame
from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK


class LiveCanvas(tk.Canvas):
    '''Canvas widget for displaying live images from a camera. 
    Resizes the image to fit the canvas and draws a crossline in the middle of the image.
    Converts the image to a PhotoImage object for display.
    
    Args:
        parent: Parent widget
        image_queue: Queue object for storing images
        width: Width of the canvas
        height: Height of the canvas
        flip: Boolean for flipping the image
    '''
    def __init__(self, parent, image_queue, width, height, refresh_rate, flip=False):
        self.image_queue = image_queue
        self._image_width = width
        self._image_height = height
        self._image = None
        self.tk_image = None
        self.flip = flip
        self.refresh_rate = refresh_rate #ms
        tk.Canvas.__init__(self, parent, width=width, height=height)
        self.grid(row=0, column=0, sticky='nsew')
        self._get_image()

    def _get_image(self):
        try:
            self._image = self.image_queue.get_nowait()
            # the image is mirrored, so we flip it
            if self.flip:
                self._image = self._image.transpose(Image.FLIP_LEFT_RIGHT)

            self._resize()
            self._draw_crossline()
            self._display_image()
        except queue.Empty:
            pass
        self.after(self.refresh_rate, self._get_image)

    def _resize(self, event=None):
        if self._image:
            self.tk_image = self._image.resize((self._image_width, self._image_height), Image.LANCZOS)

    def _draw_crossline(self):
        # Draw middle dash lines on the image
        draw = ImageDraw.Draw(self.tk_image)
        draw.line((0, self.tk_image.height / 2, self.tk_image.width, self.tk_image.height / 2), fill='black', width=1)
        draw.line((self.tk_image.width / 2, 0, self.tk_image.width / 2, self.tk_image.height), fill='black', width=1)
        del draw

    def _display_image(self):
        self.tk_image = ImageTk.PhotoImage(self.tk_image)
        self.create_image(0, 0, image=self.tk_image, anchor='nw')

class ImageAcquisitionThread(threading.Thread):
    '''Thread for acquiring images from a camera in PIL format and putting them into a queue.
    
    Args:
        camera: TLCamera object
    '''
    def __init__(self, camera):
        super().__init__()
        camera.exposure_time_us = 40000
        self._camera = camera
        self._previous_timestamp = 0
        self._is_color = False
        self._bit_depth = camera.bit_depth
        self._camera.image_poll_timeout_ms = 0
        self._image_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

    def get_output_queue(self):
        return self._image_queue

    def stop(self):
        self._stop_event.set()

    def _get_image(self, frame):
        # Process the image frame to convert it to a PIL image
        scaled_image = frame.image_buffer >> (self._bit_depth - 8)
        image8bit = np.asarray(scaled_image, np.uint8)
        processed_image = image8bit.squeeze()
        return Image.fromarray(processed_image)

    def run(self):
        while not self._stop_event.is_set():
            try:
                frame = self._camera.get_pending_frame_or_null()
                if frame is not None:
                    pil_image = self._get_image(frame)
                    self._image_queue.put(pil_image)

            except queue.Full:
                pass
            except Exception as error:
                print(f"Encountered error: {error}, image acquisition will stop.")
                break
        
        print("Image acquisition has stopped")


class CameraController:
    def __init__(self, sdk):
        self.sdk = sdk
        self.widefield_camera = None
        self.smallfield_camera = None

    def setup_cameras(self):
        camera_list = self.sdk.discover_available_cameras()
        if not camera_list:
            raise Exception("No cameras found")

        if '14938' not in camera_list or '14628' not in camera_list:
            raise Exception("Make sure both cameras are connected")

        self.widefield_camera = self._initialize_camera('14938')
        self.smallfield_camera = self._initialize_camera('14628')

    def _initialize_camera(self, camera_id):
        camera = self.sdk.open_camera(camera_id)
        camera.frames_per_trigger_zero_for_unlimited = 0
        camera.arm(2)
        camera.issue_software_trigger()
        return camera

    def close_cameras(self):
        self.widefield_camera.dispose()
        self.smallfield_camera.dispose()
        self.sdk.dispose()

class EdgeDetectionController:
    def __init__(self, edge_detector, captured_image_label):
        self.edge_detector = edge_detector
        self.captured_image_label = captured_image_label

    def perform_edge_detection(self, detection_type):
        mask_method = {
            "auto": self.edge_detector.auto_mask_generate,
            "point": self.edge_detector.point_prompt_mask_generate,
            "box": self.edge_detector.box_prompt_mask_generate
        }
        masked_image = mask_method[detection_type]()
        self.update_displayed_image(masked_image)

    def update_displayed_image(self, image):
        image_display = ImageTk.PhotoImage(image)
        self.captured_image_label.configure(image=image_display)
        self.captured_image_label.image = image_display

class SamplingProfileController:
    def __init__(self, sampling_method_var, config_entries):
        self.sampling_method_var = sampling_method_var
        self.config_entries = config_entries

    def handle_sampling_method_selection(self, event=None):
        selected_method = self.sampling_method_var.get()
        config = self.config_entries
        if selected_method == "Random":
            self._toggle_state(config, {"num_points": "normal"}, {"rows": "disabled", "cols": "disabled", "rings": "disabled", "interval": "disabled"})
        elif selected_method == "Grid":
            self._toggle_state(config, {"rows": "normal", "cols": "normal"}, {"num_points": "disabled", "rings": "disabled", "interval": "disabled"})
        elif selected_method == "Rings":
            self._toggle_state(config, {"num_points": "normal", "rings": "normal", "interval": "normal"}, {"rows": "disabled", "cols": "disabled"})

    def _toggle_state(self, widgets, enabled, disabled):
        for key in enabled:
            widgets[key].configure(state=enabled[key])
        for key in disabled:
            widgets[key].configure(state=disabled[key])

class LiveViewUI(tk.Frame):
    def __init__(self, parent, camera_controller, edge_controller, sampling_controller, icons, width=288, height=216):
        super().__init__(parent)
        self.camera_controller = camera_controller
        self.edge_controller = edge_controller
        self.sampling_controller = sampling_controller
        self.icons = icons
        self.width = width
        self.height = height
        self.create_widgets()

    def create_widgets(self):
        self.create_live_frame()
        self.create_capture_frame()

    def create_live_frame(self):
        live_frame = ttk.Labelframe(self, text="Wide FOV", width=self.width, padding=5)
        live_frame.grid(row=0, column=0, sticky='nsew', pady=5, padx=5)

        self.camera_widget = LiveCanvas(parent=live_frame, image_queue=self.active_camera_thread.get_output_queue(), width=self.width, height=self.height, refresh_rate=10, flip=False)
        self.camera_widget.grid(row=0, column=0, columnspan=2, sticky='nsew')
        
        switch_view_button = ttk.Button(live_frame, image=self.icons['exchange'], command=lambda: self.switch_view('TO_SMALL'), style='info')
        switch_view_button.grid(row=1, column=0, sticky='nsew', pady=5, padx=5)
        capture_button = ttk.Button(live_frame, image=self.icons['camera'], command=self.capture_frame, style='info')
        capture_button.grid(row=1, column=1, sticky='nsew', pady=5, padx=5)

    def create_capture_frame(self):
        capture_frame = ttk.Labelframe(self, text="Captured Frame", width=self.width, padding=5)
        capture_frame.grid(row=2, column=0, sticky='nsew', pady=5, padx=5)

        self.initial_image = Image.fromarray(np.zeros((self.height, self.width), dtype=np.uint8))
        self.image_to_display = ImageTk.PhotoImage(self.initial_image)
        self.captured_image_label = ttk.Label(capture_frame, text="No Frame Captured", image=self.image_to_display, compound='center', foreground='white')
        self.captured_image_label.grid(row=0, column=0, sticky='nsew')
        self.captured_image_label.image = self.image_to_display

        sampling_frame = self.create_sampling_controls(capture_frame)
        sampling_frame.grid(row=1, column=0, sticky='nsew', pady=5, padx=5)

    def create_sampling_controls(self, parent):
        sampling_frame = ttk.Labelframe(parent, text="Sampling Method", padding=5)
        
        self.sampling_method_var = tk.StringVar(value="Random")
        self.random_radio = ttk.Radiobutton(sampling_frame, text="Random", variable=self.sampling_method_var, value="Random", command=self.sampling_controller.handle_sampling_method_selection)
        self.grid_radio = ttk.Radiobutton(sampling_frame, text="Grid", variable=self.sampling_method_var, value="Grid", command=self.sampling_controller.handle_sampling_method_selection)
        self.rings_radio = ttk.Radiobutton(sampling_frame, text="Rings", variable=self.sampling_method_var, value="Rings", command=self.sampling_controller.handle_sampling_method_selection)

        self.random_radio.grid(row=0, column=0)
        self.grid_radio.grid(row=0, column=1)
        self.rings_radio.grid(row=0, column=2)
        
        self.config_entries = {
            "num_points": ttk.Entry(sampling_frame, width=5, state="disabled"),
            "rows": ttk.Entry(sampling_frame, width=5, state="disabled"),
            "cols": ttk.Entry(sampling_frame, width=5, state="disabled"),
            "rings": ttk.Entry(sampling_frame, width=5, state="disabled"),
            "interval": ttk.Entry(sampling_frame, width=5, state="disabled")
        }
        for i, (key, entry) in enumerate(self.config_entries.items()):
            entry.grid(row=1, column=i)

        return sampling_frame

class LiveViewApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Live View")
        self.style = ttk.Style()
        self.style.theme_use('superhero')

        self.sdk = TLCameraSDK()
        self.camera_controller = CameraController(self.sdk)
        self.camera_controller.setup_cameras()

        edge_detector = EdgeDetectorSAM(self.camera_controller.widefield_camera)
        edge_controller = EdgeDetectionController(edge_detector, self.captured_image_label)

        sampling_controller = SamplingProfileController(self.sampling_method_var, self.config_entries)

        icons = {
            "camera": self.load_icon("assets/camera_icon.png"),
            "exchange": self.load_icon("assets/exchange_icon.png")
        }
        
        self.ui = LiveViewUI(self, self.camera_controller, edge_controller, sampling_controller, icons)
        self.ui.grid(row=0, column=0, sticky='nsew')
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_icon(self, path):
        icon = Image.open(path)
        icon = icon.resize((30, 30))
        return ImageTk.PhotoImage(icon)

    def on_closing(self):
        self.camera_controller.close_cameras()
        self.destroy()

if __name__ == "__main__":
    app = LiveViewApp()
    app.mainloop()
