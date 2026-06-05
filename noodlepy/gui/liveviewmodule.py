import tkinter as tk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageOps
import typing
import threading
import queue
import ttkbootstrap as ttk
import numpy as np
from noodlepy.gui.edgedetectorSAM import EdgeDetectorSAM
from noodlepy.gui.edgedetectorUnet import EdgeDetectorUnet
import os
from threading import Thread
from noodlepy.gui.publisher_subscriber import Publisher
import time
import cv2
from skimage import filters

try:
    from noodlepy.utils.windows_setup import configure_path
    print("Configuring path...")
    configure_path()
except ImportError:
    configure_path = None

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame
from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK
from PIL import ImageEnhance


class LiveCanvas(tk.Canvas):
    '''Canvas widget for displaying live images from a camera. 
    Resizes the image to fit the canvas and draws a crossline in the middle of the image.
    Converts the image to a PhotoImage object for display.
    
    Args:
        parent: Parent widget
        image_queue: Queue object for storing images
        width: Width of the canvas
        height: Height of the canvas

    '''
    def __init__(self, parent, image_queue, canvas_width, canvas_height, refresh_rate):
        self.image_queue = image_queue
        self._image_width = canvas_width
        self._image_height = canvas_height
        self._image = None
        self.tk_image = None
        self.refresh_rate = refresh_rate #ms

        tk.Canvas.__init__(self, parent, width=canvas_width, height=canvas_height)
        self.grid(row=0, column=0, sticky='nsew')
        self._get_image()

    def _get_image(self):
        try:
            self._image = self.image_queue.get_nowait()
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
        flip: Boolean for flipping the image
    '''
    def __init__(self, camera, flip, crop, rotate, center_x, center_y, crop_width, crop_height):
        super().__init__()
        camera.exposure_time_us = 40000
        self._camera = camera
        self._previous_timestamp = 0
        self._bit_depth = camera.bit_depth
        self._camera.image_poll_timeout_ms = 0
        self._image_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._flip = flip
        self._crop = crop
        self._rotate = rotate
        self._center_x = center_x
        self._center_y = center_y
        self._crop_width = crop_width
        self._crop_height = crop_height


                # setup color processing if necessary
        if self._camera.camera_sensor_type != SENSOR_TYPE.BAYER:
            # Sensor type is not compatible with the color processing library
            self._is_color = False
        else:
            self._mono_to_color_sdk = MonoToColorProcessorSDK()
            self._image_width = self._camera.image_width_pixels
            self._image_height = self._camera.image_height_pixels
            self._mono_to_color_processor = self._mono_to_color_sdk.create_mono_to_color_processor(
                SENSOR_TYPE.BAYER,
                self._camera.color_filter_array_phase,
                self._camera.get_color_correction_matrix(),
                self._camera.get_default_white_balance_matrix(),
                self._camera.bit_depth
            )
            self._is_color = True

    def get_output_queue(self):
        return self._image_queue

    def stop(self):
        self._stop_event.set()

    def _get_image(self, frame):
        # Process the image frame to convert it to a PIL image
        scaled_image = frame.image_buffer >> (self._bit_depth - 8)
        image8bit = np.asarray(scaled_image, np.uint8)
        processed_image = image8bit.squeeze()
        image = Image.fromarray(processed_image)
        if self._flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if self._rotate:
            image = image.rotate(-90)
        
        if self._crop:
            image = image.crop((self._center_x - self._crop_width//2, self._center_y - self._crop_height//2, self._center_x + self._crop_width//2, self._center_y + self._crop_height//2))
        
        # Autoscale
        image = ImageOps.autocontrast(image)

        return image
    
    def _get_color_image(self, frame):
        # type: (Frame) -> Image
        # verify the image size
        width = frame.image_buffer.shape[1]
        height = frame.image_buffer.shape[0]
        if (width != self._image_width) or (height != self._image_height):
            self._image_width = width
            self._image_height = height
            print("Image dimension change detected, image acquisition thread was updated")
        # color the image. transform_to_24 will scale to 8 bits per channel
        color_image_data = self._mono_to_color_processor.transform_to_24(frame.image_buffer,
                                                                         self._image_width,
                                                                         self._image_height)
        color_image_data = color_image_data.reshape(self._image_height, self._image_width, 3)

        # convert the image to a single channel image
        color_image_data = color_image_data.mean(axis=2).astype('uint8')
        # return PIL Image object
        image = Image.fromarray(color_image_data, mode="L") # use mode='RGB' for color images
        if self._flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if self._rotate:
            image = image.rotate(-90)

        if self._crop:
            image = image.crop((self._center_x - self._crop_width//2, self._center_y - self._crop_height//2, self._center_x + self._crop_width//2, self._center_y + self._crop_height//2))

        return image
    
    def run(self):
        while not self._stop_event.is_set():
            try:
                frame = self._camera.get_pending_frame_or_null()
                if frame is not None:
                    if self._is_color:
                        pil_image = self._get_color_image(frame)
                    else:
                        pil_image = self._get_image(frame)
                    self._image_queue.put(pil_image, block=False)

            except queue.Full:
                pass
            except Exception as error:
                print(f"Encountered error: {error}, image acquisition will stop.")
                break

        print("Image acquisition has stopped")
        if self._is_color:
            self._mono_to_color_processor.dispose()
            self._mono_to_color_sdk.dispose()

class CameraManger():
    def __init__(self):
        self.sdk = TLCameraSDK()
        self.camera_serial_number_list = self.sdk.discover_available_cameras()

    def open_camera(self, serial_number, flip=False, crop=False, rotate=False, center_x=None, center_y=None, crop_width=None, crop_height=None):
        if serial_number not in self.camera_serial_number_list:
            raise ValueError(f"Camera with serial number {serial_number} not found")
        camera = self.sdk.open_camera(serial_number)
        image_acquisition_thread = ImageAcquisitionThread(camera, flip, crop, rotate, center_x, center_y, crop_width, crop_height)

        camera.frames_per_trigger_zero_for_unlimited = 0
        camera.arm(2)
        camera.issue_software_trigger()
        image_acquisition_thread.start()
        return camera, image_acquisition_thread

    def get_camera_list(self):
        return self.camera_serial_number_list
    
    def close_camera(self, camera, image_acquisition_thread):
        image_acquisition_thread.stop()
        image_acquisition_thread.join()

class LiveViewModule(Publisher, tk.Frame):
    def __init__(self, parent, settings):
        ttk.Frame.__init__(self, parent)  # Initialize ttk.Frame and Publisher
        Publisher.__init__(self, ['switch_view', 
                                  'test_all_sampling_points', 
                                  'update_captured_frame_center', 
                                  'focus_widefield_camera', 
                                  'focus_objective_camera', 
                                  'update_focus_score', 
                                  'task_completed',
                                  'abort_aquisition',
                                  'update_sampling_points_to_protocol_module'])
        
        self.name = 'LiveViewModule_publisher'

        self.settings = settings

        self.apply_settings()

        # # the canvas width and height that the image will be resized to
        # self.width = 500 
        # self.height = 375
        # # calibrated size in object plane per pixel on 11/11/2025
        # self.x_pixel_size_objective_camera = 0.191 #um
        # self.y_pixel_size_objective_camera = 0.191 #um
        # self.x_pixel_size_widefield_camera = 2.309 #um
        # self.y_pixel_size_widefield_camera = 2.309 #um

        # # crop the objective camera image to the center
        # self.crop_center_x = 798 # calibrated on 11/11/2025
        # self.crop_center_y = 452 # calibrated on 11/11/2025

        # self.crop_width=500
        # self.crop_height = 375

        # # for the rings sampling method
        # self.offset_from_the_edge = 40

        self.camera_icon = self.load_icon(os.path.join(os.getcwd(), "noodlepy","assets","camera_icon.png"))
        self.exchange_icon = self.load_icon(os.path.join(os.getcwd(), "noodlepy","assets","exchange_icon.png"))
        self.focus_icon = self.load_icon(os.path.join(os.getcwd(), "noodlepy","assets","focus_icon.png"))
        self.camera_manager = CameraManger()

        self.objective_field_camera, self.objective_field_camera_thread = self.camera_manager.open_camera(self.objective_camera_serial, flip=self.flip_objective_camera, crop=self.crop_objective_camera, rotate=self.rotate_objective_camera, center_x=self.objective_crop_center_x, center_y=self.objective_crop_center_y, crop_width=self.objective_crop_width, crop_height=self.objective_crop_height)
        self.widefield_camera, self.widefield_camera_thread = self.camera_manager.open_camera(self.widefield_camera_serial) 
        self.active_camera_thread = self.widefield_camera_thread
        self.current_live_view = 'WIDEFIELD'
        self.captured_view = None
        self.create_widgets()

    def apply_settings(self):
        s = self.settings

        # canvas size
        self.width = s.width
        self.height = s.height

        # pixel sizes
        self.x_pixel_size_objective_camera = s.x_pixel_size_objective_camera
        self.y_pixel_size_objective_camera = s.y_pixel_size_objective_camera
        self.x_pixel_size_widefield_camera = s.x_pixel_size_widefield_camera
        self.y_pixel_size_widefield_camera = s.y_pixel_size_widefield_camera

        # crop
        self.objective_crop_center_x = s.objective_crop_center_x
        self.objective_crop_center_y = s.objective_crop_center_y
        self.objective_crop_width = s.objective_crop_width
        self.objective_crop_height = s.objective_crop_height

        # sampling
        self.sampling_offset_from_the_edge = s.sampling_offset_from_the_edge

        # camera
        self.widefield_camera_serial = s.widefield_camera_serial
        self.objective_camera_serial = s.objective_camera_serial
        self.flip_objective_camera = s.flip_objective_camera
        self.crop_objective_camera = s.crop_objective_camera
        self.rotate_objective_camera = s.rotate_objective_camera

    def handle_update_settings(self, new_settings):
        self.settings = new_settings.liveview
        self.apply_settings()
        # self.camera_manager.close_camera(self.widefield_camera, self.widefield_camera_thread)
        # self.camera_manager.close_camera(self.objective_field_camera, self.objective_field_camera_thread)
        # self.widefield_camera, self.widefield_camera_thread = self.camera_manager.open_camera(self.widefield_camera_serial)
        # self.objective_field_camera, self.objective_field_camera_thread = self.camera_manager.open_camera(self.objective_camera_serial, flip=self.flip_objective_camera, crop=self.crop_objective_camera, rotate=self.rotate_objective_camera, center_x=self.objective_crop_center_x, center_y=self.objective_crop_center_y, crop_width=self.objective_crop_width, crop_height=self.objective_crop_height)
        
        # if self.current_live_view == 'WIDEFIELD':
        #     self.active_camera_thread = self.widefield_camera_thread
        # else:
        #     self.active_camera_thread = self.objective_field_camera_thread
        # self.camera_widget.update_image_queue(self.active_camera_thread.get_output_queue())

    def create_widgets(self):
        self.live_frame = ttk.Labelframe(self, text="Live Camera (Widefield)", width=self.width, padding=5)
        self.live_frame.grid(row=0, column=0, sticky='nsew', pady=5, padx=5)

        self.camera_widget = LiveCanvas(parent=self.live_frame, image_queue=self.active_camera_thread.get_output_queue(), canvas_width=self.width, canvas_height=self.height, refresh_rate=10)
        self.camera_widget.grid(row=0, column=0, columnspan=3, sticky='nsew')
        self.switch_view_button = ttk.Button(self.live_frame, image = self.exchange_icon, command= lambda: self.on_switch_view_button_clicked('TO_OBJECTIVE'), style='info', state=DISABLED)
        self.switch_view_button.grid(row=1, column=0, columnspan=1, sticky='nsew', pady=5, padx=5)
        self.camera_autofocus_button = ttk.Button(self.live_frame, image=self.focus_icon, command= self.focus_camera, style='info', state=DISABLED)
        self.camera_autofocus_button.grid(row=1, column=1, columnspan=1, sticky='nsew', pady=5, padx=5)
        capture_button = ttk.Button(self.live_frame, image=self.camera_icon, command= self.capture_frame, style='info')
        capture_button.grid(row=1, column=2, columnspan=1, sticky='nsew', pady=5, padx=5)

        capture_frame = ttk.Labelframe(self, text="Captured Frame", width=self.width, padding=5)
        capture_frame.grid(row=2, column=0, sticky='nsew', pady=5, padx=5)
        self.initial_image = Image.fromarray(np.zeros((self.height, self.width), dtype=np.uint8))
        self.image_to_display = ImageTk.PhotoImage(self.initial_image)
        self.captured_image_label = ttk.Label(capture_frame, text="No Frame Captured", image= self.image_to_display, compound='center', foreground='white')
        self.captured_image_label.grid(row=0, column=0, sticky='nsew')
        self.captured_image_label.image = self.image_to_display

        # selecting the sampling method
        self.edge_detection_point_button = ttk.Button(capture_frame, text="Point Detection", command=lambda: self.edge_detection('point'), state=DISABLED, style='info')
        self.edge_detection_point_button.grid(row=1, column=0, sticky='nsew', pady=5, padx=5)
        sampling_method_frame = ttk.Labelframe(capture_frame, text="Sampling Method", padding=5)
        sampling_method_frame.grid(row=2, column=0, sticky='nsew', pady=5, padx=5)
        sampling_method_frame.columnconfigure(0, weight=1, uniform="sampling")
        sampling_method_frame.columnconfigure(1, weight=1, uniform="sampling")
        sampling_method_frame.columnconfigure(2, weight=1, uniform="sampling")
        sampling_method_frame.columnconfigure(3, weight=1, uniform="sampling")

        self.sampling_method_var = tk.StringVar()
        self.sampling_method_var.set("Rings")

        self.random_radio = ttk.Radiobutton(sampling_method_frame, text="Random", variable=self.sampling_method_var, value="Random", command=lambda: self.on_sampling_method_selected('Random'), style='info', state=NORMAL)
        self.random_radio.grid(row=0, column=0, sticky='nesw', padx=5, pady=5)
        self.rings_radio = ttk.Radiobutton(sampling_method_frame, text="Rings", variable=self.sampling_method_var, value="Rings", command=lambda: self.on_sampling_method_selected('Rings'), style='info', state=NORMAL)
        self.rings_radio.grid(row=0, column=1, sticky='nesw', padx=5, pady=5)
        self.grid_radio = ttk.Radiobutton(sampling_method_frame, text="Grid", variable=self.sampling_method_var, value="Grid", command=lambda: self.on_sampling_method_selected('Grid'), style='info', state=NORMAL)
        self.grid_radio.grid(row=0, column=2, sticky='nesw', padx=5, pady=5)
        self.line_radio = ttk.Radiobutton(sampling_method_frame, text="Line", variable=self.sampling_method_var, value="Line", command=lambda: self.on_sampling_method_selected('Line'), style='info', state=NORMAL)
        self.line_radio.grid(row=0, column=3, sticky='nesw', padx=5, pady=5)

        # action buttons (column 3, aligned with the two label/entry row pairs)
        self.create_sampling_profile_buttons= ttk.Button(sampling_method_frame, text="Create", command=self.on_create_button_clicked, state=DISABLED, style='info')
        self.create_sampling_profile_buttons.grid(row=1, column=3, rowspan=2, sticky='nsew', pady=5, padx=5)
        self.test_sampling_points_button = ttk.Button(sampling_method_frame, text="Test", command= self.on_test_button_clicked, state=DISABLED, style='info')
        self.test_sampling_points_button.grid(row=3, column=3, rowspan=2, sticky='nsew', pady=5, padx=5)

        # --- first label/entry row pair (rows 1-2) ---
        # entry for the number of sampling points
        self.number_of_sampling_points_label = ttk.Label(sampling_method_frame, text="# of Points")
        self.number_of_sampling_points_label.grid(row=1, column=0, sticky='nesw', padx=5)
        self.number_of_sampling_points_entry = ttk.Entry(sampling_method_frame, width=5)
        self.number_of_sampling_points_entry.grid(row=2, column=0, sticky='ew', padx=5)
        self.number_of_sampling_points_entry.insert(0, "10")

        # entry for the number of rings
        self.rings_number_label = ttk.Label(sampling_method_frame, text="# of Rings")
        self.rings_number_label.grid(row=1, column=1, sticky='nesw', padx=5)
        self.rings_number_entry = ttk.Entry(sampling_method_frame, width=5)
        self.rings_number_entry.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        self.rings_number_entry.insert(0, "1")

        # entry for the interval between rings
        self.interval_label = ttk.Label(sampling_method_frame, text="Interval [px]")
        self.interval_label.grid(row=1, column=2, sticky='nesw', padx=5)
        self.interval_entry = ttk.Entry(sampling_method_frame, width=5)
        self.interval_entry.grid(row=2, column=2, sticky='ew', padx=5, pady=5)
        self.interval_entry.insert(0, "30")

        # --- second label/entry row pair (rows 3-4) ---
        # entry for the number of rows and columns for the grid
        self.row_number_label = ttk.Label(sampling_method_frame, text="# of Rows")
        self.row_number_label.grid(row=3, column=0, sticky='nesw', padx=5)
        self.row_number_entry = ttk.Entry(sampling_method_frame, width=5)
        self.row_number_entry.grid(row=4, column=0, sticky='ew', padx=5, pady=5)
        self.row_number_entry.insert(0, "5")
        self.row_number_entry.configure(state=DISABLED)

        self.column_number_label = ttk.Label(sampling_method_frame, text="# of Columns")
        self.column_number_label.grid(row=3, column=1, sticky='nesw', padx=5)
        self.column_number_entry = ttk.Entry(sampling_method_frame, width=5)
        self.column_number_entry.grid(row=4, column=1, sticky='ew', padx=5, pady=5)
        self.column_number_entry.insert(0, "5")
        self.column_number_entry.configure(state=DISABLED)

        # entry for the buffer distance (in pixels) of the outermost points from the
        # detected sample edge, placed directly below the interval entry. Positive keeps
        # points inside the edge, negative lets the points spill out of the segmented mask.
        self.buffer_label = ttk.Label(sampling_method_frame, text="Buffer [px]")
        self.buffer_label.grid(row=3, column=2, sticky='nesw', padx=5)
        self.buffer_entry = ttk.Entry(sampling_method_frame, width=5)
        self.buffer_entry.grid(row=4, column=2, sticky='ew', padx=5, pady=5)
        self.buffer_entry.insert(0, str(self.sampling_offset_from_the_edge))

    def load_icon(self, icon_path):
        icon = Image.open(icon_path)
        icon = icon.resize((40, 40))
        icon = ImageTk.PhotoImage(icon)
        return icon

    def run_in_thread(self, func, *args):
        thread = Thread(target=func, args=args, daemon=True)
        thread.start()

    def edge_detection(self, detection_type):
        # self.edgedetector = EdgeDetectorSAM(self.captured_image)
        self.edgedetector = EdgeDetectorUnet(self.captured_image)
        
        # if detection_type == "auto":
        #     masked_image = self.edgedetector.auto_mask_generate()
        if detection_type == "point":
            masked_image =self.edgedetector.point_prompt_mask_generate()
        # elif detection_type == "box":   
        #     masked_image = self.edgedetector.box_prompt_mask_generate()
        print("Edge detection button clicked")

        self.masked_image = masked_image
        resized_masked_image = self.masked_image.resize((self.width, self.height), Image.LANCZOS)
        self.image_to_display = ImageTk.PhotoImage(resized_masked_image)
        self.captured_image_label.configure(image=self.image_to_display)
        self.captured_image_label.image = self.image_to_display

        self.create_sampling_profile_buttons.configure(state=NORMAL)
        self.test_sampling_points_button.configure(state=NORMAL)
        self.random_radio.configure(state=NORMAL)
        self.rings_radio.configure(state=NORMAL)
        self.grid_radio.configure(state=NORMAL)
        self.line_radio.configure(state=NORMAL)
        self.number_of_sampling_points_entry.configure(state=NORMAL)


    def on_sampling_method_selected(self, event):
        selected_method = self.sampling_method_var.get()
        if selected_method == "Random":
            self.row_number_entry.configure(state=DISABLED)
            self.column_number_entry.configure(state=DISABLED)
            self.number_of_sampling_points_entry.configure(state=NORMAL)
            self.rings_number_entry.configure(state=DISABLED)
            self.interval_entry.configure(state=DISABLED)

        elif selected_method == "Grid":
            self.row_number_entry.configure(state=NORMAL)
            self.column_number_entry.configure(state=NORMAL)
            self.number_of_sampling_points_entry.configure(state=DISABLED)
            self.rings_number_entry.configure(state=DISABLED)
            self.interval_entry.configure(state=DISABLED)

        elif selected_method == "Rings":
            self.row_number_entry.configure(state=DISABLED)
            self.column_number_entry.configure(state=DISABLED)
            self.number_of_sampling_points_entry.configure(state=NORMAL)
            self.rings_number_entry.configure(state=NORMAL)
            self.interval_entry.configure(state=NORMAL)

        elif selected_method == "Line":
            self.row_number_entry.configure(state=DISABLED)
            self.column_number_entry.configure(state=DISABLED)
            self.number_of_sampling_points_entry.configure(state=NORMAL)
            self.rings_number_entry.configure(state=DISABLED)
            self.interval_entry.configure(state=DISABLED)


    def optimize_the_sequence_of_sampling_points(self, x, y, shape):
        if shape == 'rings':
            # Compute the centroid
            centroid_x = np.mean(x)
            centroid_y = np.mean(y)

            # Compute the angles of each point w.r.t. the centroid
            angles = np.arctan2(y - centroid_y, x - centroid_x)

            # Sort indices by angle in counter-clockwise order
            sorted_indices = np.argsort(angles)

            # Sort x and y arrays based on the sorted indices
            sorted_x = x[sorted_indices]
            sorted_y = y[sorted_indices]

        elif shape == 'grid':
            # Stack x and y into a single array of points
            points = np.stack([x, y], axis=1)
            
            # Sort by x (left to right), then by y (bottom to top)
            sorted_indices = np.lexsort((y, x))
            sorted_points = points[sorted_indices]

            # Unstack the sorted points
            sorted_x = sorted_points[:, 0]
            sorted_y = sorted_points[:, 1]

        else:
            sorted_x = x
            sorted_y = y

        return sorted_x, sorted_y

    def on_create_button_clicked(self):
        selected_method = self.sampling_method_var.get()
        try:
            buffer = int(float(self.buffer_entry.get()))
        except (ValueError, tk.TclError):
            buffer = self.sampling_offset_from_the_edge

        if selected_method == "Random":
            num_points = int(self.number_of_sampling_points_entry.get())
            sampled_mask_image, self.sampling_position_x, self.sampling_position_y, self.coordinates_order = self.edgedetector.generate_sampling_points(shape='random', num_points=num_points, buffer=buffer)

        elif selected_method == "Grid":
            row_number = int(self.row_number_entry.get())
            col_number = int(self.column_number_entry.get())
            sampled_mask_image, self.sampling_position_x, self.sampling_position_y, self.coordinates_order = self.edgedetector.generate_sampling_points(shape='grid', row_number=row_number, col_number=col_number, buffer=buffer)

        elif selected_method == "Rings":
            num_points = int(self.number_of_sampling_points_entry.get())
            num_rings = int(self.rings_number_entry.get())
            interval = int(self.interval_entry.get())
            sampled_mask_image, self.sampling_position_x, self.sampling_position_y, self.coordinates_order = self.edgedetector.generate_sampling_points(shape='rings', num_points=num_points, num_rings=num_rings, interval=interval, buffer=buffer)

        elif selected_method == "Line":
            num_points = int(self.number_of_sampling_points_entry.get())
            sampled_mask_image, self.sampling_position_x, self.sampling_position_y, self.coordinates_order = self.edgedetector.generate_sampling_points(shape='line', num_points=num_points, buffer=buffer)

        self.sampled_mask_image = sampled_mask_image
        # resize the image to fit the canvas
        self.sampled_mask_image = self.sampled_mask_image.resize((self.width, self.height), Image.LANCZOS)
        self.image_to_display = ImageTk.PhotoImage(self.sampled_mask_image)
        self.captured_image_label.configure(image=self.image_to_display)
        self.captured_image_label.image = self.image_to_display
        print("Sampling points generated")

        # Save the sampled mask image (add index if file exists)
        base_name = 'sampled_mask_image.png'
        if os.path.exists(base_name):
            idx = 1
            while os.path.exists(f'sampled_mask_image_{idx}.png'):
                idx += 1
            filename = f'sampled_mask_image_{idx}.png'
        else:
            filename = base_name
        sampled_mask_image.save(filename)

        # self.test_sampling_points_button.configure(state=NORMAL)
        return True

    def capture_frame(self):
        try:
                self.captured_image = self.active_camera_thread.get_output_queue().get()
                # Adjust gamma to enhance contrast
                gamma = 2  # Lower gamma to darken the image
                self.captured_image = self.captured_image.point(lambda x: 255 * ((x / 255) ** gamma))

                print("Frame captured")
                self.image_to_display = ImageTk.PhotoImage(self.captured_image.resize((self.width, self.height), Image.LANCZOS))

                self.captured_image_label.configure(image=self.image_to_display)
                self.captured_image_label.configure(text="")
                self.captured_image_label.image = self.image_to_display
                # save the image with high DPI (600) and no compression (use TIFF if possible)
                try:
                    base_tiff = 'captured_image.tiff'
                    if os.path.exists(base_tiff):
                        idx = 1
                        while os.path.exists(f'captured_image_{idx}.tiff'):
                            idx += 1
                        filename = f'captured_image_{idx}.tiff'
                    else:
                        filename = base_tiff
                    self.captured_image.save(filename, dpi=(600, 600), compression='none')
                except Exception:
                    # fallback to PNG with DPI and minimal compression, also avoid overwrite by adding index
                    base_png = 'captured_image.png'
                    if os.path.exists(base_png):
                        idx = 1
                        while os.path.exists(f'captured_image_{idx}.png'):
                            idx += 1
                        filename = f'captured_image_{idx}.png'
                    else:
                        filename = base_png
                    self.captured_image.save(filename, dpi=(600, 600), compress_level=0)

                self.edge_detection_point_button.configure(state=NORMAL)

        except queue.Empty:
            print("No frame available to capture")
        except Exception as e:
            print(f"Failed to capture frame: {e}")

        self.captured_view = self.current_live_view
        self.dispatch('update_captured_frame_center', self.captured_view)
        return True

    def handling_create_sampling_points_during_aquisition(self):
        # add stablization delay
        time.sleep(1)
        self.capture_frame()
        self.edge_detection('point')
        self.on_create_button_clicked()

        relative_distance_to_camera_center = self.convert_pixel_position_to_relative_distance(self.sampling_position_x, self.sampling_position_y)
        
        # turn it into [[x1, y1], [x2, y2]...] format
        relative_distance_to_camera_center = np.stack([relative_distance_to_camera_center[0], relative_distance_to_camera_center[1]], axis=1)
        self.dispatch('update_sampling_points_to_protocol_module', relative_distance_to_camera_center, self.coordinates_order)
        self.dispatch('task_completed')
        return True

    def convert_pixel_position_to_relative_distance(self, x, y):
        x_centered = x - self.captured_image.size[0] / 2
        y_centered = y - self.captured_image.size[1] / 2

        # the pixel size to use for calculating the relative distance in physical space
        if self.captured_view == 'OBJECTIVE':
            x_pixel_size = self.x_pixel_size_objective_camera
            y_pixel_size = self.y_pixel_size_objective_camera
        elif self.captured_view == 'WIDEFIELD':
            x_pixel_size = self.x_pixel_size_widefield_camera
            y_pixel_size = self.y_pixel_size_widefield_camera
        else:
            raise ValueError("Unknown view")
        # convert the pixel position to relative distance
        x_distance = x_centered * x_pixel_size
        y_distance = y_centered * y_pixel_size
        return (x_distance, y_distance)


    def on_test_button_clicked(self):
        relative_distance_to_camera_center = self.convert_pixel_position_to_relative_distance(self.sampling_position_x, self.sampling_position_y)
        view_to_inspect_in = self.current_live_view
        self.dispatch('test_all_sampling_points', view_to_inspect_in, relative_distance_to_camera_center)

    def calculate_focus_score_of_current_image(self):
        image = self.active_camera_thread.get_output_queue().get()
        image = np.array(image)
        image = cv2.medianBlur(image, 3)

        def tenengrad(img):
            gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            # energy = sum of gradient magnitudes squared
            fm = np.sum(gx*gx + gy*gy)
            return fm

        def laplacian_energy(img, thresh=10):
            L = cv2.Laplacian(img, cv2.CV_64F)
            # take absolute so that dark→light and light→dark both count
            A = np.abs(L)
            # mask out anything below your noise floor
            A = A[A > thresh]
            # sum of the remaining “true edges”
            return A.sum()
        
        def brenner(img):
            # roll by two pixels in x and y
            dx = img.astype(np.int32) - np.roll(img.astype(np.int32), -2, axis=1)
            dy = img.astype(np.int32) - np.roll(img.astype(np.int32), -2, axis=0)
            return np.sum(dx*dx + dy*dy)
        
        focus_score_tenengrad = tenengrad(image)

        return focus_score_tenengrad
    
    def handle_calculate_focus_score(self):
        focus_score = self.calculate_focus_score_of_current_image()
        self.dispatch('update_focus_score', focus_score)
        return None

    def focus_camera(self):
        if self.current_live_view == 'OBJECTIVE':
            print("No action will be taken for objective camera autofocus")
            pass
        elif self.current_live_view == 'WIDEFIELD':
            self.dispatch('focus_widefield_camera')

    def on_switch_view_button_clicked(self, view):
        if view == 'TO_WIDE':
            self.active_camera_thread = self.widefield_camera_thread
            self.camera_widget.image_queue = self.active_camera_thread.get_output_queue()
            self.dispatch('switch_view', 'TO_WIDE')
            self.live_frame.configure(text="Live Camera (Widefield)")
            self.switch_view_button.configure(command= lambda: self.on_switch_view_button_clicked('TO_OBJECTIVE'))
            self.current_live_view = 'WIDEFIELD'

        elif view == 'TO_OBJECTIVE':
            self.active_camera_thread = self.objective_field_camera_thread
            self.camera_widget.image_queue = self.active_camera_thread.get_output_queue()
            self.dispatch('switch_view', 'TO_OBJECTIVE')
            self.live_frame.configure(text="Live Camera (Objective)")
            self.switch_view_button.configure(command= lambda: self.on_switch_view_button_clicked('TO_WIDE'))
            self.camera_autofocus_button.configure(state=DISABLED)
            self.current_live_view = 'OBJECTIVE'

    def handle_switch_view_during_aquisition(self, view):
        self.on_switch_view_button_clicked(view)
        self.dispatch('task_completed')

    def on_closing(self):
        print("Stopping image acquisition thread...")
        self.widefield_camera_thread.stop()
        self.objective_field_camera_thread.stop()
        self.widefield_camera_thread.join()
        self.objective_field_camera_thread.join()
        self.widefield_camera.dispose()
        self.objective_field_camera.dispose()

        self.camera_manager.sdk.dispose()
        self.master.destroy()


    def handle_activate_camera_autofocus_button(self):
        self.camera_autofocus_button.configure(state=NORMAL)
        return None

    def handle_activate_switch_view_button(self):
        self.switch_view_button.configure(state=NORMAL)
        return None

    def handle_get_current_camera_view(self):
        # return the current live camera view ("WIDEFIELD" or "OBJECTIVE")
        return self.current_live_view

    def handle_check_current_camera_view(self, view_to_check):
        if self.current_live_view == view_to_check:
            self.dispatch('task_completed')
        else:
            print(f"Current view is {self.current_live_view}, but expected {view_to_check}.")
            self.dispatch('abort_aquisition')
        return True

if __name__ == "__main__":
    root = ttk.Window()
    root.style.theme_use('superhero')
    live_view_frame = LiveViewModule(root)
    live_view_frame.grid(row=0, column=0, sticky='nsew')
    root.protocol("WM_DELETE_WINDOW", live_view_frame.on_closing)
    root.mainloop()
