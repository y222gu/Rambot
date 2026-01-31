import itertools
import numpy as np
import matplotlib.pyplot as plt
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import StringVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread
import queue
import logging
import threading
import os
import scipy.signal
from noodlepy.gui.publisher_subscriber import Publisher
from noodlepy.gui.wasatch_manager import WasatchManager
from noodlepy.gui.nanodrive import NanoDrive
import time
from PIL import Image, ImageTk

class WasatchAutofocusModule(Publisher, ttk.Frame):
    def __init__(self, parent):
        Publisher.__init__(self, ["update_nanodrive_position", 'task_completed', 'move_prusa_to'])
        ttk.Frame.__init__(self, parent)
        self.name = "WasatchAutofocusModule"

        self.initial_nanodrive_position = 50
        self.initial_integ_time_ms = 1000
        self.initial_autofocus_integ_time_ms = 100
        self.initial_laser_power_mW = 450
        self.image_height = 300
        self.image_width = 600
        self.laser_power_spinbox_increment = 20

        self.LASER_POWER_SPINBOX_LOWER_LIMIT = 0 # fixed 
        self.LASER_POWER_SPINBOX_UPPER_LIMIT = 450 # fixed

        self.integ_time_spinbox_lower_limit = 100
        self.integ_time_spinbox_upper_limit = 100000
        self.integ_time_spinbox_increment = 5

        self.nanodrive_movement_stabilization_time = 0.02 # seconds
        self.prusa_initial_movement_stabilization_time = 1 # seconds
        self.wasatch_prusa_focus_rough_fine_overlap_step_number = 4 # number of steps to overlap in the fine autofocus
        self.wasatch_nanodrive_focus_rough_fine_overlap_step_number = 2

        self.save_folder_path = os.path.join(os.getcwd(), "captured_spectra")

        self.prusa_focus_score_method = 'ratio'  #  or 'entropy'
        self.nanodrive_focus_score_method = 'ratio'  #  or 'entropy'

        self.wasatch_focus_prusa_lower_limit = None
        self.wasatch_focus_prusa_upper_limit = None

        # self.latest_spectrum = np.zeros(self.image_width)  # Initialize with zeros
        self.playing_live_spectrum_plot = False
        self.units = "wavelength"
        self.nano_drive = NanoDrive()
        self.create_live_spectrum_widgets()  
        self.create_widgets_for_autofocus()

        self.spectrum_queue = queue.Queue()
        self.wasatch_manager = WasatchManager(self.initial_integ_time_ms, self.initial_laser_power_mW)
        if self.wasatch_manager.connect():
            self.start_spectrum_thread()
            # self.update_live_spectrum_plot()
        else:
            logging.error('Failed to connect to Wasatch spectrometer. Check connection')

    def create_live_spectrum_widgets(self):
        play_icon = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","play_icon.png")).resize((55, 55))
        pause_icon = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","pause_icon.png")).resize((55, 55))
        self.play_icon = ImageTk.PhotoImage(play_icon)
        self.pause_icon = ImageTk.PhotoImage(pause_icon)

        self.spectrum_frame = ttk.Labelframe(self, text="Wasatch Live Spectrum", padding=5)
        self.spectrum_frame.grid(row=0, column=0, columnspan=6, sticky='nsew', padx=5, pady=5)

        self.laser_power_label = ttk.Label(self.spectrum_frame, text="Power (mW)", width=5)
        self.laser_power_label.grid(row=1, column=0, sticky='nsew', pady=5, padx=5)
        self.laser_power_var = StringVar()
        self.laser_power_var.set(self.initial_laser_power_mW)
        self.laser_power_spinbox = ttk.Spinbox(self.spectrum_frame, textvariable=self.laser_power_var, from_=self.LASER_POWER_SPINBOX_LOWER_LIMIT, to=self.LASER_POWER_SPINBOX_UPPER_LIMIT, increment=self.laser_power_spinbox_increment, width=5, justify='center')
        self.laser_power_spinbox.grid(row=2, column=0, sticky='ew', pady=5, padx=5)
        self.laser_power_spinbox.bind("<FocusOut>", self.update_wasatch_settings)
        self.laser_power_spinbox.bind("<Return>", self.update_wasatch_settings)

        self.integ_time_label = ttk.Label(self.spectrum_frame, text="Expo. Time (ms)", width=5)
        self.integ_time_label.grid(row=1, column=1, sticky='nsew', pady=5, padx=5)
        self.integ_time_var = StringVar()
        self.integ_time_var.set(self.initial_integ_time_ms)
        self.integ_time_spinbox = ttk.Spinbox(self.spectrum_frame, textvariable=self.integ_time_var, from_=self.integ_time_spinbox_lower_limit, to=self.integ_time_spinbox_upper_limit, increment=self.integ_time_spinbox_increment, width=5, justify='center')
        self.integ_time_spinbox.grid(row=2, column=1, sticky='ew', pady=5, padx=5)
        self.integ_time_spinbox.bind("<FocusOut>", self.update_wasatch_settings)
        self.integ_time_spinbox.bind("<Return>", self.update_wasatch_settings)

        self.laser_button = ttk.Button(self.spectrum_frame, text="Laser On", command=self.turn_laser_on, bootstyle ='info-outline', width=5)
        self.laser_button.grid(row=1, column=2, rowspan=2, sticky='nsew', pady=5, padx=5)

        self.play_button = ttk.Button(self.spectrum_frame, image=self.play_icon, command=self.start_to_play_live_spectrum_plot, bootstyle ='dark', width=5)
        self.play_button.grid(row=1, column=3, rowspan =2, sticky='ew', pady=5, padx=5)

        self.capture_button = ttk.Button(self.spectrum_frame, text="Capture", bootstyle ='info-outline', command=self.capture_live_spectrum)
        self.capture_button.grid(row=1, column=4, columnspan=2, sticky='ew', pady=5, padx=5)

        self.capture_number_var = StringVar()
        self.capture_number_var.set(1)
        self.capture_number_entry = ttk.Entry(self.spectrum_frame, textvariable=self.capture_number_var, width=5)
        self.capture_number_entry.grid(row=2, column=4, sticky='ew', pady=5, padx=5)

        # add toggle to show plotted captured spectra when capturing
        self.plot_flag_var = ttk.BooleanVar(value=False)
        self.plot_flag_check = ttk.Checkbutton(self.spectrum_frame, text="Show Plot", variable=self.plot_flag_var, bootstyle="info")
        self.plot_flag_check.grid(row=2, column=5, sticky='ew', pady=5, padx=5)

        self.live_spectrum_canvas = ttk.Canvas(self.spectrum_frame, width=self.image_width, height=self.image_height)
        self.live_spectrum_canvas.grid(row=0, column=0, columnspan=6, sticky='nsew', pady=5, padx=5)

        self.unit_toggle_var = ttk.BooleanVar(value=False)
        self.unit_toggle_btn = ttk.Checkbutton(self.spectrum_frame, variable=self.unit_toggle_var, text="To cm⁻¹", bootstyle="info-round-toggle", command=self.update_live_spectrum_units)
        self.unit_toggle_btn.grid(row=0, column=5, sticky='en', pady=5, padx=5)
        self.live_spectrum_fig, self.live_spectrum_ax = plt.subplots(figsize=(8, 4))
        self.live_spectrum_line, = self.live_spectrum_ax.plot([], [])
        self.live_spectrum_line.set_linewidth(0.8)
        self.live_spectrum_line.set_color('#5bc0de')

        if self.units == "wavelength":
            self.live_spectrum_ax.set_xlabel("Wavelength (nm)", fontsize=6, color='white')
        elif self.units == "wavenumber":
            self.live_spectrum_ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=6, color='white')
        else:
            print("Invalid units")

        self.live_spectrum_ax.set_ylabel("Intensity (counts)", fontsize=6, color='white')
        self.live_spectrum_ax.set_title("Raman Spectrum", fontsize=8, color='white')
        self.live_spectrum_ax.tick_params(axis='both', which='major', labelsize=6)
        self.live_spectrum_ax.tick_params(axis='both', which='minor', labelsize=6)
        self.live_spectrum_ax.tick_params(axis='x', colors='white')
        self.live_spectrum_ax.tick_params(axis='y', colors='white')
        self.live_spectrum_ax.spines['bottom'].set_color('white')
        self.live_spectrum_ax.spines['left'].set_color('white')
        self.live_spectrum_ax.spines['right'].set_visible(False)
        self.live_spectrum_ax.spines['top'].set_visible(False)
        self.live_spectrum_ax.set_facecolor("none")
        self.live_spectrum_ax.xaxis.label.set_color('white')
        self.live_spectrum_ax.yaxis.label.set_color('white')
        plt.tight_layout(pad=0.5)
        self.live_spectrum_fig.patch.set_alpha(0)
        self.live_spectrum_canvas = FigureCanvasTkAgg(self.live_spectrum_fig, master=self.live_spectrum_canvas)
        self.live_spectrum_canvas.draw()
        self.live_spectrum_canvas.get_tk_widget().grid(row=0, column=0, columnspan=4, sticky='nsew')
        self.background = self.live_spectrum_canvas.copy_from_bbox(self.live_spectrum_ax.bbox)

    def create_widgets_for_autofocus(self):

        self.autofocus_frame = ttk.Labelframe(self, text='Wasatch Autofocus', padding=5)
        self.autofocus_frame.grid(row=1, column=0, columnspan=6, sticky="nsew", padx=5, pady=5)

        self.wasatch_autofocus_fig, self.entropy_ax, self.intensity_ax = self.initialize_wasatch_autofocus_figure()
        self.wasatch_autofocus_canvas = FigureCanvasTkAgg(self.wasatch_autofocus_fig, master=self.autofocus_frame)
        self.wasatch_autofocus_canvas.draw()
        self.wasatch_autofocus_canvas.get_tk_widget().grid(row=0, column=0, columnspan=6, rowspan=2, sticky="nsew", padx=5, pady=5)

        # create initial values for autofocus parameters
        self.wasatch_autofocus_num_rep_var = StringVar()
        self.wasatch_autofocus_num_rep_var.set(3)
        self.wasatch_prusa_autofocus_step_num_var = StringVar()
        self.wasatch_prusa_autofocus_step_num_var.set(20)
        self.wasatch_autofocus_laser_power_var = StringVar()
        self.wasatch_autofocus_laser_power_var.set(self.initial_laser_power_mW)
        self.wasatch_autofocus_integ_time_var = StringVar()
        self.wasatch_autofocus_integ_time_var.set(self.initial_autofocus_integ_time_ms)
        self.wasatch_nanodrive_autofocus_step_num_var = StringVar()
        self.wasatch_nanodrive_autofocus_step_num_var.set(10)
        self.wasatch_nanodrive_autofocus_range_low_var = StringVar()
        self.wasatch_nanodrive_autofocus_range_low_var.set(10)
        self.wasatch_nanodrive_autofocus_range_high_var = StringVar()
        self.wasatch_nanodrive_autofocus_range_high_var.set(80)
        
        # Create the entries for the autofocus parameters
        self.wasatch_prusa_autofocus_step_num_label = ttk.Label(self.autofocus_frame, text="Prusa Number of Steps")
        self.wasatch_prusa_autofocus_step_num_label.grid(row=2, column=0, pady=5, padx=5, sticky="e")
        self.wasatch_prusa_autofocus_step_num_entry = ttk.Entry(self.autofocus_frame, width=5, textvariable=self.wasatch_prusa_autofocus_step_num_var)
        self.wasatch_prusa_autofocus_step_num_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        self.wasatch_nanoDrive_autofocus_step_num_label = ttk.Label(self.autofocus_frame, text="NanoDrive Number of Steps")
        self.wasatch_nanoDrive_autofocus_step_num_label.grid(row=2, column=2, pady=5, padx=5, sticky="e")
        self.wasatch_nanodrive_autofocus_step_num_entry = ttk.Entry(self.autofocus_frame, width=5, textvariable=self.wasatch_nanodrive_autofocus_step_num_var)
        self.wasatch_nanodrive_autofocus_step_num_entry.grid(row=2, column=3, padx=5, pady=5, sticky="w")
        
        self.wasatch_nanodrive_autofocus_range_label = ttk.Label(self.autofocus_frame, text="NanoDrive Z Min [um]")
        self.wasatch_nanodrive_autofocus_range_label.grid(row=3, column=0, pady=5, padx=5, sticky="e")
        self.wasatch_nanodrive_autofocus_range_low_spinbox = ttk.Spinbox(self.autofocus_frame, from_=self.nano_drive.min_position_um, to=self.nano_drive.max_position_um, increment=1, textvariable=self.wasatch_nanodrive_autofocus_range_low_var, width=5)
        self.wasatch_nanodrive_autofocus_range_low_spinbox.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        self.wasatch_nanodrive_autofocus_range_label = ttk.Label(self.autofocus_frame, text="NanoDrive Z Max [um]")
        self.wasatch_nanodrive_autofocus_range_label.grid(row=3, column=2, pady=5, padx=5, sticky="e")
        self.wasatch_autofocus_range_high_spinbox = ttk.Spinbox(self.autofocus_frame, from_=self.nano_drive.min_position_um, to=self.nano_drive.max_position_um, increment=1, textvariable=self.wasatch_nanodrive_autofocus_range_high_var, width=5)
        self.wasatch_autofocus_range_high_spinbox.grid(row=3, column=3, padx=5, pady=5, sticky="w")

        self.wasatch_autofocus_laser_power_label = ttk.Label(self.autofocus_frame, text="Laser Power (mW)")
        self.wasatch_autofocus_laser_power_label.grid(row=4, column=0, pady=5, padx=5, sticky="e")
        self.wasatch_autofocus_laser_power_entry = ttk.Entry(self.autofocus_frame, width=5, textvariable=self.wasatch_autofocus_laser_power_var)
        self.wasatch_autofocus_laser_power_entry.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        self.wasatch_autofocus_integ_time_label = ttk.Label(self.autofocus_frame, text="Expo. Time (ms)")
        self.wasatch_autofocus_integ_time_label.grid(row=4, column=2, pady=5, padx=5, sticky="e")
        self.wasatch_autofocus_integ_time_entry = ttk.Entry(self.autofocus_frame, width=5, textvariable=self.wasatch_autofocus_integ_time_var)
        self.wasatch_autofocus_integ_time_entry.grid(row=4, column=3, padx=5, pady=5, sticky="w")

        self.wasatch_autofocus_rep_num_label = ttk.Label(self.autofocus_frame, text="Number of Reps")
        self.wasatch_autofocus_rep_num_label.grid(row=2, column=4, pady=5, padx=5, sticky="e")
        self.wasatch_autofocus_rep_num_entry = ttk.Entry(self.autofocus_frame, width=5, textvariable=self.wasatch_autofocus_num_rep_var)
        self.wasatch_autofocus_rep_num_entry.grid(row=2, column=5, padx=5, pady=5, sticky="w")

        self.wasatch_prusa_autofocus_button = ttk.Button(self.autofocus_frame, text="Focus Prusa", command=self.start_prusa_autofocus, bootstyle="info", width=15)
        self.wasatch_prusa_autofocus_button.grid(row=3, column=4, columnspan=2, padx=5, pady=5, sticky="news")

        self.wasatch_nanodrive_autofocus_button = ttk.Button(self.autofocus_frame, text="Focus Nanodrive", command=self.start_nanodrive_autofocus, bootstyle="info", width=15)
        self.wasatch_nanodrive_autofocus_button.grid(row=4, column=4, columnspan=2, padx=5, pady=5, sticky="news")


    def start_to_play_live_spectrum_plot(self):
        self.playing_live_spectrum_plot = True
        self.play_button.config(image=self.pause_icon, command=self.stop_to_play_live_spectrum_plot, bootstyle ='info')
        self.update_live_spectrum_plot()

    def stop_to_play_live_spectrum_plot(self):
        self.playing_live_spectrum_plot = False
        self.play_button.config(image=self.play_icon, command=self.start_to_play_live_spectrum_plot, bootstyle ='dark')

    def initialize_wasatch_autofocus_figure(self):
                # Create the figure and axes
        fig, axes = plt.subplots(2, 1, figsize=(8, 4))

        entropy_ax = axes[0]
        intensity_ax = axes[1]

        entropy_ax.set_xlabel('Z-axis position',fontsize=6, color='white')
        entropy_ax.set_ylabel('Entropy', fontsize=6, color='white')
        entropy_ax.set_title('Entropy', fontsize=8, color='white')

        intensity_ax.set_xlabel('Z-axis position', fontsize=6, color='white')
        intensity_ax.set_ylabel('Intensity', fontsize=6, color='white')
        intensity_ax.set_title('Average Spectrum at Current Position', fontsize=8, color='white')

        for ax in entropy_ax, intensity_ax:
            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.tick_params(axis='both', which='minor', labelsize=6)
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.set_facecolor("none")
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.grid(color='white', linestyle='-', linewidth=0.2)

        plt.tight_layout(pad=0.5)
        fig.patch.set_alpha(0)
        return fig, entropy_ax, intensity_ax
    
    def update_live_spectrum_units(self):
        if self.unit_toggle_var.get():
            self.units = "wavenumber"
            self.live_spectrum_ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=6, color='white')
            self.unit_toggle_btn.configure(text="To nm")
        else:
            self.units = "wavelength"
            self.live_spectrum_ax.set_xlabel("Wavelength (nm)", fontsize=6, color='white')
            self.unit_toggle_btn.configure(text="To cm⁻¹")

        self.live_spectrum_ax.relim()
        self.live_spectrum_ax.autoscale_view()
        self.live_spectrum_canvas.draw()

    def update_live_spectrum_plot(self):
        """Update the spectrum plot with the latest data from the queue."""

        try:
            # self.latest_spectrum = self.spectrum_queue.get_nowait()
            
            if self.units == "wavelength":
                x_data = self.wasatch_manager.settings.wavelengths
            else:
                x_data = self.wasatch_manager.settings.wavenumbers
            
            self.live_spectrum_line.set_data(x_data, self.latest_spectrum)
            self.live_spectrum_ax.relim()
            self.live_spectrum_ax.autoscale_view()

            # Update the plot
            self.live_spectrum_canvas.draw_idle()
            # print("Spectrum updated at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            # print("First 10 values of the spectrum:", self.latest_spectrum[:10])
            # print('--------------------------------------------------------------------------')

            # print("threading.current_thread().name: ", threading.current_thread().name)

        except queue.Empty:
            pass
        
        if self.playing_live_spectrum_plot:
            # print("Plotting live spectrum with integration time:", self.wasatch_manager.integ_time_ms)
            self.after(self.wasatch_manager.integ_time_ms, self.update_live_spectrum_plot)

    def start_spectrum_thread(self):
        """Start a thread to collect spectrum data."""
        if not hasattr(self, 'spectrum_thread') or not self.spectrum_thread.is_alive():
            self.spectrum_thread = Thread(target=self.collect_spectrum)
            self.spectrum_thread.daemon = True
            self.spectrum_thread.start()

    def collect_spectrum(self):
        while True:
            # spectrum = None
            spectrum = self.wasatch_manager.get_spectrum()
            # print("Spectrum collected with integration time:", self.wasatch_manager.integ_time_ms)
            # print(spectrum)
            if spectrum is not None:
                # self.spectrum_queue.put(spectrum)
                self.latest_spectrum = spectrum
                # # print the time stamp of the spectrum
                # print("Spectrum collected at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                # #print the first 10 values of the spectrum
                # print("First 10 values of the spectrum:", spectrum[:10])
                # # print(f'size now {self.spectrum_queue.qsize()}/{self.spectrum_queue.maxsize}; free slots now {self.spectrum_queue.maxsize - self.spectrum_queue.qsize()}')

    def update_wasatch_settings(self, event=None):
        try:
            self.wasatch_manager.set_integration_time(int(self.integ_time_var.get()))
            self.wasatch_manager.set_laser_power(int(self.laser_power_var.get()))
        except ValueError:
            pass

    def capture_live_spectrum(self):
        # Measure requested number of spectra and save them (like handle_measure_spectra_and_save_to_specific_folder).
        num_rep = int(float(self.capture_number_var.get() or 1))

        # Acquire spectra (this uses self.latest_spectrum internally)
        wavelength, wavelengths, intensities = self.measure_spectra(num_rep)

        # Prepare save folders
        spectra_folder = os.path.join(self.save_folder_path, "spectra")
        metadata_folder = os.path.join(self.save_folder_path, "metadata")
        os.makedirs(spectra_folder, exist_ok=True)
        os.makedirs(metadata_folder, exist_ok=True)

        # Determine next file number to avoid overwriting
        existing = [f for f in os.listdir(spectra_folder) if f.endswith(".txt")]
        file_number = 1
        while f"Captured_spectra_{file_number}.txt" in existing:
            file_number += 1
        filename_base = f"Captured_spectra_{file_number}"

        # Flatten and save spectral data (wavelength,intensity pairs)
        wavelengths_flatten = list(itertools.chain.from_iterable(wavelengths))
        intensities_flatten = list(itertools.chain.from_iterable(intensities))
        with open(os.path.join(spectra_folder, filename_base + ".txt"), "w") as outfile:
            for i in range(len(wavelengths_flatten)):
                outfile.write(f"{wavelengths_flatten[i]:0.2f}, {intensities_flatten[i]}\n")

        # Save metadata similar to handle_measure_spectra_and_save_to_specific_folder
        with open(os.path.join(metadata_folder, filename_base + "_metadata.txt"), "w") as metafile:
            metafile.write(f"Number of repetitions: {num_rep}\n")
            metafile.write(f"Integration time (ms): {self.wasatch_manager.integ_time_ms}\n")
            metafile.write(f"Laser power (mW): {self.wasatch_manager.laser_power_mW}\n")
            metafile.write(f"Prusa position (mm): {getattr(self, 'best_prusa_focus_wasatch', None)}\n")
            metafile.write(f"Nanodrive position (um): {getattr(self, 'best_nanodrive_focus_position', None)}\n")
            metafile.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")

        if self.plot_flag_var.get():
            if self.units == "wavelength":
                x_axis = wavelength
            else:
                x_axis = self.wasatch_manager.settings.wavenumbers

            new_window = ttk.Toplevel()
            new_window.title(f"Captured Spectra x{num_rep}")

            fig, ax = plt.subplots(figsize=(6, 3))
            for idx, spec in enumerate(intensities):
                ax.plot(x_axis, spec, label=f"rep {idx+1}", linewidth=0.8)
            ax.set_ylabel("Intensity (counts)", fontsize=6, color='white')
            ax.set_title("Captured Raman Spectra", fontsize=8, color='white')
            if self.units == "wavelength":
                ax.set_xlabel("Wavelength (nm)", fontsize=6, color='white')
            else:
                ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=6, color='white')

            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_facecolor("none")
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.legend(fontsize=6, loc='upper right')

            plt.tight_layout(pad=0.5)
            fig.patch.set_alpha(0)
            canvas = FigureCanvasTkAgg(fig, master=new_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=ttk.TOP, fill=ttk.BOTH, expand=1)

            # also save the plotted figure
            fig.savefig(os.path.join(self.save_folder_path, filename_base + "_plot.png"), dpi=300)
            plt.close(fig)
        
    def debug_with_polystyrene(self):
        expected_peak = 1006.22
        expected_counts = 1500 
        peak_tolerance_cm = 5

        print("Take a sample spectrum")
        # get the latest spectrum
        measurement = self.latest_spectrum

        print('Comparing peaks to polystyrene')
        peak_pixels = scipy.signal.find_peaks(measurement, height=700)[0]

        peak_pixel = None
        for pixel in peak_pixels:
            peak_cm = self.wasatch_manager.settings.wavenumbers[pixel]
            if abs(expected_peak - peak_cm) <= peak_tolerance_cm:
                print(f"Found expected {expected_peak}cm⁻¹ at pixel {pixel} ({peak_cm:0.2f}cm⁻¹)")
                peak_pixel = pixel
                break

        if peak_pixel is None:
            print(f"Failed to find {expected_peak}cm⁻¹ peak in sample")
            return False

        counts = measurement[peak_pixel]
        if counts < expected_counts:
            print(f"Failed. {expected_peak}cm⁻¹ peak counts too low ({counts} < {expected_counts}): adjust working distance")
            return False
        
        print(f"Success! {expected_peak}cm⁻¹ peak found with {counts} counts.")
        return True
    
    def turn_laser_on(self):
        if self.wasatch_manager is None:
            laser_power_mW = int(self.laser_power_var.get())
            integ_time_ms = int(self.integ_time_var.get())
            self.wasatch_manager = WasatchManager(integ_time_ms, laser_power_mW)
            if not self.wasatch_manager.connect():
                print("Failed to connect to Wasatch spectrometer. Check connection and try again.")
                return
        self.wasatch_manager.turn_laser_on()
        self.laser_button.config(text="Laser Off", command=self.turn_laser_off, bootstyle ='info')

    def turn_laser_off(self):
        self.wasatch_manager.turn_laser_off()
        self.laser_button.config(text="Laser On", command=self.turn_laser_on, bootstyle ='info-outline')

    def start_prusa_autofocus(self):
        # Stop the live spectrum and set up parameters
        self.stop_to_play_live_spectrum_plot()
        self.autofocus_worker_thread = threading.Thread(target=self.rough_focus_wasatch_with_prusa)
        self.autofocus_worker_thread.start()

    def rough_focus_wasatch_with_prusa(self):
        if self.wasatch_focus_prusa_lower_limit and self.wasatch_focus_prusa_upper_limit:
            print("Starting autofocus for Wasatch with Prusa in range: [", self.wasatch_focus_prusa_lower_limit, ' , ',self.wasatch_focus_prusa_upper_limit, '] mm')
            num_step = int(float(self.wasatch_prusa_autofocus_step_num_var.get()))
            num_rep = int(float(self.wasatch_autofocus_num_rep_var.get()))
            self.wasatch_manager.set_laser_power(int(self.wasatch_autofocus_laser_power_var.get()))
            self.wasatch_manager.set_integration_time(int(self.wasatch_autofocus_integ_time_var.get()))

            # move nanodrive to the initial position
            self.nano_drive.move_to(self.initial_nanodrive_position)
            
            # wait until the nanodrive reaches the position
            while True:
                position = self.nano_drive.get_current_position()
                if position == self.initial_nanodrive_position:
                    break
            self.dispatch("update_nanodrive_position", position)
            time.sleep(self.nanodrive_movement_stabilization_time)

            # rough focus with Prusa
            self.best_prusa_focus_wasatch, fineMin, fineMax = self.focus_wasatch_with_prusa(self.wasatch_focus_prusa_lower_limit, self.wasatch_focus_prusa_upper_limit, num_step, num_rep, self.wasatch_prusa_focus_rough_fine_overlap_step_number, 'low_to_high')
            # move prusa to the best focus position
            self.dispatch("move_prusa_to", self.best_prusa_focus_wasatch)

            # set the laser power back to the initial value
            self.wasatch_manager.set_laser_power(int(self.laser_power_var.get()))
            self.wasatch_manager.set_integration_time(int(self.integ_time_var.get()))

            if self.best_prusa_focus_wasatch:
                print("Prusa Autofocus complete")
                return True
        else:
            print("Prusa Z-axis limits are not set. Please set them before starting autofocus.")
            return False

    def focus_wasatch_with_prusa(self, lower_limit, upper_limit, num_step, num_rep, wasatch_rough_fine_focus_overlap_step_number, moving_sequence):
        prusa_z_axis_range =  np.round(np.linspace(lower_limit, upper_limit, num_step), 2)
        if moving_sequence == "high_to_low":
            prusa_z_axis_range = prusa_z_axis_range[::-1]
        if moving_sequence == "low_to_high":
            prusa_z_axis_range = prusa_z_axis_range
        print("Focus wasatch with prusa number of steps:", num_step)
        print("Focus Wasatch with Prusa Z-axis range: [", prusa_z_axis_range, "] mm")
        x1_data, y1_data = [], []
        current_step = 0
        while current_step < len(prusa_z_axis_range):
            # Move prusa and get current position
            z_pos = prusa_z_axis_range[current_step]
            self.dispatch("move_prusa_to", z_pos)
            if current_step == 0:
                time.sleep(self.prusa_initial_movement_stabilization_time)

            # Measure spectra in the background thread (self.num_rep reps)
            wavelength, wavelengths, intensities = self.measure_spectra(num_rep)
            median_intensity = np.median(intensities, axis=0)

            # Ensure wavelength is a NumPy array so comparisons are vectorized
            wavelength = np.asarray(wavelength)
            if self.prusa_focus_score_method == "entropy":
                score = self.calculate_entropy(wavelength, median_intensity)
            if self.prusa_focus_score_method == "ratio":
                score = self.calculate_ratio_score(wavelength, median_intensity)
            else:
                score = self.calculate_entropy(wavelength, median_intensity)
            x1_data.append(z_pos)
            y1_data.append(score)

            # Update plot in the main thread
            self.after(0, self.plot_autofocus_data, x1_data, y1_data, wavelength, median_intensity)

            # Proceed to the next step
            current_step += 1

        best_prusa_focus, fineMin, fineMax = self.refine_focus_range(y1_data, prusa_z_axis_range, wasatch_rough_fine_focus_overlap_step_number)

        return best_prusa_focus, fineMin, fineMax

    def start_nanodrive_autofocus(self):
        # Stop the live spectrum and set up parameters
        self.stop_to_play_live_spectrum_plot()
        self.autofocus_worker_thread = threading.Thread(target=self.nanodrive_rough_autofocus)
        self.autofocus_worker_thread.start()

    def nanodrive_rough_autofocus(self):
        print("Starting rough nanodrive autofocus for Wasatch")
        min = float(self.wasatch_nanodrive_autofocus_range_low_var.get())
        max = float(self.wasatch_nanodrive_autofocus_range_high_var.get())
        num_step = int(float(self.wasatch_nanodrive_autofocus_step_num_var.get()))
        num_rep = int(float(self.wasatch_autofocus_num_rep_var.get()))
        z_axis_range =  np.round(np.linspace(min, max, num_step))
        x1_data, y1_data = [], []
        current_step = 0
        self.wasatch_manager.set_laser_power(int(self.wasatch_autofocus_laser_power_var.get()))
        self.wasatch_manager.set_integration_time(int(self.wasatch_autofocus_integ_time_var.get()))

        while current_step < len(z_axis_range):
            # Move nano drive and get current position
            z_pos = z_axis_range[current_step]
            self.nano_drive.move_to(z_pos)

            while True:
                position = self.nano_drive.get_current_position()
                if position == z_pos:
                    break
            
            self.dispatch("update_nanodrive_position", position)
            print("Current position:", position)

            # Measure spectra in the background thread (self.num_rep reps)
            wavelength, wavelengths, intensities = self.measure_spectra(num_rep)
            median_intensity = np.median(intensities, axis=0)

            # Ensure wavelength is a NumPy array so comparisons are vectorized
            wavelength = np.asarray(wavelength)
            if self.nanodrive_focus_score_method == 'entropy':
                ent = self.calculate_entropy(wavelength, median_intensity)
            elif self.nanodrive_focus_score_method == 'ratio':
                ent = self.calculate_ratio_score(wavelength, median_intensity)
            else:
                ent = self.calculate_entropy(wavelength, median_intensity)
            x1_data.append(position)
            y1_data.append(ent)

            # Update plot in the main thread
            self.after(0, self.plot_autofocus_data, x1_data, y1_data, wavelength, median_intensity)

            # Proceed to the next step
            current_step += 1

        # Once done with rough autofocus, proceed to fine autofocus
        self.best_nanodrive_focus_position, fineMin, fineMax = self.refine_focus_range(y1_data, z_axis_range, self.wasatch_nanodrive_focus_rough_fine_overlap_step_number)
        print('Rough Nanodrive focus position found:', self.best_nanodrive_focus_position)
        if self.best_nanodrive_focus_position is not None:
            self.nano_drive.move_to(self.best_nanodrive_focus_position)
            time.sleep(self.nanodrive_movement_stabilization_time)
            self.dispatch("update_nanodrive_position", self.nano_drive.get_current_position())

        # set the laser power back to the initial value
        self.wasatch_manager.set_laser_power(int(self.laser_power_var.get()))
        self.wasatch_manager.set_integration_time(int(self.integ_time_var.get()))

        print("Nanodrive focusing complete")
        return True

    def nanodrive_fine_autofocus(self, fineMin, fineMax, num_rep):
        print("Starting fine nanodrive autofocus for Wasatch")
        step_interval = 3  # um (finest step size of nano piezo)Define the interval between steps
        z_axis_range = np.arange(fineMin, fineMax + step_interval, step_interval)

        # z_axis_range = np.round(np.linspace(fineMin, fineMax, num_rep))  # Generate range with the calculated step interval
        print("Fine Nanodrive Z-axis range:", z_axis_range)
        x2_data, y2_data = [], []
        current_step = 0

        while current_step < len(z_axis_range):
            z_pos = z_axis_range[current_step]
            self.nano_drive.move_to(z_pos)

            while True:
                position = self.nano_drive.get_current_position()
                if position == z_pos:
                    break

            self.dispatch("update_nanodrive_position", position)

            # Measure spectra in the background thread
            wavelength, wavelengths, intensities = self.measure_spectra(num_rep)
            median_intensity = np.median(intensities, axis=0)

            # Ensure wavelength is a NumPy array so comparisons are vectorized
            wavelength = np.asarray(wavelength)
            if self.nanodrive_focus_score_method == 'entropy':
                ent = self.calculate_entropy(wavelength, median_intensity)
            elif self.nanodrive_focus_score_method == 'convolution':
                ent = self.calculate_convolution_score(median_intensity)
            else:
                ent = self.calculate_ratio_score(wavelength, median_intensity)
            x2_data.append(position)
            y2_data.append(ent)

            # Update plot in the main thread
            self.after(0, self.plot_autofocus_data, x2_data, y2_data, wavelength, median_intensity)

            # Proceed to the next fine autofocus step
            current_step += 1

        # After fine autofocus, finalize focus position
        focusedPos = z_axis_range[np.argmin(y2_data)]
        print("Fine Nanodrive focus position found", focusedPos)
        return focusedPos

    def handle_autofocus_wasatch_with_nanodrive_during_aquisition(self):
        def on_focus_complete(result):
            if result:
                self.dispatch('task_completed')
            else:
                self.dispatch('abort_aquisition')

        # Call the thread-running function, passing the callback
        self.stop_to_play_live_spectrum_plot()
        self.run_in_thread_with_callback(self.nanodrive_rough_autofocus, on_focus_complete)

    def handle_autofocus_wasatch_with_prusa_during_aquisition(self):
        def on_focus_complete(result):
            if result:
                self.dispatch('task_completed')
            else:
                self.dispatch('abort_aquisition')

        # Call the thread-running function, passing the callback
        self.stop_to_play_live_spectrum_plot()
        self.run_in_thread_with_callback(self.rough_focus_wasatch_with_prusa, on_focus_complete)

    def run_in_thread_with_callback(self, func, callback, *args):
        def wrapper():
            # Execute the target function and store the result
            result = func(*args)
            # Pass the result to the callback function
            callback(result)

        # Run the wrapper function in a separate thread
        thread = Thread(target=wrapper, daemon=True)
        thread.start()

    def measure_spectra(self, num_rep):
        intensities = []
        wavelengths = []
        wavelength = self.wasatch_manager.settings.wavelengths
        # wait for one exposure time to ensure the spectrometer is ready
        time.sleep(self.wasatch_manager.integ_time_ms / 1000.0)

        for i in range(int(num_rep)):
            # spectrum = self.spectrum_queue.get()

            intensities.append(self.latest_spectrum)
            wavelengths.append(wavelength)

            # print('Spectrum acquired:', i + 1)
            # print('Intensity:', self.latest_spectrum)
            # wait for the integration time
            time.sleep(self.wasatch_manager.integ_time_ms / 1000.0)

        return wavelength, wavelengths, intensities

    # def calculate_convolution_score(self, intensities):
    #     smoothed_intensity = scipy.signal.savgol_filter(intensities, self.smoothing_window_length, self.smoothing_polyorder)
    #     normalized_intensity = (smoothed_intensity - np.min(smoothed_intensity)) / (np.max(smoothed_intensity) - np.min(smoothed_intensity))
        
    #     min_len = min(len(self.focus_ref_intensity), len(normalized_intensity))
        
    #     # calculate convolution score
    #     if min_len > 0:
    #         score = np.max(np.correlate(normalized_intensity[:min_len],
    #                                          self.focus_ref_intensity[:min_len], mode='valid'))
    #     else:
    #         score = 0.0
    #     return score
    
    def calculate_entropy(self, wavelength, intensity):
        mask = (wavelength >= 830) & (wavelength <= 920)
        cropped_intensity = intensity[mask]
        normalized_int = cropped_intensity / np.sum(cropped_intensity)
        entropy = -np.sum(normalized_int * np.log2(normalized_int + 1e-12))  # add a small value to avoid log(0)
        return entropy
    
    def calculate_ratio_score(self, wavelength, intensity):
        mask = (wavelength >= 790) & (wavelength <= 795)
        reflected_laser = intensity[mask]
        quartz_peak = intensity[(wavelength >= 808) & (wavelength <= 815)]
        ratio = np.sum(quartz_peak) / np.sum(reflected_laser)
        return ratio

    def refine_focus_range(self, entropy_list, z_axis_range, overlap_step_number):
        k = np.argmin(entropy_list)
        best_focus_position = z_axis_range[k]
        print('During this round of focus, the best position is:', z_axis_range[k], ' mm')
        print('Refining the focus range...')
        fineMin = z_axis_range[max(int(k - overlap_step_number/2), 0)]
        fineMax = z_axis_range[min(int(k + overlap_step_number/2), len(z_axis_range) - 1)]
        print("The fine range is: [", fineMin, ",", fineMax, "] mm")
        print('--------------------')
        return best_focus_position, fineMin, fineMax

    def plot_autofocus_data(self, x_data, y_data, wavelengths, intensities):
        self.entropy_ax.clear()
        self.intensity_ax.clear()
        self.entropy_ax.set_xlabel('Z-axis position', fontsize=6, color='white')
        self.entropy_ax.set_ylabel('Entropy', fontsize=6, color='white')
        self.entropy_ax.set_title('Entropy', fontsize=8, color='white')
        self.intensity_ax.set_xlabel('Wavelength (nm)', fontsize=6, color='white')
        self.intensity_ax.set_ylabel('Intensity', fontsize=6, color='white')
        self.intensity_ax.set_title('Spectrum', fontsize=8, color='white')
        self.entropy_ax.grid(color='white', linestyle='-', linewidth=0.2)
        self.intensity_ax.grid(color='white', linestyle='-', linewidth=0.2)
        self.entropy_ax.set_facecolor("none")
        self.intensity_ax.set_facecolor("none")

        self.entropy_ax.tick_params(axis='both', which='major', labelsize=6)
        self.entropy_ax.tick_params(axis='both', which='minor', labelsize=6)
        self.entropy_ax.tick_params(axis='x', colors='white')
        self.entropy_ax.tick_params(axis='y', colors='white')
        self.intensity_ax.tick_params(axis='both', which='major', labelsize=6)
        self.intensity_ax.tick_params(axis='both', which='minor', labelsize=6)
        self.intensity_ax.tick_params(axis='x', colors='white')
        self.intensity_ax.tick_params(axis='y', colors='white')

        self.entropy_ax.plot(x_data, y_data, color='#5bc0de')
        self.entropy_ax.scatter([x_data[-1]], [y_data[-1]], color='red')
        self.intensity_ax.plot(wavelengths, intensities, color='#5bc0de')
        self.intensity_ax.lines[0].set_linewidth(0.5)

        self.wasatch_autofocus_canvas.draw_idle()
        self.wasatch_autofocus_canvas.flush_events()  # Process any pending events for real-time updates
        self.update_idletasks()

    def update_nanodrive_settings(self, event):
        self.wasatch_nanodrive_autofocus_step_num_var.set(float(self.wasatch_nanodrive_autofocus_step_num_var.get()))
        self.wasatch_autofocus_num_rep_var.set(int(self.wasatch_autofocus_num_rep_var.get()))
        self.wasatch_nanodrive_autofocus_range_low_var.set(float(self.wasatch_nanodrive_autofocus_range_low_var.get()))
        self.wasatch_nanodrive_autofocus_range_high_var.set(float(self.wasatch_nanodrive_autofocus_range_high_var.get()))

    def handle_get_nanodrive_position(self):
        self.dispatch("update_nanodrive_position", np.round(self.nano_drive.get_current_position()))
        return None

    def handle_move_nanodrive_by_request(self, distance):
        self.nano_drive.move_by(distance)
        # print("Moving nanodrive up by {distance} um")
        self.handle_get_nanodrive_position()
        return None

    def handle_move_nanodrive_to_request(self, position):
        self.nano_drive.move_to(position) # starting from the lowest position
        self.handle_get_nanodrive_position()
        return None

    # def handle_update_save_folder(self, folder_path):
    #     self.save_folder_path = folder_path
    #     return None

    def handle_measure_spectra_and_save_to_specific_folder(self, num_rep, folder_path, filename):
        wavelength, wavelengths, intensities = self.measure_spectra(num_rep)

        # flatten the intensities
        wavelengths_flatten = list(itertools.chain.from_iterable(wavelengths))
        intensities_flatten = list(itertools.chain.from_iterable(intensities))

        # Save the spectra to a file
        os.makedirs(os.path.join(folder_path, 'spectra'), exist_ok=True)
        os.makedirs(os.path.join(folder_path, 'metadata'), exist_ok=True)
        
        with open(os.path.join(folder_path, 'spectra',filename + ".txt"), "w") as outfile:
            for i in range(len(wavelengths_flatten)):
                outfile.write(f"{wavelengths_flatten[i]:0.2f}, {intensities_flatten[i]}\n")
        
        # save a metadata file
        with open(os.path.join(folder_path, 'metadata', filename + "_metadata.txt"), "w") as metafile:
            metafile.write(f"Number of repetitions: {num_rep}\n")
            metafile.write(f"Integration time (ms): {self.wasatch_manager.integ_time_ms}\n")
            metafile.write(f"Laser power (mW): {self.wasatch_manager.laser_power_mW}\n")
            metafile.write(f"Prusa position (mm): {self.best_prusa_focus_wasatch}\n")
            metafile.write(f"Nanodrive position (um): {self.best_nanodrive_focus_position}\n")
            metafile.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")

        self.dispatch("task_completed")
        return None

    def handle_get_nanodrive_min_max(self):
        return self.nano_drive.min_position_um, self.nano_drive.max_position_um

    def handle_turn_laser_on(self, wait_for_initialization):
        self.turn_laser_on()
        if wait_for_initialization=="wait":
            time.sleep(6)  # wait for the laser to turn on
        self.dispatch("task_completed")
        return None
    
    def handle_turn_laser_off(self):
        self.turn_laser_off()
        self.dispatch("task_completed")
        return None

    def handle_update_prusa_focus_range_for_wasatch(self, lower_limit, upper_limit):
        """Update the Prusa focus range for Wasatch autofocus."""
        self.wasatch_focus_prusa_lower_limit = lower_limit
        self.wasatch_focus_prusa_upper_limit = upper_limit
        print(f"Updated Prusa focus range for Wasatch: {lower_limit} to {upper_limit}")
        return

if __name__ == "__main__":
    root = ttk.Window(themename="noodlepy")
    root.title("Wasatch Raman Spectrometer")
    root.geometry("800x400")
    wasatch_autofocus_frame = WasatchAutofocusModule(root)
    wasatch_autofocus_frame.grid(row=0, column=0, sticky="nsew")
    root.mainloop()
