import os
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from noodlepy.utils.spectrum import Spectrum
from tkinter import filedialog, Text, END, RIGHT, LEFT, Y, BOTH, VERTICAL
from ttkbootstrap.constants import *
import ttkbootstrap as ttk
import yaml
import copy
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from noodlepy.utils.spectrumcalibrator import SpectrumCalibrator

class AnalysisGUI(ttk.Frame):
    def __init__(self, parent, file_types='*.txt'):
        super().__init__(parent)
        self.file_types = file_types if file_types else ["*"]
        self.config_path = None
        self.neon_file_path = None
        self.white_lamp_file_path = None
        self.folder_path = None
        self.create_widgets()

    def create_widgets(self):
        control_frame = ttk.Frame(self)
        control_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        plot_frame = ttk.Frame(self)
        plot_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        data_frame = ttk.Labelframe(control_frame, text='Load Data', padding=5)
        data_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        self.folder_path_entry = ttk.Entry(data_frame)
        self.folder_path_entry.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.browse_button = ttk.Button(data_frame, text="Browse", command=self.browse_folder)
        self.browse_button.grid(row=0, column=1, padx=5, pady=5)
        self.select_all_button = ttk.Button(data_frame, text="Select All", command=self.select_all)
        self.select_all_button.grid(row=0, column=2, padx=5, pady=5)
        self.scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL)
        self.file_listbox = ttk.Treeview(data_frame, yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.file_listbox.yview)
        self.file_listbox.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        self.scrollbar.grid(row=1, column=3, sticky="ns")
        self.file_listbox.bind('<<TreeviewSelect>>', self.update_plots)

        calibration_frame = ttk.Labelframe(control_frame, text='Calibration', padding=5)
        calibration_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        self.neon_file_entry = ttk.Entry(calibration_frame)
        self.neon_file_entry.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.load_neon_file_button = ttk.Button(calibration_frame, text="Load Neon File",command = self.load_neon_file, width=15)
        self.load_neon_file_button.grid(row=0, column=1, padx=5, pady=5)

        self.white_lamp_file_entry = ttk.Entry(calibration_frame)
        self.white_lamp_file_entry.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.load_white_lamp_file_button = ttk.Button(calibration_frame, text="Load White Lamp File", command = self.load_white_lamp_file, width=15)
        self.load_white_lamp_file_button.grid(row=1, column=1, padx=5, pady=5)

        self.calibration_x_axis_checkbutton_var = tk.BooleanVar(value=False)
        self.calibration_x_checkbutton_axis = ttk.Checkbutton(calibration_frame, text="X axis", variable=self.calibration_x_axis_checkbutton_var, command=self.plot_processed_spectra, state=DISABLED)
        self.calibration_x_checkbutton_axis.grid(row=2, column=0, sticky = 'ew', padx=5, pady=5)

        self.calibration_y_axis_checkbutton_var = tk.BooleanVar(value=False)
        self.calibration_y_checkbutton_axis = ttk.Checkbutton(calibration_frame, text="Y axis", variable=self.calibration_y_axis_checkbutton_var, command=self.plot_processed_spectra, state=DISABLED)
        self.calibration_y_checkbutton_axis.grid(row=2, column=1, sticky = 'ew', padx=5, pady=5)


        preprocess_frame = ttk.Labelframe(control_frame, text='Preprocess', padding=5)
        preprocess_frame.grid(row=3, column=1, columnspan=3, sticky="ew", padx=5, pady=5)

        self.config_path_entry = ttk.Entry(preprocess_frame)
        self.config_path_entry.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.load_button = ttk.Button(preprocess_frame, text="Load Config", command=self.load_config, width=15)
        self.load_button.grid(row=0, column=2, padx=5, pady=5)

        self.cropping_checkbutton_var = tk.BooleanVar(value=False)
        self.cropping_checkbutton = ttk.Checkbutton(preprocess_frame, text="Cropping", variable=self.cropping_checkbutton_var, command=self.plot_processed_spectra, state=DISABLED)
        self.cropping_checkbutton.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self.baseline_correction_checkbutton_var = tk.BooleanVar(value=False)
        self.baseline_correction_checkbutton = ttk.Checkbutton(preprocess_frame, text="Baseline Correction", variable=self.baseline_correction_checkbutton_var, command=self.plot_processed_spectra, state=DISABLED)
        self.baseline_correction_checkbutton.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        self.smoothing_checkbutton_var = tk.BooleanVar(value=False)
        self.smoothing_checkbutton = ttk.Checkbutton(preprocess_frame, text="Smoothing", variable=self.smoothing_checkbutton_var, command=self.plot_processed_spectra, state=DISABLED)
        self.smoothing_checkbutton.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        self.normalization_checkbutton_var = tk.BooleanVar(value=False)
        self.normalization_checkbutton = ttk.Checkbutton(preprocess_frame, text="Normalization", variable=self.normalization_checkbutton_var, command=self.plot_processed_spectra, state=DISABLED)
        self.normalization_checkbutton.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

        self.despiking_checkbutton_var = tk.BooleanVar(value=False)
        self.despiking_checkbutton = ttk.Checkbutton(preprocess_frame, text="Despiking", variable=self.despiking_checkbutton_var, command=self.plot_processed_spectra, state=DISABLED)
        self.despiking_checkbutton.grid(row=5, column=0, sticky="ew", padx=5, pady=5)


        self.fig_raw, self.ax_raw = plt.subplots(figsize=(3, 1.5), tight_layout=True)
        self.ax_raw.set_xlabel('Raman Shift (cm^-1)', fontsize=4, labelpad=0.5)
        self.ax_raw.set_ylabel('Intensity',fontsize=4, labelpad=0.5)
        self.ax_raw.set_title('Spectra',fontsize=6, pad=0.5)
        self.ax_raw.tick_params(axis='both', which='major', labelsize=4, pad=0.8, length=0)
        self.ax_raw.tick_params(axis='both', which='minor', labelsize=4, pad=0.8, length=0)
        self.ax_raw.margins(x=0)
        self.ax_raw.margins(y=0)
        self.ax_raw.figure.tight_layout(pad=0.1)
        self.ax_raw.spines['top'].set_visible(False)
        self.ax_raw.spines['right'].set_visible(False)
        self.canvas_raw = FigureCanvasTkAgg(self.fig_raw, master=plot_frame)
        self.canvas_raw.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        self.fig_processed, self.ax_processed = plt.subplots(figsize=(3, 1.5), tight_layout=True)
        self.ax_processed.set_xlabel('Raman Shift (cm^-1)', fontsize=4, labelpad=0.5)
        self.ax_processed.set_ylabel('Intensity',fontsize=4, labelpad=0.5)
        self.ax_processed.set_title('Processed Spectra',fontsize=6, pad=0.5)
        self.ax_processed.tick_params(axis='both', which='major', labelsize=4, pad=0.8, length=0)
        self.ax_processed.tick_params(axis='both', which='minor', labelsize=4, pad=0.8, length=0)
        self.ax_processed.margins(x=0)
        self.ax_processed.margins(y=0)
        self.ax_processed.figure.tight_layout(pad=0.1)
        self.ax_processed.spines['top'].set_visible(False)
        self.ax_processed.spines['right'].set_visible(False)
        self.canvas_processed = FigureCanvasTkAgg(self.fig_processed, master=plot_frame)
        self.canvas_processed.get_tk_widget().grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

    def update_plots(self, event):
        self.plot_raw_spectra()
        self.plot_processed_spectra()

    def browse_folder(self):
        current_folder = os.getcwd()+ '/noodlepy' +'/data'
        folder_path = filedialog.askdirectory(initialdir=current_folder)
        if folder_path:
            self.folder_path_entry.delete(0, tk.END)
            self.folder_path_entry.insert(0, folder_path)
            self.load_files()

    def select_all(self):
        for item in self.file_listbox.get_children():
            self.file_listbox.selection_add(item)

    def load_files(self):
        self.folder_path = self.folder_path_entry.get()
        if self.folder_path:
            self.file_listbox.delete(*self.file_listbox.get_children())
            for root, dirs, files in os.walk(self.folder_path):
                for file in files:
                    if any(file.endswith(ft) for ft in self.file_types):
                        file_path = os.path.join(root, file)
                        self.file_listbox.insert('', 'end', iid=file_path, text=file)

    def load_neon_file(self):
        current_folder = os.getcwd()+ '/noodlepy' +'/data'
        self.neon_file_path = filedialog.askopenfilename(initialdir=current_folder)
        if self.neon_file_path:
            self.neon_file_entry.delete(0, tk.END)
            self.neon_file_entry.insert(0, self.neon_file_path)
            self.calibration_x_checkbutton_axis.config(state=NORMAL)

    def load_white_lamp_file(self):
        current_folder = os.getcwd()+ '/noodlepy' +'/data'
        self.white_lamp_file_path = filedialog.askopenfilename(initialdir=current_folder)
        if self.white_lamp_file_path:
            self.white_lamp_file_entry.delete(0, tk.END)
            self.white_lamp_file_entry.insert(0, self.white_lamp_file_path)
            self.calibration_y_checkbutton_axis.config(state=NORMAL)


    def load_config(self):
        current_folder = os.getcwd()+ '/noodlepy' +'/config'
        self.config_path = filedialog.askopenfilename(initialdir=current_folder)

        if self.config_path:
            self.config_path_entry.delete(0, tk.END)
            self.config_path_entry.insert(0, self.config_path)
            self.cropping_checkbutton.config(state=NORMAL)
            self.baseline_correction_checkbutton.config(state=NORMAL)
            self.smoothing_checkbutton.config(state=NORMAL)
            self.normalization_checkbutton.config(state=NORMAL)
            self.despiking_checkbutton.config(state=NORMAL)


    def plot_raw_spectra(self):
        selected_files = self.file_listbox.selection()
        list_of_spectra = []
        self.ax_raw.clear()

        for file in selected_files:
            file_path = file
            spectra = Spectrum.load_from_file(file_path)
            list_of_spectra.append(spectra)

            for spectrum in spectra:
                raman_shift_cm = spectrum.raman_shift_cm
                intensity = spectrum.intensity
                self.ax_raw.plot(raman_shift_cm, intensity, linewidth=0.5)

        self.ax_raw.set_xlabel('Raman Shift (cm^-1)', fontsize=4)
        self.ax_raw.set_ylabel('Intensity',fontsize=4)
        self.ax_raw.set_title('Raw Spectra',fontsize=6)
        self.ax_raw.tick_params(axis='both', which='major', labelsize=4)
        self.ax_raw.tick_params(axis='both', which='minor', labelsize=4)
        self.ax_raw.margins(x=0)
        self.ax_raw.margins(y=0)
        self.ax_raw.figure.tight_layout(pad=0.1)
        self.ax_raw.spines['top'].set_visible(False)
        self.ax_raw.spines['right'].set_visible(False)
        self.canvas_raw.draw()


    def plot_processed_spectra(self):
        calibrator = None
        preprocessor = None

        if self.neon_file_path is not None and self.white_lamp_file_path is not None:
            calibrate_x_axis = self.calibration_x_axis_checkbutton_var.get()
            calibrate_y_axis = self.calibration_y_axis_checkbutton_var.get()
            calibrator = SpectrumCalibrator(
                measured_neon_file=self.neon_file_path, 
                measured_white_lamp=self.white_lamp_file_path, 
                standard_neon_file=None, 
                calibrate_x = calibrate_x_axis, 
                calibrate_y = calibrate_y_axis)

        if self.config_path is not None:
            cropping = self.cropping_checkbutton_var.get()
            baseline_correction = self.baseline_correction_checkbutton_var.get()
            remove_cosmic_rays = self.despiking_checkbutton_var.get()
            normalization = self.normalization_checkbutton_var.get()
            smoothing = self.smoothing_checkbutton_var.get()
            preprocessor = SpectrumPreprocessor(
                cropping= cropping,
                baseline_correction=baseline_correction,
                remove_cosmic_rays= remove_cosmic_rays,
                normalization= normalization,
                smoothing= smoothing,
                config_path=self.config_path
            )

        selected_files = self.file_listbox.selection()
        list_of_spectra = []
        self.ax_processed.clear()

        for file in selected_files:
            file_path = file
            spectra = Spectrum.load_from_file(file_path)
            list_of_spectra.append(spectra)

            for spectrum in spectra:
                preprocessed_spectrum = copy.deepcopy(spectrum)

                if calibrator:
                    preprocessed_spectrum = calibrator.calibrate(preprocessed_spectrum)

                if preprocessor:
                    preprocessed_spectrum = preprocessor.preprocess(preprocessed_spectrum)

                raman_shift_cm = preprocessed_spectrum.raman_shift_cm
                intensity = preprocessed_spectrum.intensity
                self.ax_processed.plot(raman_shift_cm, intensity, linewidth=0.5)

        self.ax_processed.set_xlabel('Raman Shift (cm^-1)', fontsize=4)
        self.ax_processed.set_ylabel('Intensity',fontsize=4)
        self.ax_processed.set_title('Processed Spectra',fontsize=6)
        self.ax_processed.tick_params(axis='both', which='major', labelsize=4)
        self.ax_processed.tick_params(axis='both', which='minor', labelsize=4)
        self.ax_processed.margins(x=0)
        self.ax_processed.margins(y=0)
        self.ax_processed.figure.tight_layout(pad=0.1)
        self.ax_processed.spines['top'].set_visible(False)
        self.ax_processed.spines['right'].set_visible(False)
        self.canvas_processed.draw()


if __name__ == '__main__':
    root = ttk.Window()
    root.style.theme_use('superhero')
    app = AnalysisGUI(root)
    app.pack()
    root.mainloop()