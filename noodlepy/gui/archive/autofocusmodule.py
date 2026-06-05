from tkinter import ttk
import ttkbootstrap as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
import numpy as np
from ctypes import *
import time
from tkinter import StringVar
from noodlepy.gui.publisher_subscriber import Subscriber, Publisher
from noodlepy.gui.wasatchmodule import WasatchModule
from noodlepy.gui.stagecontrolmodule import StageControlModule

class NanoDrive:
    def __init__(self):
        # Initialize DLL and handle
        self.dll_path = os.path.join(os.getcwd(),"noodlepy",'dlls','Madlib.dll')
        self.axis = c_uint(3) # Move along Z-axis
        self.mcldll = CDLL(self.dll_path)
        self.current_position_um = None
        self.mcldll.MCL_ReleaseHandle.restype = None
        self.mcldll.MCL_SingleReadN.restype = c_double
        self.handle = self.mcldll.MCL_InitHandle()
        if self.handle == 0:
            raise RuntimeError("Failed to initialize MCL handle. Error code: 8")
        print("MCL Handle = ", self.handle)
        self.initialize_position() # the nanodrive will initialize at 0 um

    def initialize_position(self):
        pos_um = c_double(0)
        error = self.mcldll.MCL_SingleWriteN(pos_um, self.axis, self.handle)

        time.sleep(0.025)
        if error != 0:
            raise RuntimeError(f"Nanodrive error initializing position: {error}")
        else:
            self.current_position_um = self.get_current_position()
            print("Nanodrive initialized.")


    def get_current_position(self):
        position = self.mcldll.MCL_SingleReadN(self.axis, self.handle)
        return position

    def move_to(self, z_pos_um):
        if type(z_pos_um) != float:
            z_pos_um = float(z_pos_um)
        
        z_pos_um = max(0, min(z_pos_um, 100)) # Ensure new_position stays within bounds [0, 100]um
        pos = c_double(z_pos_um)
        error = self.mcldll.MCL_SingleWriteN(pos, self.axis, self.handle)

        time.sleep(0.025)  # Wait for nanopositioner to settle
        if error != 0:
            print("Nanodrive move_to_position Error =", error)
            print(f"Nanodrive attempting to move to position: {z_pos_um} on axis: {self.axis.value} with handle: {self.handle}")
        else:
            self.current_position_um = self.get_current_position()
            # print(f"Nanodrive moved to position: {self.current_position_um}.")

    def move_by(self, delta_z_um: float):
        # Get the current position
        current_position_um = self.mcldll.MCL_SingleReadN(self.axis, self.handle)
     
        # Calculate the new position
        new_position_um = current_position_um + delta_z_um
        new_position_um = max(0, min(new_position_um, 100)) # Ensure new_position stays within bounds [0, 100]
        new_position_c_double_um = c_double(new_position_um)
        
        time.sleep(0.025)
        error = self.mcldll.MCL_SingleWriteN(new_position_c_double_um, self.axis, self.handle)
        if error != 0:
            raise RuntimeError(f"MCL Error: {error}")
        else:
            self.current_position_um = self.get_current_position()
            # print(f"Nanodrive moved by {delta_z_um} to position: {self.current_position_um}.")

    def close(self):
        self.mcldll.MCL_ReleaseHandle(self.handle)
        print("NanoDrive shutdown")

class AutoFocusModule(ttk.Frame, Publisher, Subscriber):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        Publisher.__init__(self, ["update_nanodrive_position"])
        Subscriber.__init__(self)
        self.name = "AutoFocusModule_obserableobserver"

        self.nano_drive = NanoDrive()
        # self.spectrum_updated = False
        self.spectrum = None
        self.wavelengths = None        
        self.create_widgets_for_autofocus()

    def create_widgets_for_autofocus(self):

        self.nanodrive_frame = ttk.Labelframe(self, text='Auto Focus', padding=5)
        self.nanodrive_frame.grid(row=1, column=0, columnspan=5, sticky="nsew", padx=5, pady=5)

        self.autofocus_fig, self.entropy_ax, self.intensity_ax = self.initialize_autofocus_figure()
        self.autofocus_canvas = FigureCanvasTkAgg(self.autofocus_fig, master=self.nanodrive_frame)
        self.autofocus_canvas.draw()
        self.autofocus_canvas.get_tk_widget().grid(row=0, column=0, columnspan=5, rowspan=2, sticky="nsew", padx=5, pady=5)

        # create initial values for autofocus parameters
        self.autofocus_step_size_var = StringVar()
        self.autofocus_step_size_var.set(10)
        self.autofocus_num_rep_var = StringVar()
        self.autofocus_num_rep_var.set(5)
        self.autofocus_range_low_var = StringVar()
        self.autofocus_range_low_var.set(0)
        self.autofocus_range_high_var = StringVar()
        self.autofocus_range_high_var.set(100)
        
        # Create the entries for the autofocus parameters
        self.autofocus_step_size_label = ttk.Label(self.nanodrive_frame, text="Number of Steps")
        self.autofocus_step_size_label.grid(row=2, column=0, pady=5, padx=5)
        self.autofocus_step_size_entry = ttk.Entry(self.nanodrive_frame, width=5, textvariable=self.autofocus_step_size_var)
        self.autofocus_step_size_entry.grid(row=2, column=1, padx=5, pady=5)
        self.autofocus_step_size_entry.bind("<FocusOut>", self.update_settings)
        self.autofocus_step_size_entry.bind("<Return>", self.update_settings)

        self.autofocus_num_rep_label = ttk.Label(self.nanodrive_frame, text="Number of Reps")
        self.autofocus_num_rep_label.grid(row=2, column=2, pady=5, padx=5)
        self.autofocus_num_rep_entry = ttk.Entry(self.nanodrive_frame, width=5, textvariable=self.autofocus_num_rep_var)
        self.autofocus_num_rep_entry.grid(row=2, column=3, padx=5, pady=5)
        self.autofocus_num_rep_entry.bind("<FocusOut>", self.update_settings)
        self.autofocus_num_rep_entry.bind("<Return>", self.update_settings)
    
        self.autofocus_range_label = ttk.Label(self.nanodrive_frame, text="Z Min [um]")
        self.autofocus_range_label.grid(row=3, column=0,pady=5, padx=5)
        self.autofocus_range_low_entry = ttk.Entry(self.nanodrive_frame, width=5, textvariable=self.autofocus_range_low_var)
        self.autofocus_range_low_entry.grid(row=3, column=1, padx=5, pady=5)
        self.autofocus_range_low_entry.bind("<FocusOut>", self.update_settings)
        self.autofocus_range_low_entry.bind("<Return>", self.update_settings)

        self.autofocus_range_label = ttk.Label(self.nanodrive_frame, text="Z Max [um]")
        self.autofocus_range_label.grid(row=3, column=2,pady=5, padx=5)
        self.autofocus_range_low_entry = ttk.Entry(self.nanodrive_frame, width=5, textvariable=self.autofocus_range_high_var)
        self.autofocus_range_low_entry.grid(row=3, column=3, padx=5, pady=5)
        self.autofocus_range_low_entry.bind("<FocusOut>", self.update_settings)
        self.autofocus_range_low_entry.bind("<Return>", self.update_settings)

        self.autofocus_button = ttk.Button(self.nanodrive_frame, text="Focus", command=self.autofocus, bootstyle="info", width=5)
        self.autofocus_button.grid(row=2, column=4, rowspan=2, padx=5, pady=5, sticky="news")

    def initialize_autofocus_figure(self):
                # Create the figure and axes
        fig, axes = plt.subplots(1, 2, figsize=(4, 2))

        entropy_ax = axes[0]
        intensity_ax = axes[1]

        entropy_ax.set_xlabel('Z-axis position',fontsize=6, color='white')
        entropy_ax.set_ylabel('Entropy', fontsize=6, color='white')
        entropy_ax.set_title('Entropy', fontsize=8, color='white')

        intensity_ax.set_xlabel('Z-axis position', fontsize=6, color='white')
        intensity_ax.set_ylabel('Intensity', fontsize=6, color='white')
        intensity_ax.set_title('Spectrum', fontsize=8, color='white')

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

    def autofocus(self):
        # Rough Autofocusing
        min = float(self.autofocus_range_low_var.get())
        max = float(self.autofocus_range_high_var.get())
        step_size = int(float(self.autofocus_step_size_var.get()))
        num_rep = int(float(self.autofocus_num_rep_var.get()))

        z_axis_range = np.linspace(min, max, step_size)
        entropy_list, x1_data, y1_data= [], [], []

        for i, z_pos in enumerate(z_axis_range):
            self.nano_drive.move_to(z_pos)
            self.dispatch("update_nanodrive_position", self.nano_drive.get_current_position())

            # # Wait for a new spectrum update
            # self.wait_for_spectrum_update()

            position, wavelengths, intensities = self.measure_spectra(num_rep)
            ent = self.calculate_entropy(intensities)
            entropy_list.append(ent)
            x1_data.append(position)
            y1_data.append(ent)

            # Plot entropy and spectrum
            self.plot_autofocus_data(x1_data, y1_data, wavelengths, np.mean(intensities, 0))

        fineMin, fineMax = self.refine_focus_range(entropy_list, z_axis_range)
        self.fine_autofocus(fineMin, fineMax, num_rep)

    def fine_autofocus(self, fineMin, fineMax, num_rep=5):
        z_axis_range = np.linspace(fineMin, fineMax, 10)
        entropy_list, x2_data, y2_data = [], [], []

        for i, z_pos in enumerate(z_axis_range):
            self.nano_drive.move_to(z_pos)
            self.dispatch("update_nanodrive_position", self.nano_drive.get_current_position())

            position, wavelengths, intensities = self.measure_spectra(num_rep)
            ent = self.calculate_entropy(intensities)
            entropy_list.append(ent)
            x2_data.append(position)
            y2_data.append(ent)

            # Plot fine focus data
            self.plot_autofocus_data(x2_data, y2_data, wavelengths, np.mean(intensities, 0))

        focusedPos = z_axis_range[np.argmin(entropy_list)]
        self.nano_drive.move_to(focusedPos)
        self.dispatch("update_nanodrive_position", self.nano_drive.get_current_position())

        print("Focused Position =", focusedPos)
        print("Autofocus complete")

    def measure_spectra(self, num_rep):
        intensities, wavelengths = [], []
        wavelengths = self.get_wavelengths()
        for _ in range(num_rep):
            spectrum = self.get_spectrum()
            intensities.append(spectrum)

        lengths = [len(seq) for seq in intensities]
        print("Lengths of sequences in intensities:", lengths)

        intensities = np.array(intensities).reshape((num_rep, len(wavelengths)))
        position = self.nano_drive.get_current_position()
        return position, wavelengths, intensities

    def calculate_entropy(self, intensities):
        median_intensities = np.median(intensities, 0)
        normalized_int = median_intensities / np.sum(median_intensities)
        entropy = -np.sum(normalized_int * np.log2(normalized_int))
        return entropy

    def refine_focus_range(self, entropy_list, z_axis_range):
        k = np.argmin(entropy_list)
        fineMin = z_axis_range[np.clip(k - 1, 0, len(z_axis_range) - 1)]
        fineMax = z_axis_range[np.clip(k + 1, 0, len(z_axis_range) - 1)]
        print("The fine range is:", fineMin, fineMax)
        return fineMin, fineMax

    def update_settings(self, event):
        self.autofocus_step_size_var.set(float(self.autofocus_step_size_var.get()))
        self.autofocus_num_rep_var.set(int(self.autofocus_num_rep_var.get()))
        self.autofocus_range_low_var.set(float(self.autofocus_range_low_var.get()))
        self.autofocus_range_high_var.set(float(self.autofocus_range_high_var.get()))

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

        self.autofocus_canvas.draw()
        self.autofocus_canvas.flush_events()  # Process any pending events for real-time updates
        self.update_idletasks()

    def handle_update_spectrum(self, updated_from_wasatch):
        # Update the spectrum and wavelengths upon new spectrum data arrival
        self.spectrum = updated_from_wasatch["spectrum"]
        self.wavelengths = updated_from_wasatch["wavelengths"]
        self.spectrum_updated = True  # Indicate a new spectrum is ready

    def wait_for_spectrum_update(self):
        # Wait until a new spectrum update is received
        while not self.spectrum_updated:
            time.sleep(0.01)  # Small delay to avoid excessive CPU usage
        self.spectrum_updated = False  # Reset the flag after using the new spectrum

    def handle_move_nanodrive_by_request(self, distance):
        self.nano_drive.move_by(distance)
        # print("Moving nanodrive up by {distance} um")
        self.dispatch("update_nanodrive_position", self.nano_drive.get_current_position())

    def handle_move_nanodrive_to_request(self, position):
        self.nano_drive.move_to(position) # starting from the lowest position
        self.dispatch("update_nanodrive_position", self.nano_drive.get_current_position())

    def get_spectrum(self):
        return self.spectrum
    
    def get_wavelengths(self):
        return self.wavelengths


if __name__ == '__main__':
    root = ttk.Window()
    root.style.theme_use('noodlepy')

    stage_control_module = StageControlModule(root)
    stage_control_module.grid(row=0, column=1, sticky="nsew")

    wasatch_module = WasatchModule(root)
    wasatch_module.grid(row=0, column=0, sticky="nsew")

    autofocus_module = AutoFocusModule(root)
    autofocus_module.grid(row=1, column=0, sticky="nsew")

    wasatch_module.add_subscriber('update_spectrum', autofocus_module, autofocus_module.handle_update_spectrum)
    root.mainloop()
