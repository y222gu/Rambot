import threading
import time
from tkinter import Tk, Button, Label, Entry, StringVar, filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from noodlepy.gui.publisher_subscriber import Publisher, Subscriber
import os
from concurrent.futures import ThreadPoolExecutor
import queue

# Protocol Manager
class ProtocolModule(Publisher, ttk.Frame):
    def __init__(self, parent):
        Publisher.__init__(self, ["update_nanodrive_position",
                                  "check_current_camera_view",
                                  "move_stage_to_target_sample_drop_during_aquisition",
                                  "focus_widefield_camera",
                                  "focus_wasatch_with_prusa",
                                  "focus_wasatch_with_nanodrive",
                                  "capture_current_image_and_detect_sample_drop_and_create_sampling_points",
                                  "call_switch_view_button_in_live_camera_module",
                                  "move_to_a_single_sampling_point",
                                  "measure_spectra_and_save_to_specific_folder",
                                  "reposition_stage_and_nanodrive_in_objective_view", 
                                  "turn_on_laser",
                                  "turn_off_laser"]) # "update_capture_save_folder"
        ttk.Frame.__init__(self, parent)

        self.name = "ProtocolModule"
        self.parent = parent
        self.focus_range_is_setup = False
        self.stage_reference_is_setup = False

        self.state = "IDLE"   # The initial state
        self.task_queue = queue.Queue()  # Queue for tasks
        self.current_task = None  # Keep track of the current task
        self.abort_flag = False  # Flag to abort the current task
        self.current_task_generator = None  # Track the task generator

        self.protocol_canvas_circle_spacing = 80  # Enlarge the spacing for better visualization
        self.protocol_canvas_circle_radius = 30 # Enlarge the circle radius for better visualization
        self.protocol_canvas_grid_pattern = [4,4,4,4,4]  # Column-wise circle counts for symmetry
        # calculate the canvas size based on the circle spacing and radius
        self.protocol_canvas_width = len(self.protocol_canvas_grid_pattern) * self.protocol_canvas_circle_spacing
        self.protocol_canvas_height = max(self.protocol_canvas_grid_pattern) * self.protocol_canvas_circle_spacing
        self.selected_circles = set()  # Track selected circles
        self.selected_circle_positions = []  # Track selected circle positions
        self.number_of_spectra = 3
        self.sampling_points_relative_distance_to_camera_center = None

        self.create_widgets()
        # Periodically check the state machine
        self.check_state_machine()

    def check_state_machine(self):
        # Check abort flag first, regardless of state
        if self.abort_flag:
            if self.state != "IDLE":  # Only cleanup if not already idle
                self.handle_abort_cleanup()
                self.state = "IDLE"
        
        elif self.state == "IDLE":
            if not self.task_queue.empty() and not self.abort_flag:
                self.current_task = self.task_queue.get()
                self.state = "BUSY"
                self.execute_task()

        elif self.state == "BUSY":
            # Wait for task completion; the task will update the state itself.
            pass

        elif self.state == "TASK_COMPLETED":
            self.state = "IDLE"  # Return to idle, next task will be handled in next cycle

        elif self.state == "ABORT":
            self.handle_abort_cleanup()
            self.state = "IDLE"  # Reset state to IDLE

        # Continue checking the state machine after a short delay (50 ms)
        self.after(50, self.check_state_machine)

    def execute_task(self):
        if not self.abort_flag:  # Check abort flag before executing
            event, args = self.current_task
            self.dispatch(event, *args)

    def enqueue_task(self, event, *args):
        if not self.abort_flag:  # Don't enqueue if aborting
            self.task_queue.put((event, args))

    def clear_task_queue(self):
        """Clear the task queue safely."""
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break

    def start_aquisition(self):
        print("Checking setup status")
        if not self.focus_range_is_setup:
            print("Focus range not setup")
            return
        if not self.stage_reference_is_setup:
            print("Stage not referenced")
            return
        
        print("Executing protocol")
        self.abort_flag = False  # Reset abort flag
        if self.generate_sample_points_upfront_var.get() == "True":
            print("Generating sampling points upfront.")
            self.current_task_generator = self.task_sequence_generator_sampling_all_points_upfront(self.selected_circle_positions)
        else:
            self.current_task_generator = self.task_sequence_generator_one_by_one(self.selected_circle_positions)
        
        self.enqueue_next_task()

    def enqueue_next_task(self):
        if self.abort_flag:
            print("Aborting task generation.")
            return
        
        # Check if generator exists before trying to use it
        if self.current_task_generator is None:
            print("No task generator available.")
            return
            
        try:
            next_task = next(self.current_task_generator)
            self.enqueue_task(*next_task)
        except StopIteration:
            print("All tasks for the current sample drop are completed.")
            self.current_task_generator = None

    def task_sequence_generator_one_by_one(self, selected_circle_positions):
        for i_sample_drop, sample_drop_row_column in enumerate(selected_circle_positions):
            if self.abort_flag:
                print("Aborting task sequence at sample drop", i_sample_drop)
                break
                
            yield ('check_current_camera_view', "WIDEFIELD")
            if self.abort_flag: break
            
            print('Moving the sample drop at row', sample_drop_row_column[0], 'column', sample_drop_row_column[1])
            yield ("move_stage_to_target_sample_drop_during_aquisition", 'WIDEFIELD', sample_drop_row_column)
            if self.abort_flag: break

            if self.autofocus_widefield_var.get() == "True":
                yield ("focus_widefield_camera",)
                if self.abort_flag: break

            yield ("capture_current_image_and_detect_sample_drop_and_create_sampling_points",)
            if self.abort_flag: break
            
            yield ("call_switch_view_button_in_live_camera_module", "TO_OBJECTIVE")
            if self.abort_flag: break
            
            yield ("check_current_camera_view", "OBJECTIVE")
            if self.abort_flag: break
            
            print("Sampling points relative to camera center:", self.sampling_points_relative_distance_to_camera_center)

            # turn on laser
            if i_sample_drop == 0:
                yield ("turn_on_laser", "wait")
                print("Turning on laser for the first sample drop and first sampling point.")
                print("Waiting for 6 seconds to allow the laser to stabilize.")# Wait for the laser to stabilize
            else:
                yield ("turn_on_laser", "no_wait") # no wait
            if self.abort_flag: break

            for i_rep in range(int(self.number_of_repeats_var.get())):

                for i_sampling_point, point in enumerate(self.sampling_points_relative_distance_to_camera_center):
                    if self.abort_flag:
                        print("Aborting task sequence at sampling point", i_sampling_point)
                        break

                    print(f"Measuring #{i_sampling_point} point at position {point}")
                    yield ("move_to_a_single_sampling_point", sample_drop_row_column, point)
                    if self.abort_flag: break

                    if self.wasatch_autofocus_with_prusa_var.get() == "True":
                        print("Autofocusing Wasatch with Prusa for the first sampling point.")
                        print("This step is only performed once per sample drop.")
                        yield ("focus_wasatch_with_prusa",)
                        if self.abort_flag: break

                    if self.wasatch_autofocus_with_nanodrive_var.get() == "True":
                        yield ("focus_wasatch_with_nanodrive",)
                        if self.abort_flag: break

                    folder = os.path.join(self.folder_path.get())
                    if not os.path.exists(folder):
                        os.makedirs(folder)

                    coordinates_order = self.coordinates_order[i_sampling_point]
                    # join keys and values (key then value) with "_" sorted by ascending key
                    sorted_items = sorted(coordinates_order.items(), key=lambda item: item[0])
                    parts = []
                    for k, v in sorted_items:
                        parts.append(str(k))
                        parts.append(f"{v}" if isinstance(v, (int, float)) else str(v))
                    coordicates_str = "_".join(parts)

                    filename = f"{self.file_base_name.get()}_sample_{i_sample_drop}_point_{i_sampling_point}_rep_{i_rep+1}_{coordicates_str}_x{point[0]:.2f}_y{point[1]:.2f}"
                    yield ("measure_spectra_and_save_to_specific_folder", self.number_of_spectra_var.get(), folder, filename)
                    if self.abort_flag: break
                    print(f"Data saved to {folder}")

            print("All points for the current sample drop are completed.")

            yield ('turn_off_laser',)
            if self.abort_flag: break
            
            yield ("reposition_stage_and_nanodrive_in_objective_view",)
            if self.abort_flag: break
            
            yield ("call_switch_view_button_in_live_camera_module", "TO_WIDE")
            if self.abort_flag: break
            
            yield ("check_current_camera_view", "WIDEFIELD")
            if self.abort_flag: break

    def task_sequence_generator_sampling_all_points_upfront(self, selected_circle_positions):
        sampling_points_for_all_drops = []
        coordinates_order_for_all_drops = []

        for i_sample_drop, sample_drop_row_column in enumerate(selected_circle_positions):
            if self.abort_flag:
                print("Aborting task sequence at sample drop", i_sample_drop)
                break
                
            yield ('check_current_camera_view', "WIDEFIELD")
            if self.abort_flag: break
            
            print('Moving the sample drop at row', sample_drop_row_column[0], 'column', sample_drop_row_column[1])
            yield ("move_stage_to_target_sample_drop_during_aquisition", 'WIDEFIELD', sample_drop_row_column)
            if self.abort_flag: break

            if self.autofocus_widefield_var.get() == "True":
                yield ("focus_widefield_camera",)
                if self.abort_flag: break

            yield ("capture_current_image_and_detect_sample_drop_and_create_sampling_points",)
            if self.abort_flag: break
            
            print("Sampling points relative to camera center:", self.sampling_points_relative_distance_to_camera_center)
            sampling_points_for_all_drops.append(self.sampling_points_relative_distance_to_camera_center)
            coordinates_order_for_all_drops.append(self.coordinates_order)

        # move back to the first droplet before starting measurements
        yield ("move_stage_to_target_sample_drop_during_aquisition", "WIDEFIELD", selected_circle_positions[0])
        yield ("call_switch_view_button_in_live_camera_module", "TO_OBJECTIVE")
        yield ("check_current_camera_view", "OBJECTIVE")

        # Now loop again to do the measurements for all sample drops
        for i_rep in range(int(self.number_of_repeats_var.get())):
            if self.abort_flag: break

            for i_sample_drop, sample_drop_row_column in enumerate(selected_circle_positions):
                # yield ("move_stage_to_target_sample_drop_during_aquisition", 'OBJECTIVE', sample_drop_row_column)
                # if self.abort_flag: break

                # turn on laser
                if i_sample_drop == 0 and i_rep == 0:
                    yield ("turn_on_laser", "wait")
                    print("Turning on laser for the first sample drop and first sampling point.")
                    print("Waiting for 6 seconds to allow the laser to stabilize.")# Wait for the laser to stabilize
                else:
                    yield ("turn_on_laser", "no_wait") # no wait
                if self.abort_flag: break

                for i_sampling_point, point in enumerate(sampling_points_for_all_drops[i_sample_drop]):
                    if self.abort_flag: break

                    print(f"Measuring #{i_sampling_point} point at position {point}")
                    yield ("move_to_a_single_sampling_point", sample_drop_row_column, point)
                    if self.abort_flag: break

                    if self.wasatch_autofocus_with_prusa_var.get() == "True":
                        print("Autofocusing Wasatch with Prusa for the first sampling point.")
                        print("This step is only performed once per sample drop.")
                        yield ("focus_wasatch_with_prusa",)
                        if self.abort_flag: break

                    if self.wasatch_autofocus_with_nanodrive_var.get() == "True":
                        yield ("focus_wasatch_with_nanodrive",)
                        if self.abort_flag: break

                    folder = os.path.join(self.folder_path.get())
                    if not os.path.exists(folder):
                        os.makedirs(folder)

                    coordinates_order = coordinates_order_for_all_drops[i_sample_drop][i_sampling_point]
                    # join keys and values (key then value) with "_" sorted by ascending key
                    sorted_items = sorted(coordinates_order.items(), key=lambda item: item[0])
                    parts = []
                    for k, v in sorted_items:
                        parts.append(str(k))
                        parts.append(f"{v}" if isinstance(v, (int, float)) else str(v))
                    coordicates_str = "_".join(parts)

                    filename = f"{self.file_base_name.get()}_sample_{i_sample_drop}_point_{i_sampling_point}_rep_{i_rep+1}_{coordicates_str}_x{point[0]:.2f}_y{point[1]:.2f}"
                    yield ("measure_spectra_and_save_to_specific_folder", self.number_of_spectra_var.get(), folder, filename)
                    if self.abort_flag: break
                    print(f"Data saved to {folder}")

            print("All points for the current sample drop are completed.")

        yield ('turn_off_laser',)
        if self.abort_flag: return

        # yield ("reposition_stage_and_nanodrive_in_objective_view",)
        
        yield ("call_switch_view_button_in_live_camera_module", "TO_WIDE")
        if self.abort_flag: return

        yield ("check_current_camera_view", "WIDEFIELD")
        if self.abort_flag: return

    def handle_task_completed(self):
        # Check if we're aborting or if generator is None
        if self.abort_flag or self.current_task_generator is None:
            self.state = "IDLE"
            return
            
        self.state = "TASK_COMPLETED"
        self.enqueue_next_task()

    def handle_abort_aquisition(self):
        print("Abort acquisition requested.")
        self.abort_flag = True
        self.state = "ABORT"
        
        # Clear the task queue
        self.clear_task_queue()
        
        # Stop the task generator
        if self.current_task_generator:
            self.current_task_generator = None
            
        print("Acquisition abort initiated.")

    def handle_abort_cleanup(self):
        """Cleanup tasks when aborting"""
        print("Performing abort cleanup...")
        
        # Stop the task generator first
        self.current_task_generator = None
        
        # Clear any remaining tasks
        self.clear_task_queue()
        
        # Reset current task
        self.current_task = None
        
        # Turn off laser if it's on (only if not already in the process of aborting)
        if self.state != "IDLE":
            try:
                self.dispatch("turn_off_laser")
            except Exception as e:
                print(f"Error turning off laser during abort: {e}")
        
        # Reset abort flag after cleanup
        self.abort_flag = False
        
        print("Abort cleanup completed. System is idle.")

    def handle_update_sampling_points_to_protocol_module(self, sampling_points_relative_distance_to_camera_center, coordinates_order):
        print('Handle_update_sampling_points_to_protocol_module is running')
        self.sampling_points_relative_distance_to_camera_center = sampling_points_relative_distance_to_camera_center
        self.coordinates_order = coordinates_order

    def create_widgets(self):
        protocol_frame = ttk.Labelframe(self, text="Aquisition", padding=5)
        protocol_frame.grid(row=0, column=0, columnspan=3, sticky="nsew",padx=5, pady=5)

        # file name entry for saving the data
        self.save_data = ttk.Frame(protocol_frame, padding=5)
        self.save_data.grid(row=0, column=0,columnspan=3, padx=5, pady=5, sticky="nsew")

        # file path entry for saving the data
        default_path = os.path.join(os.path.expanduser("~"), "Documents", "RamanData")
        self.folder_path = StringVar()
        self.folder_path.set(default_path)
        self.folder_path_label = ttk.Label(self.save_data, text="Folder Path:").grid(row=0, column=0, padx=5, pady=5)
        self.folder_path_entry = ttk.Entry(self.save_data, textvariable=self.folder_path, width = 62).grid(row=0, column=1, padx=5, pady=5, sticky="nsew")     
        self.folder_path_button = ttk.Button(self.save_data, text="Browse", command=self.browse_folder, bootstyle ='info').grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        self.file_base_name = StringVar()
        self.file_base_name.set("patient_staging")
        self.file_base_name_label = ttk.Label(self.save_data, text="File Name:").grid(row=1, column=0, padx=5, pady=5)
        self.file_base_name_entry = ttk.Entry(self.save_data, textvariable=self.file_base_name, width = 62).grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        # set protocol button
        self.select_sample_drop_button = ttk.Button(protocol_frame, text="Set Protocol", command=self.create_protocol_window, bootstyle = 'info')
        self.select_sample_drop_button.grid(row=1, column=0,  padx=5, pady=5, sticky="nsew")

        # button to start the protocol
        self.start_protocol_button = ttk.Button(protocol_frame, text="Start Aquisition", command=self.start_aquisition, bootstyle = 'info', state=DISABLED)
        self.start_protocol_button.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        # button to abort the acquisition
        self.abort_aquisition_button = ttk.Button(protocol_frame, text="Abort Aquisition", command=self.handle_abort_aquisition, bootstyle='danger', state=DISABLED)
        self.abort_aquisition_button.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")

    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        self.folder_path.set(folder_path)

    def handle_update_setup_status(self, status):
        if status == "focus_range_is_setup":
            self.focus_range_is_setup = True
        elif status == "stage_reference_is_setup":
            self.stage_reference_is_setup = True

        if self.focus_range_is_setup and self.stage_reference_is_setup:
            self.start_protocol_button.configure(state=NORMAL)
            self.abort_aquisition_button.configure(state=NORMAL)
        return None

    def create_protocol_window(self):
        # Check if the window is already open
        if hasattr(self, 'set_protocol_window') and self.set_protocol_window.winfo_exists():
            print("The sample selection window is already open.")
            self.set_protocol_window.focus_force()
            return

        # For sample drop selection
        self.set_protocol_window = ttk.Toplevel(self.parent)
        self.set_protocol_window.title("Set Protocol")
        self.set_protocol_window.resizable(False, False)
        self.set_protocol_window.focus_force()
        self.set_protocol = ttk.Frame(self.set_protocol_window)
        self.set_protocol.grid(row=0, column=0, padx=10, pady=5)

        # canvas for selecting the sample drops
        self.select_sample_drop_frame = ttk.Labelframe(self.set_protocol, text="Select Sample Drops", padding=5)
        self.select_sample_drop_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=5, sticky="nsew")
        self.protocol_select_sample_canvas = ttk.Canvas(self.select_sample_drop_frame, width=self.protocol_canvas_width, height=self.protocol_canvas_height)
        self.protocol_select_sample_canvas.grid(row=0, column=0,columnspan=2, padx=40, pady=5, sticky='nsew')
        # Ensure selected lists exist before grid creation so the grid can reflect previous choices
        if not hasattr(self, 'selected_circle_positions'):
            self.selected_circle_positions = []
        if not hasattr(self, 'selected_circles'):
            self.selected_circles = set()
        self.protocol_select_sample_create_circle_grid()
        
        select_all_button = ttk.Button(self.select_sample_drop_frame, text="Select All", command=self.protocol_select_all_samples, bootstyle='info_outline')
        select_all_button.grid(row=1, column=0, pady=10, padx=10, sticky='ew')
        
        cancel_all_button = ttk.Button(self.select_sample_drop_frame, text="Cancel All", command=self.protocol_cancel_all_samples, bootstyle='danger-outline')
        cancel_all_button.grid(row=1, column=1, pady=10, padx=10, sticky='ew')
        
        # Configure grid to center align the canvas and buttons
        self.select_sample_drop_frame.grid_columnconfigure(0, weight=1)
        self.select_sample_drop_frame.grid_columnconfigure(1, weight=1)
        self.select_sample_drop_frame.grid_rowconfigure(0, weight=1)
        self.select_sample_drop_frame.grid_rowconfigure(1, weight=1)

        # entry for the number of reps - reuse existing StringVar if present so value persists between opens
        if not hasattr(self, 'number_of_spectra_var') or not isinstance(getattr(self, 'number_of_spectra_var'), StringVar):
            # create and initialize from the plain attribute if available
            try:
                initial = int(self.number_of_spectra)
            except Exception:
                initial = 3
            self.number_of_spectra_var = StringVar(value=str(initial))
        self.number_of_samples_label = ttk.Label(self.set_protocol, text="Number of repetitions per point per sample drop:").grid(row=1, column=0, padx=5, pady=5)
        self.number_of_samples_entry = ttk.Entry(self.set_protocol, textvariable=self.number_of_spectra_var, width = 5).grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        # entry for the number of times to repeat each whole sample (new)
        if not hasattr(self, 'number_of_repeats_var') or not isinstance(getattr(self, 'number_of_repeats_var'), StringVar):
            try:
                initial_repeats = int(getattr(self, 'number_of_repeats', 1))
            except Exception:
                initial_repeats = 1
            self.number_of_repeats_var = StringVar(value=str(initial_repeats))
        self.number_of_repeats_label = ttk.Label(self.set_protocol, text="Number of times to repeat each sample:").grid(row=2, column=0, padx=5, pady=5)
        self.number_of_repeats_entry = ttk.Entry(self.set_protocol, textvariable=self.number_of_repeats_var, width = 5).grid(row=2, column=1, padx=5, pady=5, sticky="nsew")

        # check box for autofocusing in widefield, objective, and wasatch - reuse StringVars if present
        if not hasattr(self, 'autofocus_widefield_var') or not isinstance(getattr(self, 'autofocus_widefield_var'), StringVar):
            self.autofocus_widefield_var = StringVar(value="True")
        if not hasattr(self, 'wasatch_autofocus_with_prusa_var') or not isinstance(getattr(self, 'wasatch_autofocus_with_prusa_var'), StringVar):
            self.wasatch_autofocus_with_prusa_var = StringVar(value="True")
        if not hasattr(self, 'wasatch_autofocus_with_nanodrive_var') or not isinstance(getattr(self, 'wasatch_autofocus_with_nanodrive_var'), StringVar):
            self.wasatch_autofocus_with_nanodrive_var = StringVar(value="True")
        if not hasattr(self, 'generate_sample_points_upfront_var') or not isinstance(getattr(self, 'generate_sample_points_upfront_var'), StringVar):
            self.generate_sample_points_upfront_var = StringVar(value="False")

         # Frame to hold the autofocus checkbuttons

        self.autofocus_checkbuttons_frame = ttk.Frame(self.set_protocol)
        self.autofocus_checkbuttons_frame.grid(row=3, column=0, columnspan=3, pady=10, padx=10, sticky='ew')
        self.autofocus_widefield = ttk.Checkbutton(self.autofocus_checkbuttons_frame, text="Autofocus Widefield", variable=self.autofocus_widefield_var, onvalue="True", offvalue="False", bootstyle='info')
        self.autofocus_widefield.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.autofocus_objective = ttk.Checkbutton(self.autofocus_checkbuttons_frame, text="Autofocus Objective", variable=self.wasatch_autofocus_with_prusa_var, onvalue="True", offvalue="False", bootstyle='info')
        self.autofocus_objective.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.autofocus_wasatch = ttk.Checkbutton(self.autofocus_checkbuttons_frame, text="Autofocus Wasatch", variable=self.wasatch_autofocus_with_nanodrive_var, onvalue="True", offvalue="False", bootstyle='info')
        self.autofocus_wasatch.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        self.generate_sample_points_upfront = ttk.Checkbutton(self.autofocus_checkbuttons_frame, text="Generate Sample Points Upfront", variable=self.generate_sample_points_upfront_var, onvalue="True", offvalue="False", bootstyle='info')
        self.generate_sample_points_upfront.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

        # button to confirm the selection
        self.ok_button = ttk.Button(self.set_protocol, text="OK", command=self.confirm_sample_selection, bootstyle='info')
        self.ok_button.grid(row=4, column=0, columnspan=3, pady=10, padx=10, sticky='ew')

        self.set_protocol_window.update_idletasks()
        self.set_protocol_window.geometry(f"{self.set_protocol_window.winfo_reqwidth()}x{self.set_protocol_window.winfo_reqheight()}")

    def protocol_select_sample_create_circle_grid(self):
        skip_positions = [(0, 0), (3, 0), (0, 4), (3, 4)]  # Positions to skip

        # Ensure `selected_circles` and `selected_circle_positions` exist and are persistent
        if not hasattr(self, 'selected_circles'):
            self.selected_circles = set()  # Track selected circles by their IDs
        if not hasattr(self, 'selected_circle_positions'):
            self.selected_circle_positions = []  # Track (row, column) positions of selected circles

        self.circles = []  # Store references to all circles

        for col_index, num_circles in enumerate(self.protocol_canvas_grid_pattern):
            vertical_offset = (max(self.protocol_canvas_grid_pattern) - num_circles) * self.protocol_canvas_circle_spacing // 2
            for row_index in range(num_circles):
                true_row_index = row_index + vertical_offset // self.protocol_canvas_circle_spacing

                # Skip the specified corner positions
                if (true_row_index, col_index) in skip_positions:
                    continue

                x = col_index * self.protocol_canvas_circle_spacing + self.protocol_canvas_circle_spacing // 2
                y = vertical_offset + row_index * self.protocol_canvas_circle_spacing + self.protocol_canvas_circle_spacing // 2

                # Determine the initial color of the circle based on its selection status
                if (true_row_index, col_index) in self.selected_circle_positions:
                    fill_color = 'skyblue'  # Selected
                else:
                    fill_color = 'grey'  # Not selected

                circle = self.protocol_select_sample_canvas.create_oval(
                    x - self.protocol_canvas_circle_radius, y - self.protocol_canvas_circle_radius,
                    x + self.protocol_canvas_circle_radius, y + self.protocol_canvas_circle_radius,
                    fill=fill_color, outline='black', tags=f"circle_{true_row_index}_{col_index}"
                )
                self.circles.append((circle, x, y, true_row_index, col_index))

                # Bind click event for toggling selection
                self.protocol_select_sample_canvas.tag_bind(circle, '<Button-1>',
                            lambda event, c=circle, r=true_row_index, col=col_index: self.protocol_select_sample_toggle_selection(c, r, col))

        # Bind mouse events for dragging selection
        self.protocol_select_sample_canvas.bind('<Button-1>', self.protocol_select_sample_on_drag_start)
        self.protocol_select_sample_canvas.bind('<B1-Motion>', self.protocol_select_sample_on_drag_motion)
        self.protocol_select_sample_canvas.bind('<ButtonRelease-1>', self.protocol_select_sample_on_drag_end)

    def protocol_select_sample_toggle_selection(self, circle, row, col):
        """
        Toggle selection state of the clicked circle.
        """
        if circle in self.selected_circles:  # Deselect if already selected
            self.selected_circles.remove(circle)
            self.selected_circle_positions.remove((row, col))  # Remove position from the list
            self.protocol_select_sample_canvas.itemconfig(circle, fill='grey', outline='black')
        else:  # Select if not already selected
            self.selected_circles.add(circle)
            self.selected_circle_positions.append((row, col))  # Add position to the list
            self.protocol_select_sample_canvas.itemconfig(circle, fill='skyblue', outline='black')

        # Print updated selected circle positions
        print("Currently selected circles (row, col):", self.selected_circle_positions)

    def protocol_select_sample_on_drag_start(self, event):
        # Store the starting position of the drag
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        # Initialize a rectangle for visual feedback during the drag
        self.drag_rectangle = self.protocol_select_sample_canvas.create_rectangle(
            self.drag_start_x, self.drag_start_y, self.drag_start_x, self.drag_start_y, outline='skyblue', tag='drag_rectangle'
        )

    def protocol_select_sample_on_drag_motion(self, event):
        # Update the rectangle to match the current drag area
        self.protocol_select_sample_canvas.coords(
            self.drag_rectangle, self.drag_start_x, self.drag_start_y, event.x, event.y
        )
        # Compute the selection rectangle bounds
        x1, y1 = min(self.drag_start_x, event.x), min(self.drag_start_y, event.y)
        x2, y2 = max(self.drag_start_x, event.x), max(self.drag_start_y, event.y)

        # Check which circles are within the rectangle (highlight only for visual feedback)
        for circle, cx, cy, row, col in self.circles:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                if circle not in self.selected_circles:  # Highlight if not already selected
                    self.protocol_select_sample_canvas.itemconfig(circle, fill='grey', outline='skyblue')
            elif circle not in self.selected_circles:  # Revert unselected circles to grey
                self.protocol_select_sample_canvas.itemconfig(circle, fill='grey', outline='black')

    def protocol_select_sample_on_drag_end(self, event):
        # Remove the drag rectangle after the drag operation ends
        self.protocol_select_sample_canvas.delete(self.drag_rectangle)

        # Compute the selection rectangle bounds
        x1, y1 = min(self.drag_start_x, event.x), min(self.drag_start_y, event.y)
        x2, y2 = max(self.drag_start_x, event.x), max(self.drag_start_y, event.y)

        # Toggle selection for circles within the rectangle
        for circle, cx, cy, row, col in self.circles:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                if circle in self.selected_circles:  # If already selected, deselect
                    self.selected_circles.remove(circle)
                    self.selected_circle_positions.remove((row, col))  # Remove position from the list
                    self.protocol_select_sample_canvas.itemconfig(circle, fill='grey', outline='black')
                else:  # Otherwise, select it
                    self.selected_circles.add(circle)
                    self.selected_circle_positions.append((row, col))  # Add position to the list
                    self.protocol_select_sample_canvas.itemconfig(circle, fill='skyblue', outline='black')

        # Print updated selected circle positions
        print("Currently selected circles (row, col):", self.selected_circle_positions)

    def protocol_select_all_samples(self):
        for circle, cx, cy, row, col in self.circles:
            if circle not in self.selected_circles:
                self.selected_circles.add(circle)
                self.selected_circle_positions.append((row, col))
                self.protocol_select_sample_canvas.itemconfig(circle, fill='skyblue', outline='black')
        print("All circles selected:")
        self.update_circle_colors()

    def protocol_cancel_all_samples(self):
        for circle, cx, cy, row, col in self.circles:
            if circle in self.selected_circles:
                self.selected_circles.remove(circle)
                self.selected_circle_positions.remove((row, col))
                self.protocol_select_sample_canvas.itemconfig(circle, fill='grey', outline='black')
        print("All circles deselected.")
        self.update_circle_colors()

    def update_circle_colors(self):
        for circle, cx, cy, row, col in self.circles:
            if (row, col) in self.selected_circle_positions:
                self.protocol_select_sample_canvas.itemconfig(circle, fill='skyblue', outline='black')
            else:
                self.protocol_select_sample_canvas.itemconfig(circle, fill='grey', outline='black')

    def confirm_sample_selection(self):
        print("Saving selected circle positions:", self.selected_circle_positions)
        self.set_protocol_window.destroy()


if __name__ == '__main__':
    root = ttk.Window()
    root.style.theme_use('noodlepy')
    protocol_frame = ProtocolModule(root)
    protocol_frame.grid(row=0, column=0, sticky="nsew")
    root.mainloop()