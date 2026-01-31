import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import StringVar
import tkinter as tk
from noodlepy.gui.liveviewmodule import LiveViewModule
from noodlepy.gui.stagecontrolmodule import StageControlModule
from noodlepy.gui.protocolmodule import ProtocolModule
from noodlepy.gui.wasatch_autofocus_module import WasatchAutofocusModule
from noodlepy.gui.droppositionmodule import DropPositionModule
from noodlepy.gui.settingsmodule import AppSettings, SettingsModule

class AcquisitionGUI(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.settings = AppSettings()
        self.create_widgets()

    def create_widgets(self):

        first_column = ttk.Frame(self)
        first_column.grid(row=0, column=0, sticky="nsew")
  
        middle_column = ttk.Frame(self)
        middle_column.grid(row=0, column=1, rowspan=4, sticky="nsew")
        
        last_column = ttk.Frame(self)
        last_column.grid(row=0, column=2, sticky="nsew")

        # Make row 0 (Wasatch) take up extra vertical space
        last_column.grid_rowconfigure(0, weight=1)
        last_column.grid_rowconfigure(1, weight=0)   # settings row stays minimal
        last_column.grid_columnconfigure(0, weight=1)

        self.live_view_frame = LiveViewModule(first_column, self.settings.liveview)
        self.live_view_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")

        self.stage_control_frame = StageControlModule(middle_column, self.settings.stage)
        self.stage_control_frame.grid(row=0, column=0, rowspan=3, columnspan=1, sticky="nsew")
        self.live_view_frame.add_subscriber('switch_view', self.stage_control_frame, self.stage_control_frame.handle_switch_view)
        self.live_view_frame.add_subscriber('test_all_sampling_points', self.stage_control_frame, callback = self.stage_control_frame.handle_test_sampling_points)
        self.live_view_frame.add_subscriber('update_captured_frame_center', self.stage_control_frame, callback = self.stage_control_frame.handle_updated_captured_frame_center)
        self.live_view_frame.add_subscriber('focus_widefield_camera', self.stage_control_frame, callback = self.stage_control_frame.handle_focus_widefield_camera)
        self.live_view_frame.add_subscriber('update_focus_score', self.stage_control_frame, callback = self.stage_control_frame.handle_update_focus_score)
        self.stage_control_frame.add_subscriber('calculate_focus_score', self.live_view_frame, callback = self.live_view_frame.handle_calculate_focus_score)
        self.stage_control_frame.add_subscriber('activate_camera_autofocus_button', self.live_view_frame, callback = self.live_view_frame.handle_activate_camera_autofocus_button)
        self.stage_control_frame.add_subscriber('activate_switch_view_button', self.live_view_frame, callback = self.live_view_frame.handle_activate_switch_view_button)
        self.wasatch_autofocus_frame = WasatchAutofocusModule(last_column)
        self.wasatch_autofocus_frame.grid(row=0, column=0, sticky="nsew")
        self.wasatch_autofocus_frame.add_subscriber('update_nanodrive_position', self.stage_control_frame, callback = self.stage_control_frame.handle_update_nanodrive_position)
        self.stage_control_frame.add_subscriber('move_nanodrive_by', self.wasatch_autofocus_frame, callback = self.wasatch_autofocus_frame.handle_move_nanodrive_by_request)
        self.stage_control_frame.add_subscriber('move_nanodrive_to', self.wasatch_autofocus_frame, callback = self.wasatch_autofocus_frame.handle_move_nanodrive_to_request)
        self.stage_control_frame.add_subscriber('get_nanodrive_position', self.wasatch_autofocus_frame, callback = self.wasatch_autofocus_frame.handle_get_nanodrive_position)
        self.stage_control_frame.add_subscriber('get_nanodrive_min_max', self.wasatch_autofocus_frame, callback = self.wasatch_autofocus_frame.handle_get_nanodrive_min_max)
        self.stage_control_frame.add_subscriber('update_prusa_focus_range_for_wasatch', self.wasatch_autofocus_frame, callback = self.wasatch_autofocus_frame.handle_update_prusa_focus_range_for_wasatch)
        self.wasatch_autofocus_frame.add_subscriber('move_prusa_to', self.wasatch_autofocus_frame, callback = self.stage_control_frame.handle_move_prusa_to)
        self.acquisition_protocol_frame = ProtocolModule(middle_column)
        self.acquisition_protocol_frame.grid(row=3, column=0, rowspan=1, columnspan=1, sticky="nsew")
        self.stage_control_frame.add_subscriber('activate_protocol_module_state', self.acquisition_protocol_frame, callback = self.acquisition_protocol_frame.handle_update_setup_status)
        self.stage_control_frame.add_subscriber('abort_aquisition', self.acquisition_protocol_frame, callback = self.acquisition_protocol_frame.handle_abort_aquisition)
        self.acquisition_protocol_frame.add_subscriber('move_stage_to_target_sample_drop_during_aquisition', self.stage_control_frame, callback = self.stage_control_frame.handle_move_stage_to_target_sample_drop_during_aquisition)
        self.acquisition_protocol_frame.add_subscriber('focus_widefield_camera', self.stage_control_frame, callback = self.stage_control_frame.handle_focus_widefield_camera_during_acquisition)
        self.acquisition_protocol_frame.add_subscriber('check_current_camera_view', self.live_view_frame, callback = self.live_view_frame.handle_check_current_camera_view)
        self.acquisition_protocol_frame.add_subscriber('capture_current_image_and_detect_sample_drop_and_create_sampling_points', self.live_view_frame, callback = self.live_view_frame.handling_create_sampling_points_during_aquisition)
        self.acquisition_protocol_frame.add_subscriber('call_switch_view_button_in_live_camera_module', self.live_view_frame, self.live_view_frame.handle_switch_view_during_aquisition)
        self.acquisition_protocol_frame.add_subscriber('move_to_a_single_sampling_point', self.stage_control_frame, callback = self.stage_control_frame.handle_move_to_a_single_sampling_point_during_acquisition)
        self.acquisition_protocol_frame.add_subscriber('focus_wasatch_with_prusa', self.stage_control_frame, callback = self.wasatch_autofocus_frame.handle_autofocus_wasatch_with_prusa_during_aquisition)
        self.acquisition_protocol_frame.add_subscriber('focus_wasatch_with_nanodrive', self.wasatch_autofocus_frame, callback = self.wasatch_autofocus_frame.handle_autofocus_wasatch_with_nanodrive_during_aquisition)
        self.acquisition_protocol_frame.add_subscriber('measure_spectra_and_save_to_specific_folder', self.wasatch_autofocus_frame, callback = self.wasatch_autofocus_frame.handle_measure_spectra_and_save_to_specific_folder)
        # self.acquisition_protocol_frame.add_subscriber('update_capture_save_folder', self.wasatch_autofocus_frame, callback = self.wasatch_autofocus_frame.handle_update_save_folder)
        self.acquisition_protocol_frame.add_subscriber('turn_on_laser', self.wasatch_autofocus_frame, callback = self.wasatch_autofocus_frame.handle_turn_laser_on)
        self.acquisition_protocol_frame.add_subscriber('turn_off_laser', self.wasatch_autofocus_frame, callback = self.wasatch_autofocus_frame.handle_turn_laser_off)
        self.acquisition_protocol_frame.add_subscriber('reposition_stage_and_nanodrive_in_objective_view', self.stage_control_frame, callback = self.stage_control_frame.handle_reposition_stage_and_nanodrive_in_objective_view_during_acquisition)
        self.stage_control_frame.add_subscriber('task_completed', self.acquisition_protocol_frame, callback = self.acquisition_protocol_frame.handle_task_completed)
        self.live_view_frame.add_subscriber('task_completed', self.acquisition_protocol_frame, callback = self.acquisition_protocol_frame.handle_task_completed)
        self.wasatch_autofocus_frame.add_subscriber('task_completed', self.acquisition_protocol_frame, callback = self.acquisition_protocol_frame.handle_task_completed)
        self.live_view_frame.add_subscriber('abort_aquisition', self.acquisition_protocol_frame, callback = self.acquisition_protocol_frame.handle_abort_aquisition)
        self.live_view_frame.add_subscriber('update_sampling_points_to_protocol_module', self.acquisition_protocol_frame, callback = self.acquisition_protocol_frame.handle_update_sampling_points_to_protocol_module)

        self.settings_frame = SettingsModule(last_column, app_settings=self.settings)
        self.settings_frame.grid(row=1, column=0, sticky="se", padx=5, pady=5)
        self.stage_control_frame.add_subscriber('update_home_position', self.settings_frame, self.settings_frame.handle_update_home_position)
        self.settings_frame.add_subscriber('update_settings', self.stage_control_frame, self.stage_control_frame.handle_update_settings)
        self.settings_frame.add_subscriber('update_settings', self.live_view_frame, self.live_view_frame.handle_update_settings)

if __name__ == '__main__':
    root = ttk.Window()
    root.style.theme_use('noodlepy')
    app = AcquisitionGUI(root)
    app.grid(row=0, column=0, sticky="nsew")
    root.mainloop()
