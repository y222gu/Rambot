import os
from ctypes import *
import time

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
        self.max_position_um = 90
        self.min_position_um = 0
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
        # round to integer
        position = round(position)
        return position

    def move_to(self, z_pos_um):
        if type(z_pos_um) != float:
            z_pos_um = float(z_pos_um)
        
        z_pos_um = max(self.min_position_um, min(z_pos_um, self.max_position_um)) # Ensure new_position stays within bounds [0, 100]um
        pos = c_double(z_pos_um)
        error = self.mcldll.MCL_SingleWriteN(pos, self.axis, self.handle)

        time.sleep(0.025)  # Wait for nanopositioner to settle
        if error != 0:
            print("Nanodrive move_to_position Error =", error)
            print(f"Nanodrive attempting to move to position: {z_pos_um} on axis: {self.axis.value} with handle: {self.handle}")
        else:
            self.current_position_um = self.get_current_position()
            # print(f"Nanodrive moved to position: {self.current_position_um}.")
            # print(f"Requested position: {z_pos_um}")

    def move_by(self, delta_z_um: float):
        # Get the current position
        current_position_um = self.mcldll.MCL_SingleReadN(self.axis, self.handle)
     
        # Calculate the new position
        new_position_um = current_position_um + delta_z_um
        new_position_um = max(self.min_position_um, min(new_position_um, self.max_position_um)) # Ensure new_position stays within bounds [0, 100]
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
