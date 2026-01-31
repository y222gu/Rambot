import numpy as np
import time
from tkinter import ttk
from tkinter import DISABLED
from noodlepy.gui.publisher_subscriber import Publisher

class DropPositionModule(Publisher, ttk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        Publisher.__init__(self, ['test_drop_position'])
        self.name = 'DropPositionModule_Publisher'

        self.parent = parent
        self.x_interval_label = ttk.Label(parent, text="X interval:")
        self.x_interval_label.grid(row=0, column=0, padx=5, pady=5)

        self.y_interval_label = ttk.Label(parent, text="Y interval:")
        self.y_interval_label.grid(row=1, column=0, padx=5, pady=5)
        self.x_number_label = ttk.Label(parent, text="# in X:")
        self.x_number_label.grid(row=0, column=2, padx=5, pady=5)
        self.y_number_label = ttk.Label(parent, text="# in Y:")
        self.y_number_label.grid(row=1, column=2, padx=5, pady=5)
        self.x_interval_entry = ttk.Entry(parent, width=5)
        self.x_interval_entry.grid(row=0, column=1, padx=5, pady=5)
        self.x_interval_entry.insert(0, "4.5")
        self.y_interval_entry = ttk.Entry(parent, width=5)
        self.y_interval_entry.grid(row=1, column=1, padx=5, pady=5)
        self.y_interval_entry.insert(0, "4.5")
        self.x_number_entry = ttk.Entry(parent, width=5)
        self.x_number_entry.grid(row=0, column=3, padx=5, pady=5)
        self.x_number_entry.insert(0, "5")
        self.y_number_entry = ttk.Entry(parent, width=5)
        self.y_number_entry.grid(row=1, column=3, padx=5, pady=5)
        self.y_number_entry.insert(0, "5")
        self.test_sample_spot_button = ttk.Button(parent, text="Test Sample Grid", command=self.test_sample_grid, bootstyle="info", state=DISABLED)
        self.test_sample_spot_button.grid(row=0, column=4, rowspan=2, sticky='nsew', padx=5, pady=5)

    def test_sample_grid(self):
        self.parent.run_in_thread(self.parent._test_sample_grid)

    def _test_sample_grid(self):

        if not self.x_interval_entry.get() or not self.y_interval_entry.get() or not self.x_number_entry.get() or not self.y_number_entry.get():
            print("Please enter all the parameters")
            return
        
        x_interval = self.x_interval_entry.get()
        y_interval = self.y_interval_entry.get()
        x_number = self.x_number_entry.get()
        y_number = self.y_number_entry.get()

        if self.position_first_smaple is None:
            print("Please register the first sample first")
            return

        if x_interval and y_interval and x_number and y_number:
            first_x = float(self.position_first_smaple[0])
            first_y = float(self.position_first_smaple[1])
            first_z = float(self.position_first_smaple[2])

            x = np.linspace(first_x, first_x + float(x_interval) * (int(x_number) - 1), int(x_number))
            y = np.linspace(first_y - float(y_interval) * (int(y_number) - 1), first_y, int(y_number))
            xx, yy = np.meshgrid(x, y)
            # make y descending order
            yy = np.flip(yy, axis=0)
            xx = xx.flatten(order='F')
            yy = yy.flatten(order='F')
            # make 
            zz = np.ones(xx.size) * first_z

            print(f"A grid containing {xx.size} points will be tested")
        else:
            print("Please enter all the parameters")
        for i in range(xx.size):
            self.go_to_xyz(x=xx[i], y=yy[i], z=zz[i])
            print(f"Moving to {xx[i]}, {yy[i]}, {zz[i]}")
            time.sleep(1)
        print("Test completed")
