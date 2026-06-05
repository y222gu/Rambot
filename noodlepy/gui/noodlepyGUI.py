from pathlib import Path
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, Text, END, RIGHT, LEFT, Y, BOTH, VERTICAL
from ttkbootstrap import Style
from ttkbootstrap.constants import *
import pandas as pd
from noodlepy.gui.analysistab import AnalysisGUI
from noodlepy.gui.acquisitiontab import AcquisitionGUI


class NoodlepyGUI(ttk.Window):

    def __init__(self):
        super().__init__(themename="superhero")

        mainframe = ttk.Frame(self)
        mainframe.grid(row=0, column=0, sticky=NSEW)

        # Create vertical Notebook for tabs
        notebook = ttk.Notebook(mainframe)
        notebook.grid(row=0, column=0, sticky=NSEW)

        acquisition_tab = ttk.Frame(notebook)
        notebook.add(acquisition_tab, text='Aquistion')
        analysis_tab = ttk.Frame(notebook)
        notebook.add(analysis_tab, text='Analysis')

        # Configure the tabs to expand properly
        for tab in (acquisition_tab, analysis_tab):
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)

        # Create the AnalysisTab
        analysis_gui = AnalysisGUI(analysis_tab)
        analysis_gui.grid(row=0, column=0, sticky=NSEW)

        acquisiton_gui = AcquisitionGUI(acquisition_tab)
        acquisiton_gui.grid(row=0, column=0, sticky=NSEW)

if __name__ == '__main__':

    app = NoodlepyGUI()
    app.mainloop()
