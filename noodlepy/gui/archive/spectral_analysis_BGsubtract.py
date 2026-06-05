import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pybaselines import Baseline
from scipy.optimize import curve_fit

# Changeable Parameters
EXCITATION_WAVELENGTH = 785  # in nm
CUT_FIRST_POINTS = 10  # Number of points to cut from the beginning of each spectrum
BACKGROUND_FILE_PATH = '/mnt/c/Users/Yifei/Documents/NoodlePy/noodlepy/data/20240319_P88-EVs-desalted1x-10E12-oneweek_60x_2mm_785nm_42mW_853cw_quartz_grating3_01.txt'
SPECTRA_DIRECTORY = '/mnt/c/Users/Yifei/Documents/NoodlePy/noodlepy/data/356_control_subset'
OUTPUT_DIRECTORY = '/mnt/c/Users/Yifei/Documents/NoodlePy/noodlepy/data/356_control_subset/corrected_spectra'
OUTPUT_FILE_PATH = '/mnt/c/Users/Yifei/Documents/NoodlePy/noodlepy/data/combined_corrected_spectra.txt'
LAM = 1E3  # Parameter for airPLS
DIFF_ORDER = 1  # Parameter for airPLS
MAX_ITER = 15  # Parameter for airPLS
TOL = 1e-3  # Parameter for airPLS
WINDOW_LENGTH = 9  # Parameter for Savitzky-Golay filter
POLYORDER = 2  # Parameter for Savitzky-Golay filter

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

# Function for improved baseline correction using different methods
def improved_baseline_correction(x_data, y_data):
    baseline_fitter = Baseline(x_data)
    
    # Experiment with different methods and parameters
    methods = ['modpoly', 'imodpoly', 'asls']
    results = {}
    for method in methods:
        if method in ['modpoly', 'imodpoly']:
            baseline, _ = getattr(baseline_fitter, method)(y_data, poly_order=2)
        else:
            baseline, _ = getattr(baseline_fitter, method)(y_data, lam=1e6)
        results[method] = y_data - baseline
    
    # Select the best method (this might require visual inspection)
    # For demonstration, we return the result from 'asls'
    return results['asls']

# Function for Savitzky-Golay smoothing
def apply_savgol_filter(y_data, window_length=WINDOW_LENGTH, polyorder=POLYORDER):
    return savgol_filter(y_data, window_length, polyorder)

# Load the background data
background_data = pd.read_csv(BACKGROUND_FILE_PATH, sep=",", header=None)

# Function to fit the background to the active spectra
def background_fit(x, scale):
    return scale * np.interp(x, background_data[0], background_data[1])

# List all spectra files
spectra_files = [f for f in os.listdir(SPECTRA_DIRECTORY) if f.endswith('.txt')]

corrected_spectra_all = []

fig, axes = plt.subplots(4, 1, figsize=(15, 20))
plt.show()

# Process each spectra file
for spectra_file in spectra_files:
    spectra_file_path = os.path.join(SPECTRA_DIRECTORY, spectra_file)
    spectra_data = pd.read_csv(spectra_file_path, sep=",", header=None)

    # Split the spectra into separate components based on the wavelength restarting
    split_indices = [0]
    for i in range(1, len(spectra_data)):
        if spectra_data.iloc[i, 0] < spectra_data.iloc[i - 1, 0]:
            split_indices.append(i)
    split_indices.append(len(spectra_data))

    spectra_parts = [spectra_data.iloc[split_indices[i]:split_indices[i+1]] for i in range(len(split_indices) - 1)]

    # Cut the first 300 points from each spectrum
    spectra_parts = [part.iloc[CUT_FIRST_POINTS:] for part in spectra_parts]

    # Combine the parts and take the median value across columns
    combined_df = pd.concat([part.set_index(part[0]) for part in spectra_parts], axis=1, ignore_index=True)
    combined_df = combined_df.replace([np.inf, -np.inf], np.nan).dropna() # Handle NaN and infinite values
    combined_df_median = combined_df.median(axis=1) # Calculate the median across columns
    combined_spectra = pd.DataFrame({0: combined_df.index, 1: combined_df_median.values}).reset_index(drop=True) # Create a DataFrame for the median spectrum

    # Plot raw data
    for part in spectra_parts:
        axes[0].plot(part[0], part[1], label=spectra_file)
    
    # Plot data after taking median
    axes[1].plot(combined_spectra[0], combined_spectra[1], label=spectra_file)

    # Apply improved baseline correction and smoothing
    combined_spectra[1] = improved_baseline_correction(combined_spectra[0], combined_spectra[1])
    combined_spectra[1] = apply_savgol_filter(combined_spectra[1])

    # Plot data after baseline correction and smoothing
    axes[2].plot(combined_spectra[0], combined_spectra[1], label=spectra_file)

    # Fit the background to the median spectra
    x_spectra = combined_spectra[0]
    y_spectra = combined_spectra[1]
    popt, _ = curve_fit(lambda x, scale: background_fit(x, scale), x_spectra, y_spectra, p0=[1.0])
    corrected_spectra = y_spectra - background_fit(x_spectra, *popt)

    # Plot data after background subtraction
    axes[3].plot(x_spectra, corrected_spectra, label=spectra_file)

    # Save each corrected spectrum to an individual file
    output_file_path = os.path.join(OUTPUT_DIRECTORY, f"corrected_{os.path.splitext(spectra_file)[0]}.txt")
    pd.DataFrame({0: x_spectra, 1: corrected_spectra}).to_csv(output_file_path, sep='\t', index=False, header=False)

    # Append corrected spectra to list
    corrected_spectra_all.append(pd.DataFrame({0: x_spectra, 1: corrected_spectra}))

# Combine all corrected spectra into a single dataframe
corrected_spectra_all_df = pd.concat(corrected_spectra_all, ignore_index=True)

# Save the combined corrected spectra to a single txt file
corrected_spectra_all_df.to_csv(OUTPUT_FILE_PATH, sep='\t', index=False, header=False)

print(f'Corrected spectra saved to {OUTPUT_FILE_PATH}')

# Set labels and titles for the subplots
axes[0].set_title('Raw Data')
axes[1].set_title('After Taking Median')
axes[2].set_title('After Baseline Correction and Smoothing')
axes[3].set_title('After Background Subtraction')

for ax in axes:
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.grid(True)

plt.tight_layout()
plt.show()

plt.close()
