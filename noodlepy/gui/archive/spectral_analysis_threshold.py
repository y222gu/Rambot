import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from pybaselines import Baseline

# Changeable Parameters
EXCITATION_WAVELENGTH = 785  # in nm
CUT_FIRST_POINTS = 300  # Number of points to cut from the beginning of each spectrum
PERCENT_SPECTRA_TO_KEEP = 0.4  # Percentage of top spectra to keep
SPECTRA_DIRECTORY = '/mnt/c/Users/Yifei/Documents/NoodlePy/noodlepy/data/all 5 sec desalted EVs'
OUTPUT_DIRECTORY = '/mnt/c/Users/Yifei/Documents/NoodlePy/noodlepy/data/Cancer_top40%_filtered_spectra_output'

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

# Function for improved baseline correction using different methods
def improved_baseline_correction(x_data, y_data):
    if len(x_data) == 0 or len(y_data) == 0:
        return y_data
    baseline_fitter = Baseline(x_data)
    # Experiment with different methods and parameters
    baseline, _ = baseline_fitter.asls(y_data, lam=1e6)
    return y_data - baseline

# Function for Savitzky-Golay smoothing
def apply_savgol_filter(y_data, window_length=9, polyorder=2):
    if len(y_data) < window_length:
        window_length = len(y_data) - 1 if len(y_data) % 2 == 0 else len(y_data)
    return savgol_filter(y_data, window_length, polyorder)

# Function to calculate the area under the curve
def calculate_area(y_data):
    return np.trapz(y_data)

# Function to process each folder
def process_folder(folder_path, output_folder):
    spectra_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    all_median_spectra = []

    for spectra_file in spectra_files:
        spectra_file_path = os.path.join(folder_path, spectra_file)
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
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan).dropna()
        combined_df_median = combined_df.median(axis=1)
        combined_spectra = pd.DataFrame({0: combined_df.index, 1: combined_df_median.values}).reset_index(drop=True)

        # Append the median spectrum to the list
        all_median_spectra.append(combined_spectra)

    # Calculate the area under the curve for each median spectrum
    areas = [calculate_area(spectra[1]) for spectra in all_median_spectra]

    # Convert to DataFrame for easier processing
    areas_df = pd.DataFrame({'spectra': all_median_spectra, 'area': areas})

    # Keep the top PERCENT_SPECTRA_TO_KEEP spectra based on the area under the curve
    top_spectra_count = int(PERCENT_SPECTRA_TO_KEEP * len(areas_df))
    top_spectra_count = max(top_spectra_count, 1)  # Ensure at least one spectrum is kept
    top_spectra_df = areas_df.nlargest(top_spectra_count, 'area')

    # Prepare to save the top spectra
    top_spectra = [row['spectra'] for idx, row in top_spectra_df.iterrows()]

    corrected_spectra_all = []

    # Process each of the top spectra
    for combined_spectra in top_spectra:
        # Apply improved baseline correction and smoothing
        combined_spectra[1] = improved_baseline_correction(combined_spectra[0], combined_spectra[1])
        combined_spectra[1] = apply_savgol_filter(combined_spectra[1])

        # Append the processed spectra to the list
        corrected_spectra_all.append(combined_spectra)

    # Combine all corrected spectra into a single dataframe
    corrected_spectra_all_df = pd.concat(corrected_spectra_all, ignore_index=True)

    # Save the combined corrected spectra to a single txt file named after the parent folder
    parent_folder_name = os.path.basename(os.path.normpath(folder_path))
    output_file_path = os.path.join(output_folder, f"{parent_folder_name}_filtered_spectra.txt")
    corrected_spectra_all_df.to_csv(output_file_path, sep='\t', index=False, header=False)

    print(f'Filtered spectra for {parent_folder_name} saved to {output_file_path}')

# Process each subfolder in the main directory
for subfolder in os.listdir(SPECTRA_DIRECTORY):
    subfolder_path = os.path.join(SPECTRA_DIRECTORY, subfolder)
    if os.path.isdir(subfolder_path):
        process_folder(subfolder_path, OUTPUT_DIRECTORY)
