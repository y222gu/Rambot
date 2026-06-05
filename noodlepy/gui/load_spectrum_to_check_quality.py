# https://pytorch.org/get-started/locally/
from torch.utils.data import Dataset
import pandas as pd
import os
from noodlepy.utils.spectrum import Spectrum
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from noodlepy.utils.spectrumaugmentor import SpectrumAugmentor
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
from noodlepy.archive import spectrum_inspection as si
import sys
from PyQt5.QtWidgets import QApplication

# perform hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.spatial.distance import pdist
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import json

class HNC_Dataset(Dataset):
    def __init__(self, data_folder = None,
                 preprocessor = None,
                 augmentor = None):
        """
        Load the database of spectra from the txt file 

        Args:
        data_folder (str): The folder where the spectra are stored

        Returns:
        list[Spectrum]: A list of Spectrum objects
        """
        print("Loading the Raman dataset")
        self.preprocessor = preprocessor
        self.augmentor = augmentor

        list_of_spectrum_objects = []
        list_of_file_names = sorted([os.path.join(root, f) for root, _, files in os.walk(data_folder) for f in files if f.endswith('.txt')])

        for filename in list_of_file_names:
            patient_annotations = HNC_Dataset._extract_patient_labels(filename)
            spectrum_objects = HNC_Dataset._load_files_to_spectrum_objects(data_folder, filename, patient_annotations)
            if spectrum_objects != []:
                list_of_spectrum_objects+=spectrum_objects

        # preprocess the spectra
        # preprocessed_spectra = []
        # for spectrum in list_of_spectrum_objects:
        #     spectrum = self.preprocessor.preprocess(spectrum)
        #     preprocessed_spectra.append(spectrum)

        # self.db = preprocessed_spectra
        self.db = list_of_spectrum_objects

        print(f"Loaded {len(self.db)} spectra")

        
    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, 
                    idx:int
                    )-> tuple[Spectrum, Spectrum]:
        """
        Return 2 augmented spectra from the chosen spectrum

        Args:
        idx (int): The index of the spectrum to augment
        preprocessing_flag (bool): Whether to apply preprocessing to the chosen_spectrum
        augmentation_step_option_list (list[str]): The list of augmentation steps to choose and apply randomly

        returns:
        augmented_spectrum_list (list[Spectrum]): a tuple of 2 augmented Spectrum objects
        """
        augmentor = self.augmentor
        preprocessor = self.preprocessor
        # Crop, augment, preprocess, return 2 spectra
        chosen_spectrum:Spectrum = copy.deepcopy(self.db[idx])
        cropped_spectrum:Spectrum = copy.deepcopy(chosen_spectrum)
        cropped_spectrum = cropped_spectrum.crop_spectrum(625.277, 1786.791)
        preprocessed_spectrum = preprocessor.preprocess(cropped_spectrum)
        return chosen_spectrum, cropped_spectrum, preprocessed_spectrum

    
    def _extract_patient_labels(spectrum_file_path:str):

        # Patient metatdata extraction
        patient_labels = {}
        spectrum_file_name = os.path.basename(spectrum_file_path)
        f_split = spectrum_file_name.split('_')
        
        # Extract x and y positions
        x_position = float(f_split[8].split('x')[1])
        y_position = float(f_split[9].split('.txt')[0].split('y')[1])
        patient_labels['x'] = x_position
        patient_labels['y'] = y_position

        return patient_labels

    
    def _load_files_to_spectrum_objects(data_folder:str,
                               filename:str, 
                               patient_annotations:dict):
        if patient_annotations == {}: # skip the spectrum if the patient_id is not found in the metadata file
            return []
        
        with open(os.path.join(data_folder, filename)) as f:
            # if the file is empty, skip the file
            if os.stat(os.path.join(data_folder, filename)).st_size == 0:
                return []
            data = pd.read_csv(f, sep=",", header=None)

            repeated_wavelengths = data.iloc[:,0].value_counts()
            first_repeated_wavelength = repeated_wavelengths.idxmax()
            start_indexes = data[data.iloc[:,0] == first_repeated_wavelength].index.tolist()

            spectrum_objects = []
            # split the repeated measurements into individual spectra
            for i in range(len(start_indexes)):
                spectrum_id = i + 1
                if i == len(start_indexes) - 1:
                    wavelength_nm = data.iloc[start_indexes[i]:, 0].values.round(3)
                    intensity = data.iloc[start_indexes[i]:, 1].values.round(3)
                else:
                    wavelength_nm = data.iloc[start_indexes[i]:start_indexes[i + 1], 0].values.round(3)
                    intensity = data.iloc[start_indexes[i]:start_indexes[i + 1], 1].values.round(3)
                
                metadata = copy.deepcopy(patient_annotations)
                metadata['spectrum_id'] = spectrum_id
                spectrum = Spectrum(wavelength_nm=wavelength_nm,
                                    intensity=intensity,
                                    metadata=metadata,)
                spectrum_objects.append(spectrum)
        return spectrum_objects


if __name__ == "__main__":
    data_folder = r'C:\Users\yifei\Documents\RamanData'
    preprocessor = SpectrumPreprocessor(cropping=False,
                                        baseline_correction=True,
                                        remove_cosmic_rays= False,
                                        normalization=True,
                                        smoothing=False)
    
    augmentor = None
    dataset = HNC_Dataset(data_folder=data_folder, 
                          preprocessor=preprocessor, 
                          augmentor=augmentor)
    

    # display the first spectrum
    for idx in range(len(dataset)):
        spectrum = dataset.__getitem__(idx)
        spectrum[0].display(filename=f"Original Spectrum {idx}")
        spectrum[1].display(filename=f"Cropped Spectrum {idx}")
        spectrum[2].display(filename=f"Preprocessed Spectrum {idx}")

    print(f"Dataset length: {len(dataset)}")
    print(f"First spectrum metadata: {dataset[0][0].metadata}")


