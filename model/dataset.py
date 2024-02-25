#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 21:57:45 2024

@author: Atharva Tyagi
"""

import re
import os
import zarr
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Gets the date and time of the data from the file name
def extract_datetime(input_string):
    numbers_only = ''.join(re.findall(r'\d+', input_string))
    tim = datetime.strptime(numbers_only[:12], "%Y%m%d%H%M")
    return tim.strftime("%Y-%m-%d %H:%M")

# Custom dataset
class ConvInitData(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.sample_paths = self._get_sample_paths()

    # Gets all the paths to the data
    def _get_sample_paths(self):
        sample_paths = []
        for root, dirs, files in os.walk(self.data_dir):
            for dis in dirs:
                if dis == "inputs.zarr":
                    input_path = os.path.join(root, dis)
                    target_path = os.path.join(root, "target.zarr")
                    if os.path.exists(target_path):
                        sample_paths.append((input_path, target_path))
        return sample_paths

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        # Open the data
        input_path, target_path = self.sample_paths[index]
        input_store = zarr.open(input_path, mode='r')
        target_store = zarr.open(target_path, mode='r')
        # Print date and time of data (need to remove this when everything works)
        # print(extract_datetime(input_path))
        # Ignore lat, lon, and time variables (and others if you need to)
        input_variables = sorted([var for var in input_store.array_keys() if var not in ['time', 'lat', 'lon']])
        # input_variables = sorted([var for var in input_variables if var not in ['DPT_2maboveground', 'HGT_0Cisotherm', 'MSLMA_meansealevel', 'rtma_GUST_10maboveground', 'SOILW_0M0mbelowground', 'TMP_2maboveground', 'TSOIL_0M0mbelowground', 'PWAT_entireatmosphere_consideredasasinglelayer_']])
        # input_variables = sorted([item for item in input_variables if "1013D2mb" not in item])
        target_variables = sorted([var for var in target_store.array_keys() if var not in ['time', 'lat', 'lon']])
        # Combine variables and return tensors
        # Eager (works)
        input_tensor = torch.stack([torch.from_numpy(input_store[var][:]) for var in input_variables])
        target_tensor = torch.stack([torch.from_numpy(target_store[var][:]) for var in target_variables])
        # Lazy (have not tested yet)
        # input_tensor = [torch.from_numpy(input_store[var][:]) for var in input_variables]
        # target_tensor = [torch.from_numpy(target_store[var][:]) for var in target_variables]
        return input_tensor, target_tensor

# Initialize the dataset and dataloader and determine the device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
path_to_data = "../data"
dataset = ConvInitData(path_to_data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Test the dataloader/dataset
for batch_idx, (input_batch, target_batch) in enumerate(dataloader):
    # Print batch number
    print(f"Batch {batch_idx + 1}:")
    # Print the shape of the batch
    print("Input Batch Shape:", input_batch.shape)
    print("Target Batch Shape:", target_batch.shape, "\n")
    # Plot some data
    inslice = target_batch[0][0][0][:][:]
    plt.imshow(inslice.numpy(), cmap='viridis', origin='lower')
    plt.colorbar()
    plt.show()
    # Send the batch to the GPU (each piece of data takes up slightly more than 1GB of VRAM)
    # input_batch, target_batch = input_batch.to(DEVICE), target_batch.to(DEVICE)
    # Break after a certain number of batches are tested
    if batch_idx == 0:
        break
