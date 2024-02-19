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

def extract_datetime(input_string):
    numbers_only = ''.join(re.findall(r'\d+', input_string))
    tim = datetime.strptime(numbers_only[:12], "%Y%m%d%H%M")
    return tim.strftime("%Y-%m-%d %H:%M")

class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.sample_paths = self._get_sample_paths()

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
        input_path, target_path = self.sample_paths[index]
        input_store = zarr.open(input_path, mode='r')
        target_store = zarr.open(target_path, mode='r')
        print(extract_datetime(input_path))
        input_variables = sorted([var for var in input_store.array_keys() if var not in ['time', 'lat', 'lon']])
        target_variables = sorted([var for var in target_store.array_keys() if var not in ['time', 'lat', 'lon']])
        input_tensor = torch.stack([torch.from_numpy(input_store[var][:]) for var in input_variables])
        target_tensor = torch.stack([torch.from_numpy(target_store[var][:]) for var in target_variables])
        return input_tensor, target_tensor

path_to_data = "../data"
batches_to_display = 1
dataset = MyDataset(path_to_data)
dataloader = DataLoader(dataset, batch_size=12, shuffle=True)
for batch_idx, (input_batch, target_batch) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}:")
    print("Input Batch Shape:", input_batch.shape)
    print("Target Batch Shape:", target_batch.shape, "\n")
    inslice = input_batch[0][71][1][:][:]
    plt.imshow(inslice.numpy(), cmap='viridis', origin='lower')
    plt.colorbar()
    plt.show()
    # input_batch, target_batch = input_batch.to("cuda"), target_batch.to("cuda")
    if batch_idx == (batches_to_display-1):
        break
