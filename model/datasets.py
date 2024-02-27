#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 21:57:45 2024

@author: Atharva Tyagi
"""

import os
import zarr
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
path_to_data = "../tmpdat"

class TemporalUnetDataset(Dataset):
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
        input_variables = sorted([var for var in input_store.array_keys() if var not in ['DPT_2maboveground', 'HGT_0Cisotherm', 'MSLMA_meansealevel',
                                                                                'rtma_GUST_10maboveground', 'SOILW_0M0mbelowground', 'TMP_2maboveground',
                                                                                'TSOIL_0M0mbelowground', 'time', 'lat', 'lon', 'VGRD_10maboveground',
                                                                                'VGRD_80maboveground', 'UGRD_10maboveground', 'UGRD_80maboveground',
                                                                                'CAPE_0M3000maboveground', 'FRICV_surface', 'HPBL_surface', 'SFCR_surface']])
        input_variables = sorted([item for item in input_variables if "1013D2mb" not in item and "ABSV" not in item and "LFTX" not in item and "RELV" not in item and "500" not in item and "_180M" not in item])
        target_variables = sorted([var for var in target_store.array_keys() if var not in ['time', 'lat', 'lon']])
        # All 13 (t-60 to t-0)
        # input_tensor = torch.stack([torch.from_numpy(input_store[var][:]) for var in input_variables], dim=1)
        # Last 12 (t-55 to t-0)
        input_tensor = torch.stack([torch.from_numpy(input_store[var][1:]) for var in input_variables], dim=1)
        target_tensor = torch.stack([torch.from_numpy(target_store[var][:]) for var in target_variables], dim=1)
        date = os.path.basename(os.path.dirname(input_path))
        return input_tensor, target_tensor, date

# dataset = TemporalUnetDataset(path_to_data)
# dataloader = DataLoader(dataset, batch_size=8, pin_memory=True) # shuffle=True
# for batch_idx, (input_batch, target_batch, dates) in enumerate(dataloader):
#     print(f"Batch {batch_idx + 1} dates: {dates}")
#     print("Input Batch Shape:", input_batch.shape)
#     print("Target Batch Shape:", target_batch.shape, "\n")
#     inslice = input_batch[0][1][34][:][:] # bd11 (34) t-50
#     plt.imshow(inslice.numpy(), cmap='viridis', origin='lower')
#     plt.colorbar()
#     plt.show()
#     # input_batch, target_batch = input_batch.to(DEVICE), target_batch.to(DEVICE)
#     if batch_idx == 0:
#      	break



class VanillaUnetDataset(Dataset):
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
        input_variables = sorted([var for var in input_store.array_keys() if var not in ['DPT_2maboveground', 'HGT_0Cisotherm', 'MSLMA_meansealevel',
                                                                                'rtma_GUST_10maboveground', 'SOILW_0M0mbelowground', 'TMP_2maboveground',
                                                                                'TSOIL_0M0mbelowground', 'time', 'lat', 'lon', 'VGRD_10maboveground',
                                                                                'VGRD_80maboveground', 'UGRD_10maboveground', 'UGRD_80maboveground',
                                                                                'CAPE_0M3000maboveground', 'FRICV_surface', 'HPBL_surface', 'SFCR_surface']])
        input_variables = sorted([item for item in input_variables if "1013D2mb" not in item and "ABSV" not in item and "LFTX" not in item and "RELV" not in item and "500" not in item and "_180M" not in item])
        input_tensors = []
        # For all 13, remove the -1 in the line below and remove the +1 in the line below that
        for i in range(len(input_store["time"])-6):
            time_step_tensors = [torch.from_numpy(input_store[var][i+6]) for var in input_variables]
            time_step_tensor = torch.stack(time_step_tensors, dim=0)
            input_tensors.append(time_step_tensor)
        input_tensor = torch.cat(input_tensors, dim=0)
        target_variables = sorted([var for var in target_store.array_keys() if var not in ['time', 'lat', 'lon']])
        target_tensor = torch.cat([torch.from_numpy(target_store[var][:]) for var in target_variables], dim=0)
        date = os.path.basename(os.path.dirname(input_path))
        return input_tensor, target_tensor, date

# dataset = VanillaUnetDataset(path_to_data)
# dataloader = DataLoader(dataset, batch_size=8, pin_memory=True) # shuffle=True
# for batch_idx, (input_batch, target_batch, dates) in enumerate(dataloader):
#     print(f"Batch {batch_idx + 1} dates: {dates}")
#     print("Input Batch Shape:", input_batch.shape)
#     print("Target Batch Shape:", target_batch.shape, "\n")
#     inslice = input_batch[0][75][:][:] # bd11 (34+41*1) t-50
#     plt.imshow(inslice.numpy(), cmap='viridis', origin='lower')
#     plt.colorbar()
#     plt.show()
#     input_batch, target_batch = input_batch.to(DEVICE), target_batch.to(DEVICE)
#     if batch_idx == 0:
#      	break
