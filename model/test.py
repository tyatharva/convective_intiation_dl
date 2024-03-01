#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 20:37:14 2024

@author: Atharva Tyagi
"""

from torch.utils.data import DataLoader
from datasets import VanillaUnetDataset
from vanilla_unet import UNET
from tqdm import tqdm
import xarray as xr
import torch, time, gc, os, warnings

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))

def test():
    with torch.no_grad(), tqdm(total=len(test_dataloader), desc="Predicting") as pbar:
        for input_batch, _, dates in test_dataloader:
            input_batch = input_batch.to(device)
            output = model(input_batch)
            predictions = torch.sigmoid(output).cpu().numpy()
            for date, prediction in zip(dates, predictions):
                ds = xr.DataArray(
                    prediction,
                    dims=['time', 'lat', 'lon'],
                    coords={
                        'time': range(prediction.shape[0]),
                        'lat': range(prediction.shape[1]),
                        'lon': range(prediction.shape[2]),
                    },
                    name='prediction'
                )
                ds = ds.chunk({'time': 1, 'lat': 500, 'lon': 500})
                ds.to_zarr(f"../only_init/test/{date}/prediction.zarr", mode='w', consolidated=True)
            pbar.update(1)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_time = None
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    test_dataset = VanillaUnetDataset("../only_init/test")
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, pin_memory=True)
    model = UNET(in_channels=216, out_channels=12, features=[224, 288, 352, 416]).to(device)
    model.load_state_dict(torch.load("model_dict.pth"))
    model.eval()
    start_timer()
    test()
    end_timer_and_print("ALL DONE!")
