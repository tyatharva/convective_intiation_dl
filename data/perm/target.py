#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 13:08:51 2024

@author: Atharva Tyagi
"""

# python target.py --time 30 --vt 10 --rt 20
# You can run this on the same data many times as long as the info folder are mrms.zarr exists
# Applies the target variable calculation to all data folders in the directory (antyhing folder starts with 20)

import os
import shutil
import argparse
import xarray as xr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove', action='store_true')    # Remove MRMS files (not reccomended unless you really need to save space)
    parser.add_argument('--time', type=int, required=True)  # Time interval to look at in minutes (maximum of 65)
    parser.add_argument('--vt', type=int, required=True)    # Increase threshold for VIL (in kg/m^2)
    parser.add_argument('--rt', type=int, required=True)    # Increase threshold for -10 c reflectivity (in dBz)
    return parser.parse_args()

def main():
    try: os.remove("../info/instances.txt")
    except: pass
    args = parse_args()
    allitems = os.listdir("../")
    dirs = [item for item in allitems if os.path.isdir(os.path.join("../", item)) and item.startswith("20")]
    for dirname in dirs:
        try:
            ds = xr.open_zarr(f"../{dirname}/mrms.zarr")
            newds = xr.Dataset()
            tpar = []
            t = int((args.time)/5)
            for i in range(13, 25):
                dwvi = ds["VIL_500mabovemeansealevel"].isel(time=i-t)
                upvi = ds["VIL_500mabovemeansealevel"].isel(time=i)
                dwrf = ds["ReflectivityM10C_500mabovemeansealevel"].isel(time=i-t)
                uprf = ds["ReflectivityM10C_500mabovemeansealevel"].isel(time=i)
                tpvi = xr.where((upvi - dwvi) >= args.vt, 1, 0)
                tprf = xr.where((uprf - dwrf) >= args.rt, 1, 0)
                targ = xr.where((tpvi == 1) & (tprf == 1), 1, 0)
                tpar.append(targ)
            time_coords = ds["time"].isel(time=slice(13, 25))
            newds["time"] = time_coords
            newds["target"] = xr.concat(tpar, dim="time")
            instances = newds["target"].sum().compute()
            with open("../info/instances.txt", "a") as file: file.write(f"Instances of convective initiation in {dirname}: {instances.values}" + "\n")
            try: shutil.rmtree(f"../{dirname}/target.zarr/")
            except: pass
            newds.to_zarr(f"../{dirname}/target.zarr", mode='w', consolidated=True)
            if args.remove: shutil.rmtree(f"../{dirname}/mrms.zarr/")
        except Exception as e:
            with open("../info/instances.txt", "a") as file: file.write(f"Error in {dirname}: {e}" + "\n")

if __name__ == "__main__":
    main()
