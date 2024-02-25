#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 19:35:00 2024

@author: atyagi
"""

# Does what target.py does but incredibly faster (greater when your dataset is several hundred or thousand files large)

import os
import shutil
import argparse
import multiprocessing
import xarray as xr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove', action='store_true', help='remove mrms.zarr (not reccomended)')
    parser.add_argument('--time', type=int, required=True, help='timeframe (5 to 60)')
    parser.add_argument('--vt', type=int, required=True, help='VIL change threshold (in g/kg)')
    parser.add_argument('--rt', type=int, required=True, help='Reflectivity change threshold (in dBz)')
    parser.add_argument('--num', type=int, default=4, help='Number of concurrent processes')
    return parser.parse_args()

def process_directory(dirname, args):
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
        with open("../info/instances.txt", "a") as file:
            file.write(f"Instances of convective initiation in {dirname}: {instances.values}" + "\n")
        try:
            shutil.rmtree(f"../{dirname}/target.zarr/")
        except:
            pass
        newds.to_zarr(f"../{dirname}/target.zarr", mode='w', consolidated=True)
        if args.remove:
            shutil.rmtree(f"../{dirname}/mrms.zarr/")
    except Exception as e:
        with open("../info/instances.txt", "a") as file:
            file.write(f"Error in {dirname}: {e}" + "\n")

def main():
    try:
        os.remove("../info/instances.txt")
    except:
        pass
    args = parse_args()
    with open("../info/instances.txt", "a") as file:
        file.write(f"Rule: {args.vt} g/kg change in VIL and {args.rt} dBz change in ref over {args.time} minutes" + "\n")
    allitems = os.listdir("../")
    dirs = [item for item in allitems if os.path.isdir(os.path.join("../", item)) and item.startswith("20")]
    dirs = sorted(dirs)
    
    with multiprocessing.Pool(processes=args.num) as pool:
        pool.starmap(process_directory, [(dirname, args) for dirname in dirs])
    with open("../info/instances.txt", "r") as file:
        lines = file.readlines()
        lines = sorted(lines)
        lines.insert(0, lines.pop())
    with open("../info/instances.txt", "w") as file:
        file.writelines(lines)
        
if __name__ == "__main__":
    main()
