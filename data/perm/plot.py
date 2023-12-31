#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 01:30:00 UTC 2023

@author: Atharva Tyagi
"""

import argparse
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--prod', type=str, nargs='?')
    parser.add_argument('--var', type=str, nargs='?')
    parser.add_argument('--time', type=int, nargs='?')
    return parser.parse_args()

def randvar(ds):
    variables = [var for var in ds.variables if var not in ['time', 'lat', 'lon']]
    return random.choice(variables)

def randtime(ds):
    return random.randint(0, len(ds['time']) - 1)

def main():
    args = parse_args()
    
    if args.prod: products = [args.prod]
    else: products = ['goes', 'mrms', 'rtma', 'hrrr']
    
    for prod in products:
        bigds = xr.open_zarr(f"../{args.dir}/{prod}.zarr")
        if args.var: var_name = args.var
        else: var_name = randvar(bigds)
        if args.time: timestamp = args.time
        else: timestamp = randtime(bigds)
        vards = bigds[var_name]
        tslice = vards.isel(time=timestamp)
        
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        plt.imshow(tslice, extent=(tslice.lon.min(), tslice.lon.max(), tslice.lat.min(), tslice.lat.max()), origin='lower')
        plt.colorbar()
        plt.title(f"../{args.dir}/{prod}.zarr: {var_name} @ {timestamp}")
        
        if args.print: print(f"\n{tslice}\n")
        if args.save: plt.savefig(f"../PLOT_{prod}_{var_name}_{timestamp}.png")
        if args.plot: plt.show()

if __name__ == "__main__":
    main()
