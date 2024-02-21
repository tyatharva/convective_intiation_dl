#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:14:33 2023

@author: Atharva Tyagi
"""

# Requires UNIX, I reccomend a decent amount of free storage and at least 16gb of RAM
# This gets data for the last hour for everything except for MRMS, which is the previous and next hour
# You can use any MRMS variable you want but you will have to mess with the code if you are not using the defaults
# For GOES, you may need to modify a few things to use different bands
# For a differnent number of MRMS or GOES variables, you will need to get your hands dirty
# The data location method to check if the data is there before we attempt to access it covers most but not all missing data situations
# Please read the directions/settings (most of them are at the bottom in main)
# I plan to document this better at a later time

# How to run:
# Open in terminal
# cd into directory
# python data.py [--backup] --start {start date as %Y%m%d} --end {end date as %Y%m%d} --files {number of files per day}
# add --backup to the end if you want to keep backup netCDF files
# Example: python data.py --start 20230808 --end 20230808 --files 8 --backup
# This gets data for only 8/8/23 at 8 random times and grid keeping backup
# You will have to mess around with the code if you want more options (such as 1 random time or a specific grid)

import os
import time
import glob
import gzip
import boto3
import random
import shutil
import fnmatch
import argparse
import subprocess
import multiprocessing
import numpy as np
import pandas as pd
import xarray as xr
from cdo import *
from herbie import FastHerbie
from botocore import UNSIGNED
from botocore.config import Config
from datetime import datetime, timedelta



class utils():
    
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--backup', action='store_true')
        parser.add_argument('--start', required=True)
        parser.add_argument('--end', required=True)
        parser.add_argument('--files', type=int, required=True)
        return parser.parse_args()
    
    @staticmethod
    def create_dir(folder_name):
        current_directory = os.getcwd()
        parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
        main_folder_path = os.path.join(parent_directory, folder_name)
        os.makedirs(main_folder_path, exist_ok=True)
        backup_folder_path = os.path.join(main_folder_path, 'backup')
        os.makedirs(backup_folder_path, exist_ok=True)
        subfolders = ['bd02', 'bd11', 'rtma', 'hrrr', 'vil', 'rf-10', 'elev']
        for subfolder in subfolders:
            subfolder_path = os.path.join(backup_folder_path, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)
    
    @staticmethod
    def list_files_s3(bucket, prefix):
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' in response:
            files = [obj['Key'] for obj in response['Contents']]
            return files
        else: return []
    
    @staticmethod
    def elev_time(dirname, etime):
        cdo.remapnn("./mygrid", input="-setmisstoc,0 ./perm_elev.nc", options="-f nc4", output=f"../{dirname}/backup/elev/og_elev.nc")
        etime -= timedelta(hours=2)
        for i in range(2):
            etime += timedelta(hours=1)
            etime_str = etime.strftime("%Y-%m-%d,%H:%M:00,5min")
            cdo.settaxis(f"{etime_str}", input=f"../{dirname}/backup/elev/og_elev.nc", options="-f nc4 -r", output=f"../{dirname}/backup/elev/elev{i}.nc")
        etime -= timedelta(hours=1)
        etime_str = etime.strftime("%Y-%m-%d,%H:%M:00,5min")
        cdo.inttime(f"{etime_str}", input=f"-mergetime ../{dirname}/backup/elev/elev0.nc ../{dirname}/backup/elev/elev1.nc", options="-b F32 -f nc4 -r", output=f"../{dirname}/backup/elev.nc")
    
    @staticmethod
    def merge_ins(dirname, ygrd, xgrd):
        try:
            ds1 = xr.open_dataset(f"../{dirname}/backup/bd02.nc", chunks={'time': 1, 'lat': ygrd, 'lon': xgrd})
            ds2 = xr.open_dataset(f"../{dirname}/backup/bd11.nc", chunks={'time': 1, 'lat': ygrd, 'lon': xgrd})
            ds3 = xr.open_dataset(f"../{dirname}/backup/rtma.nc", chunks={'time': 1, 'lat': ygrd, 'lon': xgrd})
            ds4 = xr.open_dataset(f"../{dirname}/backup/hrrr.nc", chunks={'time': 1, 'lat': ygrd, 'lon': xgrd})
            ds5 = xr.open_dataset(f"../{dirname}/backup/elev.nc", chunks={'time': 1, 'lat': ygrd, 'lon': xgrd})
            ds = xr.merge([ds1, ds2, ds3, ds4, ds5])
            ds.to_zarr(f"../{dirname}/inputs.zarr", mode='w', consolidated=True)
        except: pass
    
    @staticmethod
    def locate_data(indate, mrmsprod1, mrmsprod2, delaytime):
        flag = 1
        dname = indate.strftime("%Y%m%d_%H%M")
        indate -= timedelta(minutes=delaytime[1])
        pdate = indate - timedelta(hours=1)
        fdate = indate + timedelta(hours=1)
        pymd = pdate.strftime("%Y%m%d")
        fymd = fdate.strftime("%Y%m%d")
        
        if pymd == fymd:
            counter = 0
            doy = str(pdate.timetuple().tm_yday).zfill(3)
            gyr = pdate.strftime("%Y")
            prd1 = utils.list_files_s3("noaa-mrms-pds", f"CONUS/{mrmsprod1}/{pymd}/")
            prd2 = utils.list_files_s3("noaa-mrms-pds", f"CONUS/{mrmsprod2}/{pymd}/")
            goes = utils.list_files_s3("noaa-goes16", f"ABI-L1b-RadC/{gyr}/{doy}/00/")
            rtma = utils.list_files_s3("noaa-rtma-pds", f"rtma2p5_ru.{pymd}/")
            hrrr = utils.list_files_s3("noaa-hrrr-bdp-pds", f"hrrr.{pymd}/conus/")
            if not prd1: counter+=1
            if not prd2: counter+=1
            if not goes: counter+=1
            if not rtma: counter+=1
            if not hrrr: counter+=1
            if counter > 0:
                relcount = counter/1
                with open("../info/warnings.txt", "a") as file: file.write(f"{dname} doesn't exist on AWS ({relcount} pieces missing)" + "\n")
                flag = 0
        
        else:
            counter = 0
            doy = str(pdate.timetuple().tm_yday).zfill(3)
            gyr = pdate.strftime("%Y")
            pprd1 = utils.list_files_s3("noaa-mrms-pds", f"CONUS/{mrmsprod1}/{pymd}/")
            pprd2 = utils.list_files_s3("noaa-mrms-pds", f"CONUS/{mrmsprod2}/{pymd}/")
            pgoes = utils.list_files_s3("noaa-goes16", f"ABI-L1b-RadC/{gyr}/{doy}/23/")
            prtma = utils.list_files_s3("noaa-rtma-pds", f"rtma2p5_ru.{pymd}/")
            phrrr = utils.list_files_s3("noaa-hrrr-bdp-pds", f"hrrr.{pymd}/conus/")
            doy = str(fdate.timetuple().tm_yday).zfill(3)
            gyr = fdate.strftime("%Y")
            fprd1 = utils.list_files_s3("noaa-mrms-pds", f"CONUS/{mrmsprod1}/{fymd}/")
            fprd2 = utils.list_files_s3("noaa-mrms-pds", f"CONUS/{mrmsprod2}/{fymd}/")
            fgoes = utils.list_files_s3("noaa-goes16", f"ABI-L1b-RadC/{gyr}/{doy}/00/")
            frtma = utils.list_files_s3("noaa-rtma-pds", f"rtma2p5_ru.{fymd}/")
            fhrrr = utils.list_files_s3("noaa-hrrr-bdp-pds", f"hrrr.{fymd}/conus/")
            if not pprd1: counter+=1
            if not pprd2: counter+=1
            if not pgoes: counter+=1
            if not prtma: counter+=1
            if not phrrr: counter+=1
            if not fprd1: counter+=1
            if not fprd2: counter+=1
            if not fgoes: counter+=1
            if not frtma: counter+=1
            if not fhrrr: counter+=1
            if counter > 0:
                relcount = counter/2
                with open("../info/warnings.txt", "a") as file: file.write(f"{dname} doesn't exist on AWS ({relcount} pieces missing)" + "\n")
                flag = 0
                
        return flag
    
    @staticmethod
    def process_data(dirname, remove):
        ins = 1182 # change to 1182 if using only mandatory levels, 2622 if all levels
        try:
            i = 0
            for prod in ["inputs", "mrms"]:
                ds = xr.open_zarr(f"../{dirname}/{prod}.zarr")
                for variable in ds.variables:
                    for timestep in ds.time:
                        try:
                            data = ds[variable].sel(time=timestep).values
                            nan_indices = np.isnan(data)
                            if np.any(nan_indices):
                                i += 1
                        except: pass
            if i > 0:
                with open("../info/warnings.txt", "a") as file: file.write(f"{dirname} contains NaN" + "\n")
            inputs_num = 0
            target_num = 0
            folder_path = f"../{dirname}/inputs.zarr"
            for root, dirs, files in os.walk(folder_path): inputs_num += len(files)
            folder_path = f"../{dirname}/mrms.zarr"
            for root, dirs, files in os.walk(folder_path): target_num += len(files)
            if (inputs_num != ins or target_num != 66):
                i+=1
                with open("../info/warnings.txt", "a") as file: file.write(f"{dirname} contains {inputs_num} inputs and {target_num} mrms" + "\n")
            print("\n" + f"Done processing {dirname}" + "\n")
        except:
            with open("../info/warnings.txt", "a") as file: file.write(f"{dirname} contains no zarr" + "\n")
            print("\n" + f"Done processing {dirname}" + "\n")



class mrms():

    def mrms(dirname, product_long, product_short, mtime, delay):
        
        gettime = mtime - timedelta(minutes=delay[0])
        modtime = gettime.minute % 2
        if modtime != 0: gettime -= timedelta(minutes=modtime)
        
        x = 0
        gettim2 = gettime + timedelta(minutes=2)
        while x < 31:
            date_str = gettim2.strftime("%Y%m%d")
            time_str = gettim2.strftime("%H%M")
            files_in_directory = utils.list_files_s3("noaa-mrms-pds", f"CONUS/{product_long}/{date_str}/")
            matching_files = [file for file in files_in_directory if fnmatch.fnmatch(file, f"*{date_str}-{time_str}*")]
    
            if matching_files:
                file_down = matching_files[0]
                file_pt1 = gettim2.strftime("%Y%m%d-%H%M")
                file_newname = f"{product_short}_{file_pt1}.grib2.gz"
                s3.download_file("noaa-mrms-pds", file_down, f"../{dirname}/backup/{product_short}/{file_newname}")
                print(f"{file_newname} downloaded successfully.")
                x += 1
            gettim2 += timedelta(minutes=2)
        
        i = 0
        while i < 30:
            date_str = gettime.strftime("%Y%m%d")
            time_str = gettime.strftime("%H%M")
            files_in_directory = utils.list_files_s3("noaa-mrms-pds", f"CONUS/{product_long}/{date_str}/")
            matching_files = [file for file in files_in_directory if fnmatch.fnmatch(file, f"*{date_str}-{time_str}*")]
    
            if matching_files:
                file_down = matching_files[0]
                file_pt1 = gettime.strftime("%Y%m%d-%H%M")
                file_newname = f"{product_short}_{file_pt1}.grib2.gz"
                s3.download_file("noaa-mrms-pds", file_down, f"../{dirname}/backup/{product_short}/{file_newname}")
                print(f"{file_newname} downloaded successfully.")
                i += 1
            gettime -= timedelta(minutes=2)
        
        files_to_process = glob.glob(f"../{dirname}/backup/{product_short}/*.gz")
        files_to_process = sorted(files_to_process)

        for file in files_to_process:
            with gzip.open(file, 'rb') as f_in, open(file[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(file)
        
        tonc = [
            "bash", "-c",
            f"for file in ../{dirname}/backup/{product_short}/*.grib2; do wgrib2 \"$file\" -nc4 -netcdf \"${{file%.grib2}}.nc\"; done"
        ]
        subprocess.run(tonc)
        
        mergetime = [
            "cdo",
            "-f", "nc4",
            "mergetime",
            f"../{dirname}/backup/{product_short}/*.nc",
            f"../{dirname}/backup/{product_short}/{product_short}tmp.nc"
        ]
        subprocess.run(mergetime)
        
        gettime += timedelta(minutes=2)
        itime = gettime.strftime("%Y-%m-%d,%H:%M:00")
        cdo.inttime(f"{itime},5min", input=f"-settaxis,{itime},2min -setmisstoc,0 -setrtomiss,-1000,0 ../{dirname}/backup/{product_short}/{product_short}tmp.nc", options='-f nc4 -r', output=f"../{dirname}/backup/{product_short}/{product_short}tmpp.nc")
        
        tmptime = mtime - timedelta(hours=1)
        stime = tmptime.strftime("%Y-%m-%d,%H:%M:00,5min")
        settaxis = [
            "cdo",
            "-f", "nc4", "-r",
            f"settaxis,{stime}",
            f"../{dirname}/backup/{product_short}/{product_short}tmpp.nc",
            f"../{dirname}/backup/{product_short}/{product_short}tmppp.nc"
        ]
        subprocess.run(settaxis)
        
        remap = [
            "cdo",
            "-b", "F32", "-f", "nc4",
            "remapnn,./mygrid",
            f"../{dirname}/backup/{product_short}/{product_short}tmppp.nc",
            f"../{dirname}/backup/{product_short}.nc"
        ]
        subprocess.run(remap)
        
        remove = [f"rm ../{dirname}/backup/{product_short}/*.nc"]
        subprocess.run(remove, shell=True)
        
    def merge_mrms(dirname, ygrd, xgrd):
        try:
            ds1 = xr.open_dataset(f"../{dirname}/backup/vil.nc", chunks={'time': 1, 'lat': ygrd, 'lon': xgrd})
            ds2 = xr.open_dataset(f"../{dirname}/backup/rf-10.nc", chunks={'time': 1, 'lat': ygrd, 'lon': xgrd})
            ds = xr.merge([ds1, ds2])
            ds.to_zarr(f"../{dirname}/mrms.zarr", mode='w', consolidated=True)
        except: pass



class hrrr():

    def mfilerdir_hrrr(directory):
        items = os.listdir(directory)
        for item in items:
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                for root, dirs, files in os.walk(item_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        new_file_name = file.split("__", 1)[-1]
                        original_folder_name = os.path.basename(item_path)
                        new_path = os.path.join(directory, original_folder_name + "_" + new_file_name)
                        shutil.move(file_path, new_path)
                shutil.rmtree(item_path)
                
    def hrrr(dirname, htime, thds, delay):
        
        hrtime = htime - timedelta(hours=1, minutes=delay[1])
        hrtime = hrtime.replace(minute=0)
        DATES = pd.date_range(start=hrtime.strftime("%Y-%m-%d %H:00"), periods=2, freq="1H",)
        fxx=range(0,1)
        data = FastHerbie(DATES, model="hrrr", product="prs", fxx=fxx, max_threads=thds,)
        #                       Soil temp and moisture at 0m       Standard vars at 500-1000mb every 25mb and 1013.2mb                                               Wind at 10 and 80m                                                        Equilibrium level           Lowest condensation level           Level of free convection (many aliases)
        # Data for vertical levels 500-100 step of 25 mb and 1013.2 mb (all levels, file size around 1.6gb, total variables around 174)
        # data.download(searchString="(0-0 m below ground)|((TMP|DPT|VVEL|UGRD|VGRD|ABSV):(([5-9][0,2,5,7][0,5])|(10[0,1][0,3])))|(CAPE)|(CIN)|(FRICV)|(MSLMA)|(RELV)|([U\|V]GRD:[1,8]0 m)|(SNOWC)|(ICEC)|(LAND)|((TMP|DPT):2 m)|(PWAT)|(HPBL)|(HGT:equilibrium level)|(HGT:level of adiabatic condensation from sfc)|(HGT:((reserved)|(no_level)|(level of free convection)))|(HGT:0C isotherm)|(LFTX)|(SFCR)", max_threads=thds, save_dir = f"../{dirname}/backup/")
        # Data for vertical levels 500, 700, 850, 925, 1000, 1013.2mb (mandatory levels, file size around 500mb, 78 total variables)
        data.download(searchString="(0-0 m below ground)|((TMP|DPT|VVEL|UGRD|VGRD|ABSV):(500|700|850|925|(10[0,1][0,3])))|(CAPE)|(CIN)|(FRICV)|(MSLMA)|(RELV)|([U\|V]GRD:[1,8]0 m)|(SNOWC)|(ICEC)|(LAND)|((TMP|DPT):2 m)|(PWAT)|(HPBL)|(HGT:equilibrium level)|(HGT:level of adiabatic condensation from sfc)|(HGT:((reserved)|(no_level)|(level of free convection)))|(HGT:0C isotherm)|(LFTX)|(SFCR)", max_threads=thds, save_dir = f"../{dirname}/backup/")
        # Make sure the ins variable in utils.process_data matches the data you are getting
        hrrr.mfilerdir_hrrr(f"../{dirname}/backup/hrrr/")
        
        tonc = [
            "bash", "-c",
            f"for file in ../{dirname}/backup/hrrr/*.grib2; do wgrib2 \"$file\" -nc4 -netcdf \"${{file%.grib2}}.nc\"; done"
        ]
        subprocess.run(tonc)
        
        tstime = htime - timedelta(hours=1)
        stime = tstime.strftime("%Y-%m-%d,%H:%M:00,5min")
        ltime = hrtime.strftime("%Y-%m-%d,%H:%M:00,5min")
        h1 = hrtime.strftime("%H")
        hrtime += timedelta(hours=1)
        h2 = hrtime.strftime("%H")
        f1 = glob.glob(f"../{dirname}/backup/hrrr/*t{h1}z*.nc")[0]
        f2 = glob.glob(f"../{dirname}/backup/hrrr/*t{h2}z*.nc")[0]
        cdo.remapnn("./mygrid", input=f"-chname,HGT_no_level,HGT_leveloffreeconvection -chname,HGT_reserved,HGT_leveloffreeconvection -settaxis,{stime} -inttime,{ltime} -mergetime {f1} {f2}", options=f"-b F32 -P {thds} -f nc4 -r", output=f"../{dirname}/backup/hrrr.nc")
        
        remove = [f"rm ../{dirname}/backup/hrrr/*.nc"]
        subprocess.run(remove, shell=True)



class rtma():

    def rtma(dirname, rtime, delay):
        
        gettime = rtime - timedelta(minutes=delay[2])
        remain = gettime.minute % 15
        if remain != 0: gettime -= timedelta(minutes=remain)
        
        i = 0
        while i < 5:
            date_str = gettime.strftime("%Y%m%d")
            time_str = gettime.strftime("%H%M")
            s3.download_file("noaa-rtma-pds", f"rtma2p5_ru.{date_str}/rtma2p5_ru.t{time_str}z.2dvaranl_ndfd.grb2", f"../{dirname}/backup/rtma/rtma_{date_str}_{time_str}.grb2")
            gettime -= timedelta(minutes=15)
            i += 1
        
        tonc = [
            "bash", "-c",
            f"for file in ../{dirname}/backup/rtma/*.grb2; do wgrib2 \"$file\" -nc4 -netcdf \"${{file%.grb2}}.nc\"; done"
        ]
        subprocess.run(tonc)
        
        gettime += timedelta(minutes=15)
        ttim_str = gettime.strftime("%Y-%m-%d,%H:%M:00,5min")
        
        tretime = rtime - timedelta(hours=1)
        ftim_str = tretime.strftime("%Y-%m-%d,%H:%M:00,5min")
        
        cdo.settaxis(f"{ftim_str}", input=f"-inttime,{ttim_str} -remapnn,./mygrid -chname,DPT_2maboveground,rtma_DPT_2maboveground,GUST_10maboveground,rtma_GUST_10maboveground,PRES_surface,rtma_PRES_surface,TMP_2maboveground,rtma_TMP_2maboveground,UGRD_10maboveground,rtma_UGRD_10maboveground,VGRD_10maboveground,rtma_VGRD_10maboveground -delname,HGT_surface,CEIL_cloudceiling,TCDC_entireatmosphere_consideredasasinglelayer_,VIS_surface,WDIR_10maboveground,WIND_10maboveground,SPFH_2maboveground -mergetime ../{dirname}/backup/rtma/*.nc", options='-b F32 -f nc4 -r', output=f"../{dirname}/backup/rtma.nc")
        
        remove = [f"rm ../{dirname}/backup/rtma/*.nc"]
        subprocess.run(remove, shell=True)



class goes():

    def goes(dirname, product, bandnum, gtime, delay):
        
        gettime = gtime - timedelta(minutes=delay[3])
        i = 0
        while i < 13:
            minute_str = gettime.strftime("%M").zfill(2)
            hour_str = gettime.strftime("%H").zfill(2)
            doy_str = str(gettime.timetuple().tm_yday).zfill(3)
            year_str = gettime.strftime("%Y").zfill(4)
            band_str = str(bandnum).zfill(2)
            files_in_directory = utils.list_files_s3("noaa-goes16", f"ABI-L1b-RadC/{year_str}/{doy_str}/{hour_str}/")
            matching_files = [file for file in files_in_directory if fnmatch.fnmatch(file, f"*C{band_str}_G16_s???????{hour_str}{minute_str}*.nc")]
    
            if matching_files:
                file_down = matching_files[0]
                file_newname = gettime.strftime("%Y%m%d-%H%M.nc")
                s3.download_file("noaa-goes16", file_down, f"../{dirname}/backup/{product}/{file_newname}")
                print(f"{product} {file_newname} downloaded successfully.")
                i += 1
            gettime -= timedelta(minutes=1)
        
        if (product == "bd11"):
            toref = [
                "bash", "-c",
                f"for file in ../{dirname}/backup/bd11/*.nc; do cdo -f nc4 expr,'bright=(planck_fk2/(log((planck_fk1/Rad)+1))-planck_bc1)/planck_bc2;' \"$file\" \"${{file%.nc}}_tmp1.nc\" && gdalwarp -q -s_srs \"+proj=geos +h=35786023.0 +a=6378137.0 +b=6356752.31414 +f=0.0033528106647475126 +lon_0=-75.0 +sweep=x +no_defs\" -t_srs EPSG:4326 -r near \"${{file%.nc}}_tmp1.nc\" \"${{file%.nc}}_tmp2.nc\"; done"
            ]
        
        else:
            toref = [
                "bash", "-c",
                f"for file in ../{dirname}/backup/bd02/*.nc; do cdo -f nc4 expr,'2ref=kappa0*Rad;' \"$file\" \"${{file%.nc}}_tmp1.nc\" && gdalwarp -q -s_srs \"+proj=geos +h=35786023.0 +a=6378137.0 +b=6356752.31414 +f=0.0033528106647475126 +lon_0=-75.0 +sweep=x +no_defs\" -t_srs EPSG:4326 -r near \"${{file%.nc}}_tmp1.nc\" \"${{file%.nc}}_tmp2.nc\"; done"
            ]
        
        subprocess.run(toref)
        
        files = glob.glob(f"../{dirname}/backup/{product}/*_tmp2.nc")
        files = sorted(files)
        for file in files:
            gettime = gettime + timedelta(minutes=5)
            newname = gettime.strftime("%Y%m%d_%H%M_tmp3")
            cdo.settaxis(gettime.strftime("%Y-%m-%d,%H:%M:00,5min"), input=f"{file}", options='-f nc4 -r', output=f"../{dirname}/backup/{product}/{newname}.nc")
        
        gtime -= timedelta(hours=1)
        time_str = gtime.strftime("%Y-%m-%d,%H:%M:00,5min")
        if (product == "bd11"): cdo.remapnn('./mygrid', input=f"-settaxis,{time_str} -setunit,Kelvin -chname,Band1,{product} -mergetime ../{dirname}/backup/{product}/*_tmp3.nc", options='-b F32 -f nc4 -r', output=f"../{dirname}/backup/{product}.nc")
        else: cdo.remapnn('./mygrid', input=f"-settaxis,{time_str} -setmisstoc,1 -setrtomiss,1,10 -setmisstoc,0 -setrtomiss,-1,0.01 -chname,Band1,{product} -mergetime ../{dirname}/backup/{product}/*_tmp3.nc", options="-b F32 -f nc4 -r", output=f"../{dirname}/backup/{product}.nc")
        
        remove = [f"rm ../{dirname}/backup/{product}/*_tmp?.nc"]
        subprocess.run(remove, shell=True)



if __name__ == "__main__":
    
    st = time.time()
    args = utils.parse_args()
    tout = 500                                                                # Set the timeout time for the processes (you may have to experiment with this based on your system)
    total_att = 5                                                             # Set the maximum number of reattempts if something goes wrong
    try: shutil.rmtree("../info")
    except: pass
    os.makedirs("../info", exist_ok=True)
    os.environ["REMAP_EXTRAPOLATE"] = "off"
    cdo = Cdo()
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    # mrms, hrrr, rtma, goes
    delaytimes = [3, 55, 20, 5]                                               # Delaytimes for retriving data (meant to simulate realtime)
    gridtype = "lonlat"                                                       # Don't change this
    xsize = 500                                                               # Points in x direction (you will have to fix the random grid generation and other things if you change this from default)
    ysize = 500                                                               # Points in y direction (you will have to fix the random grid generation and other things if you change this from default)
    xinc = 0.01                                                               # x increment (it is reccomended to keep this as default)
    yinc = 0.01                                                               # y increment (it is reccomended to keep this as default)
    # Largest possible domain that you can select smaller grids from: 4500(x)*2500(y) x starting at 116.1 y starting at 25
    fpd = args.files                                                          # --files 8 (maximum while preserving randomness is 8)
    stdate_gb = datetime.strptime(args.start,"%Y%m%d")                        # --start 20230808
    eddate_gb = datetime.strptime(args.end,"%Y%m%d")                          # --end 20230808
    step_gb = timedelta(days=1)                                               # timestep (1 day)
    files_done = 0
    prev_time = datetime(2000, 1, 1, 0, 0)
    prev_dirname = "20000101_0000"
    
    for i in range((eddate_gb - stdate_gb).days +1):
        date_cr = stdate_gb + i * step_gb
        for s in range(fpd):
            xfirst = round(random.uniform(-116.1, -76.1), 2)                  # This is random, if you want specific change it
            yfirst = round(random.uniform(25, 45), 2)                         # Same as above
            grid_specs = f"""gridtype = {gridtype}
            xsize    = {xsize}
            ysize    = {ysize}
            xfirst   = {xfirst}
            xinc     = {xinc}
            yfirst   = {yfirst}
            yinc     = {yinc}
            """
            with open("./mygrid", "w") as file: file.write(grid_specs)
            hour_cr = np.random.randint(s*(24/fpd), (s*(24/fpd))+(24/fpd))
            minute_cr = np.random.randint(0, 12) * 5
            datetime_cr = date_cr + timedelta(hours=hour_cr, minutes=minute_cr)
            if files_done > 0:
                duration = datetime_cr - prev_time
                duration_s = duration.total_seconds()
                mins_diff = divmod(duration_s, 60)[0]
                while mins_diff < 125:
                    hour_cr = np.random.randint(s*(24/fpd), (s*(24/fpd))+(24/fpd))
                    minute_cr = np.random.randint(0, 12) * 5
                    datetime_cr = date_cr + timedelta(hours=hour_cr, minutes=minute_cr)
                    duration = datetime_cr - prev_time
                    duration_s = duration.total_seconds()
                    mins_diff = divmod(duration_s, 60)[0]
            
            if utils.locate_data(datetime_cr, "VIL_00.50", "Reflectivity_-10C_00.50", delaytimes) == 1:
                lst = time.time()
                print("\n" + datetime_cr.strftime("%Y-%m-%d %H:%M") + " has been found\n")
                dirName = datetime_cr.strftime("%Y%m%d_%H%M")
                check = multiprocessing.Process(target=utils.process_data, args=(prev_dirname, True, ))
                if files_done > 0: check.start()
                attempt = 1
                while attempt <= total_att:
                    utils.create_dir(dirName)
                    tfm_mvil = multiprocessing.Process(target=mrms.mrms, args=(dirName, "VIL_00.50", "vil", datetime_cr, delaytimes, ))
                    tfm_rf10 = multiprocessing.Process(target=mrms.mrms, args=(dirName, "Reflectivity_-10C_00.50", "rf-10", datetime_cr, delaytimes, ))
                    tfm_rtma = multiprocessing.Process(target=rtma.rtma, args=(dirName, datetime_cr, delaytimes, ))
                    tfm_hrrr = multiprocessing.Process(target=hrrr.hrrr, args=(dirName, datetime_cr, 1, delaytimes, ))
                    tfm_bd02 = multiprocessing.Process(target=goes.goes, args=(dirName, "bd02", 2, datetime_cr, delaytimes, ))
                    tfm_bd11 = multiprocessing.Process(target=goes.goes, args=(dirName, "bd11", 11, datetime_cr, delaytimes, ))
                    tfm_elev = multiprocessing.Process(target=utils.elev_time, args=(dirName, datetime_cr, ))
                    mrge_mrms = multiprocessing.Process(target=mrms.merge_mrms, args=(dirName, ysize, xsize, ))
                    mrge_file = multiprocessing.Process(target=utils.merge_ins, args=(dirName, ysize, xsize, ))
                    tfm_hrrr.start()
                    tfm_bd02.start()
                    tfm_bd11.start()
                    tfm_rtma.start()
                    tfm_rf10.start()
                    tfm_mvil.start()
                    tfm_elev.start()
                    tfm_elev.join(tout)
                    tfm_mvil.join(tout)
                    tfm_rf10.join(tout)
                    tfm_rtma.join(tout)
                    tfm_bd11.join(tout)
                    tfm_bd02.join(tout)
                    tfm_hrrr.join(tout)
                    mrge_mrms.start()
                    mrge_file.start()
                    mrge_mrms.join(tout)
                    mrge_file.join(tout)
                    if not (os.path.exists(f"../{dirName}/inputs.zarr/") and os.path.exists(f"../{dirName}/mrms.zarr/")):
                        if attempt != total_att:
                            shutil.rmtree(f"../{dirName}/")
                            err = f"{dirName} retry #{attempt} (attempt #{attempt+1})"
                            print("\n\n" + err + "\n\n")
                            with open("../info/retries.txt", "a") as file: file.write(err + "\n")
                        attempt += 1
                        time.sleep(1)
                    else: attempt = total_att + 1
                
                if files_done > 0: check.join(tout)
                if not args.backup: shutil.rmtree(f"../{dirName}/backup/") # if you want to keep backup netCDF files, --backup
                let = time.time()
                lti = round(let-lst, 3)
                timing = f"{dirName} done in {lti} seconds\n"
                with open("../info/timings.txt", "a") as file: file.write(timing)
                print("\n" + timing)
                shutil.copy("./mygrid", f"../{dirName}/grid.txt")
                prev_time = datetime_cr
                prev_dirname = dirName
                files_done += 1
                
            else: print("\n" + datetime_cr.strftime("%Y-%m-%d %H:%M") + " does not exist\n")
    
    if files_done > 0: utils.process_data(prev_dirname, True)
    et = time.time()
    ti = et - st
    print("\n" + f"Completed {files_done} files from {args.start} to {args.end} with {args.files} files per day in ", ti, "seconds\n")
