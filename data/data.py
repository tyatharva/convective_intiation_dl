#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:14:33 2023

@author: Atharva Tyagi
"""

# Requires UNIX, I reccomend a solid amount of storage at at least 32gb of RAM (files can be several dozen gb large)
# This gets data for the last hour for everything except for MRMS, which is the previous and next hour
# You can use any MRMS variable you want
# For GOES, you may need to modify a few things to use different bands
# For a differnent number of MRMS or GOES variables, you will need to get your hands dirty
# The data location method to check if the data is there before we attempt to access it covers most but not all missing data situations
# Please read the directions/settings (most of them are at the bottom in main)
# I plan to document this better at a later time

import os
import time
import glob
import gzip
import boto3
import shutil
import fnmatch
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
    def create_dir(folder_name):
        current_directory = os.getcwd()
        main_folder_path = os.path.join(current_directory, folder_name)
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
        cdo.remapnn("./perm/mygrid", input="-setmisstoc,0 ./perm/perm_elev.nc", options="-f nc4", output=f"./{dirname}/backup/elev/og_elev.nc")
        etime -= timedelta(hours=2)
        for i in range(2):
            etime += timedelta(hours=1)
            etime_str = etime.strftime("%Y-%m-%d,%H:%M:00,5min")
            cdo.settaxis(f"{etime_str}", input=f"./{dirname}/backup/elev/og_elev.nc", options="-f nc4 -r", output=f"./{dirname}/backup/elev/elev{i}.nc")
        etime -= timedelta(hours=1)
        etime_str = etime.strftime("%Y-%m-%d,%H:%M:00,5min")
        cdo.inttime(f"{etime_str}", input=f"-mergetime ./{dirname}/backup/elev/elev0.nc ./{dirname}/backup/elev/elev1.nc", options="-b F32 -f nc4 -r", output=f"./{dirname}/backup/elev.nc")
    
    @staticmethod
    def merge_ins(dirname, ygrd, xgrd):
        try:
            ds1 = xr.open_dataset(f"./{dirname}/backup/bd02.nc", chunks={'time': 1, 'lat': ygrd, 'lon': xgrd})
            ds2 = xr.open_dataset(f"./{dirname}/backup/bd11.nc", chunks={'time': 1, 'lat': ygrd, 'lon': xgrd})
            ds3 = xr.open_dataset(f"./{dirname}/backup/rtma.nc", chunks={'time': 1, 'lat': ygrd, 'lon': xgrd})
            ds4 = xr.open_dataset(f"./{dirname}/backup/hrrr.nc", chunks={'time': 1, 'lat': ygrd, 'lon': xgrd})
            ds5 = xr.open_dataset(f"./{dirname}/backup/elev.nc", chunks={'time': 1, 'lat': ygrd, 'lon': xgrd})
            ds = xr.merge([ds1, ds2, ds3, ds4, ds5])
            ds.to_zarr(f"./{dirname}/inputs.zarr", mode='w', consolidated=True)
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
                with open("./warnings.txt", "a") as file: file.write(f"{dname} doesn't exist on AWS ({relcount} pieces missing)" + "\n")
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
                with open("./warnings.txt", "a") as file: file.write(f"{dname} doesn't exist on AWS ({relcount} pieces missing)" + "\n")
                flag = 0
                
        return flag
    
    @staticmethod
    def process_data(dirname, remove):
        try:
            i = 0
            for prod in ["inputs", "mrms"]:
                ds = xr.open_zarr(f"./{dirname}/{prod}.zarr")
                for variable in ds.variables:
                    for timestep in ds.time:
                        try:
                            data = ds[variable].sel(time=timestep).values
                            nan_indices = np.isnan(data)
                            if np.any(nan_indices):
                                i += 1
                        except: pass
            if i > 0:
                with open("./warnings.txt", "a") as file: file.write(f"{dirname} contains NaN" + "\n")
            inputs_num = 0
            target_num = 0
            folder_path = f"./{dirname}/inputs.zarr"
            for root, dirs, files in os.walk(folder_path): inputs_num += len(files)
            folder_path = f"./{dirname}/mrms.zarr"
            for root, dirs, files in os.walk(folder_path): target_num += len(files)
            if (inputs_num != 2622 or target_num != 66):
                i+=1
                with open("./warnings.txt", "a") as file: file.write(f"{dirname} contains {inputs_num} inputs and {target_num} targets" + "\n")
            if i==0:
                ds = xr.open_zarr(f"./{dirname}/mrms.zarr")
                newds = xr.Dataset()
                tpar = []
                for i in range(12):
                    dwvi = ds["VIL_500mabovemeansealevel"].isel(time=i)
                    upvi = ds["VIL_500mabovemeansealevel"].isel(time=i+13)                # Edit time threshold for VIL
                    dwrf = ds["ReflectivityM10C_500mabovemeansealevel"].isel(time=i)
                    uprf = ds["ReflectivityM10C_500mabovemeansealevel"].isel(time=i+13)   # Edit time threshold for -10 c ref
                    tpvi = xr.where((upvi - dwvi) >= 10, 1, 0)                            # default 10 kg/m^2 over 30mins threshold for VIL
                    tprf = xr.where((uprf - dwrf) >= 20, 1, 0)                            # default 20 dBz over 30mins threshold for -10 c ref
                    targ = xr.where((tpvi == 1) & (tprf == 1), 1, 0)
                    tpar.append(targ)
                time_coords = ds["time"].isel(time=slice(13, 25))
                newds["time"] = time_coords
                newds["target"] = xr.concat(tpar, dim="time")
                instances = newds["target"].sum().compute()
                with open("./instances.txt", "a") as file: file.write(f"Instances of convective initiation in {dirname}: {instances.values}" + "\n")
                newds.to_zarr(f"./{dirname}/target.zarr", mode='w', consolidated=True)
                if remove: shutil.rmtree(f"./{dirname}/mrms.zarr/")
                print("\n" + f"Done processing {dirname}" + "\n")
        except:
            with open("./warnings.txt", "a") as file: file.write(f"{dirname} contains no zarr" + "\n")



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
                s3.download_file("noaa-mrms-pds", file_down, f"./{dirname}/backup/{product_short}/{file_newname}")
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
                s3.download_file("noaa-mrms-pds", file_down, f"./{dirname}/backup/{product_short}/{file_newname}")
                print(f"{file_newname} downloaded successfully.")
                i += 1
            gettime -= timedelta(minutes=2)
        
        files_to_process = glob.glob(f"./{dirname}/backup/{product_short}/*.gz")
        files_to_process = sorted(files_to_process)

        for file in files_to_process:
            with gzip.open(file, 'rb') as f_in, open(file[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(file)
        
        tonc = [
            "bash", "-c",
            f"for file in ./{dirname}/backup/{product_short}/*.grib2; do wgrib2 \"$file\" -nc4 -netcdf \"${{file%.grib2}}.nc\"; done"
        ]
        subprocess.run(tonc)
        
        mergetime = [
            "cdo",
            "-f", "nc4",
            "mergetime",
            f"./{dirname}/backup/{product_short}/*.nc",
            f"./{dirname}/backup/{product_short}/{product_short}tmp.nc"
        ]
        subprocess.run(mergetime)
        
        gettime += timedelta(minutes=2)
        itime = gettime.strftime("%Y-%m-%d,%H:%M:00")
        cdo.inttime(f"{itime},5min", input=f"-settaxis,{itime},2min -setmisstoc,0 -setrtomiss,-1000,0 ./{dirname}/backup/{product_short}/{product_short}tmp.nc", options='-f nc4 -r', output=f"./{dirname}/backup/{product_short}/{product_short}tmpp.nc")
        
        tmptime = mtime - timedelta(hours=1)
        stime = tmptime.strftime("%Y-%m-%d,%H:%M:00,5min")
        settaxis = [
            "cdo",
            "-f", "nc4", "-r",
            f"settaxis,{stime}",
            f"./{dirname}/backup/{product_short}/{product_short}tmpp.nc",
            f"./{dirname}/backup/{product_short}/{product_short}tmppp.nc"
        ]
        subprocess.run(settaxis)
        
        remap = [
            "cdo",
            "-b", "F32", "-f", "nc4",
            "remapnn,./perm/mygrid",
            f"./{dirname}/backup/{product_short}/{product_short}tmppp.nc",
            f"./{dirname}/backup/{product_short}.nc"
        ]
        subprocess.run(remap)
        
        remove = [f"rm ./{dirname}/backup/{product_short}/*.nc"]
        subprocess.run(remove, shell=True)
        
    def merge_mrms(dirname, ygrd, xgrd):
        try:
            ds1 = xr.open_dataset(f"./{dirname}/backup/vil.nc", chunks={'time': 1, 'lat': ygrd, 'lon': xgrd})
            ds2 = xr.open_dataset(f"./{dirname}/backup/rf-10.nc", chunks={'time': 1, 'lat': ygrd, 'lon': xgrd})
            ds = xr.merge([ds1, ds2])
            ds.to_zarr(f"./{dirname}/mrms.zarr", mode='w', consolidated=True)
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
        #                       Soil temp and moisture at 0m       Standard vars at 500-1000mb every 25mb and 1013.2mb                                               Wind at 10 and 80m                                                        Equilibrium level           Lowest condensation level           Level of free convection (shows up as no_level sometimes)
        data.download(searchString="(0-0 m below ground)|((TMP|DPT|VVEL|UGRD|VGRD|ABSV):(([5-9][0,2,5,7][0,5])|(10[0,1][0,3])))|(CAPE)|(CIN)|(FRICV)|(MSLMA)|(RELV)|([U\|V]GRD:[1,8]0 m)|(SNOWC)|(ICEC)|(LAND)|((TMP|DPT):2 m)|(PWAT)|(HPBL)|(HGT:equilibrium level)|(HGT:level of adiabatic condensation from sfc)|(HGT:((no_level)|(level of free convection)))|(HGT:0C isotherm)|(LFTX)|(VGTYP)", max_threads=thds, save_dir = f"./{dirname}/backup/")
        hrrr.mfilerdir_hrrr(f"./{dirname}/backup/hrrr/")
        
        tonc = [
            "bash", "-c",
            f"for file in ./{dirname}/backup/hrrr/*.grib2; do wgrib2 \"$file\" -nc4 -netcdf \"${{file%.grib2}}.nc\"; done"
        ]
        subprocess.run(tonc)
        
        tstime = htime - timedelta(hours=1)
        stime = tstime.strftime("%Y-%m-%d,%H:%M:00,5min")
        ltime = hrtime.strftime("%Y-%m-%d,%H:%M:00,5min")
        h1 = hrtime.strftime("%H")
        hrtime += timedelta(hours=1)
        h2 = hrtime.strftime("%H")
        f1 = glob.glob(f"./{dirname}/backup/hrrr/*t{h1}z*.nc")[0]
        f2 = glob.glob(f"./{dirname}/backup/hrrr/*t{h2}z*.nc")[0]
        cdo.remapnn("./perm/mygrid", input=f"-settaxis,{stime} -inttime,{ltime} -mergetime {f1} {f2}", options=f"-b F32 -P {thds} -f nc4 -r", output=f"./{dirname}/backup/hrrr.nc")
        
        remove = [f"rm ./{dirname}/backup/hrrr/*.nc"]
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
            s3.download_file("noaa-rtma-pds", f"rtma2p5_ru.{date_str}/rtma2p5_ru.t{time_str}z.2dvaranl_ndfd.grb2", f"./{dirname}/backup/rtma/rtma_{date_str}_{time_str}.grb2")
            gettime -= timedelta(minutes=15)
            i += 1
        
        tonc = [
            "bash", "-c",
            f"for file in ./{dirname}/backup/rtma/*.grb2; do wgrib2 \"$file\" -nc4 -netcdf \"${{file%.grb2}}.nc\"; done"
        ]
        subprocess.run(tonc)
        
        gettime += timedelta(minutes=15)
        ttim_str = gettime.strftime("%Y-%m-%d,%H:%M:00,5min")
        
        tretime = rtime - timedelta(hours=1)
        ftim_str = tretime.strftime("%Y-%m-%d,%H:%M:00,5min")
        
        cdo.settaxis(f"{ftim_str}", input=f"-inttime,{ttim_str} -remapnn,./perm/mygrid -chname,DPT_2maboveground,rtma_DPT_2maboveground,GUST_10maboveground,rtma_GUST_10maboveground,PRES_surface,rtma_PRES_surface,TMP_2maboveground,rtma_TMP_2maboveground,UGRD_10maboveground,rtma_UGRD_10maboveground,VGRD_10maboveground,rtma_VGRD_10maboveground -delname,HGT_surface,CEIL_cloudceiling,TCDC_entireatmosphere_consideredasasinglelayer_,VIS_surface,WDIR_10maboveground,WIND_10maboveground,SPFH_2maboveground -mergetime ./{dirname}/backup/rtma/*.nc", options='-b F32 -f nc4 -r', output=f"./{dirname}/backup/rtma.nc")
        
        remove = [f"rm ./{dirname}/backup/rtma/*.nc"]
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
                s3.download_file("noaa-goes16", file_down, f"./{dirname}/backup/{product}/{file_newname}")
                print(f"{product} {file_newname} downloaded successfully.")
                i += 1
            gettime -= timedelta(minutes=1)
        
        if (product == "bd11"):
            toref = [
                "bash", "-c",
                f"for file in ./{dirname}/backup/bd11/*.nc; do cdo -f nc4 expr,'bright=(planck_fk2/(log((planck_fk1/Rad)+1))-planck_bc1)/planck_bc2;' \"$file\" \"${{file%.nc}}_tmp1.nc\" && gdalwarp -q -s_srs \"+proj=geos +h=35786023.0 +a=6378137.0 +b=6356752.31414 +f=0.0033528106647475126 +lon_0=-75.0 +sweep=x +no_defs\" -t_srs EPSG:4326 -r near \"${{file%.nc}}_tmp1.nc\" \"${{file%.nc}}_tmp2.nc\"; done"
            ]
        
        else:
            toref = [
                "bash", "-c",
                f"for file in ./{dirname}/backup/bd02/*.nc; do cdo -f nc4 expr,'2ref=kappa0*Rad;' \"$file\" \"${{file%.nc}}_tmp1.nc\" && gdalwarp -q -s_srs \"+proj=geos +h=35786023.0 +a=6378137.0 +b=6356752.31414 +f=0.0033528106647475126 +lon_0=-75.0 +sweep=x +no_defs\" -t_srs EPSG:4326 -r near \"${{file%.nc}}_tmp1.nc\" \"${{file%.nc}}_tmp2.nc\"; done"
            ]
        
        subprocess.run(toref)
        
        files = glob.glob(f"./{dirname}/backup/{product}/*_tmp2.nc")
        files = sorted(files)
        for file in files:
            gettime = gettime + timedelta(minutes=5)
            newname = gettime.strftime("%Y%m%d_%H%M_tmp3")
            cdo.settaxis(gettime.strftime("%Y-%m-%d,%H:%M:00,5min"), input=f"{file}", options='-f nc4 -r', output=f"./{dirname}/backup/{product}/{newname}.nc")
        
        gtime -= timedelta(hours=1)
        time_str = gtime.strftime("%Y-%m-%d,%H:%M:00,5min")
        if (product == "bd11"): cdo.remapnn('./perm/mygrid', input=f"-settaxis,{time_str} -setunit,Kelvin -chname,Band1,{product} -mergetime ./{dirname}/backup/{product}/*_tmp3.nc", options='-b F32 -f nc4 -r', output=f"./{dirname}/backup/{product}.nc")
        else: cdo.remapnn('./perm/mygrid', input=f"-settaxis,{time_str} -setmisstoc,1 -setrtomiss,1,10 -setmisstoc,0 -setrtomiss,-1,0.01 -chname,Band1,{product} -mergetime ./{dirname}/backup/{product}/*_tmp3.nc", options="-b F32 -f nc4 -r", output=f"./{dirname}/backup/{product}.nc")
        
        remove = [f"rm ./{dirname}/backup/{product}/*_tmp?.nc"]
        subprocess.run(remove, shell=True)



if __name__ == "__main__":
    
    st = time.time()
    tout = 270                                                                # Set the timeout time for the processes (you may have to experiment with this based on your system)
    try: os.remove("./warnings.txt")
    except: pass
    try: os.remove("./timings.txt")
    except: pass
    try: os.remove("./instances.txt")
    except: pass
    os.environ["REMAP_EXTRAPOLATE"] = "off"
    cdo = Cdo()
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    # mrms, hrrr, rtma, goes
    delaytimes = [3, 55, 20, 5]                                               # Delaytimes for retriving data (meant to simulate realtime)
    gridtype = "lonlat"                                                       # Grid format (reccomended lat lon)
    xsize = 4500                                                              # Points in x direction
    ysize = 2500                                                              # Points in y direction
    xfirst = -116.1                                                           # Starting (westernmost) longitude
    xinc = 0.01                                                               # x increment
    yfirst = 25                                                               # Starting (southernmost) latitude
    yinc = 0.01                                                               # y increment
    # The default domain is the largest possible without having NaN values
    grid_specs = f"""gridtype = {gridtype}
    xsize    = {xsize}
    ysize    = {ysize}
    xfirst   = {xfirst}
    xinc     = {xinc}
    yfirst   = {yfirst}
    yinc     = {yinc}
    """
    with open("./perm/mygrid", "w") as file: file.write(grid_specs)
    
    stdate_gb = datetime(2023, 8, 8)                                          # Start date (inclusive) for retrieving data
    eddate_gb = datetime(2023, 8, 8)                                          # End date (inclusive) for retrieving data
    step_gb = timedelta(days=1)                                               # Timestep
    files_done = 0
    prev_time = datetime(2000, 1, 1, 0, 0)
    prev_dirname = "20000101_0000"
    
    for i in range((eddate_gb - stdate_gb).days +1):
        
        date_cr = stdate_gb + i * step_gb
        hour_cr = np.random.randint(0, 24)
        minute_cr = np.random.randint(0, 12) * 5
        datetime_cr = date_cr + timedelta(hours=hour_cr, minutes=minute_cr)
        if files_done > 0:
            duration = datetime_cr - prev_time
            duration_s = duration.total_seconds()
            mins_diff = divmod(duration_s, 60)[0]
            while mins_diff < 125:
                hour_cr = np.random.randint(0, 24)
                minute_cr = np.random.randint(0, 12) * 5
                datetime_cr = date_cr + timedelta(hours=hour_cr, minutes=minute_cr)
                duration = datetime_cr - prev_time
                duration_s = duration.total_seconds()
                mins_diff = divmod(duration_s, 60)[0]
        
        if utils.locate_data(datetime_cr, "VIL_00.50", "Reflectivity_-10C_00.50", delaytimes) == 1:
            lst = time.time()
            print("\n" + datetime_cr.strftime("%Y-%m-%d %H:%M") + " has been found\n")
            dirName = datetime_cr.strftime("%Y%m%d_%H%M")
            utils.create_dir(dirName)
            
            check = multiprocessing.Process(target=utils.process_data, args=(prev_dirname, True, ))
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
            if files_done > 0: check.start()
            
            if files_done > 0: check.join(tout)
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
            shutil.rmtree(f"./{dirName}/backup/") # Keep or remove backup netCDF files
            let = time.time()
            lti = round(let-lst, 3)
            timing = f"{dirName} done in {lti} seconds\n"
            with open("./timings.txt", "a") as file: file.write(timing)
            print("\n" + timing)
            prev_time = datetime_cr
            prev_dirname = dirName
            files_done += 1
            
        else: print("\n" + datetime_cr.strftime("%Y-%m-%d %H:%M") + " does not exist\n")
    
    if files_done > 0: utils.process_data(prev_dirname, True)
    et = time.time()
    ti = et - st
    print("\n" + f"Completed {files_done} files in ", ti, "seconds\n")
