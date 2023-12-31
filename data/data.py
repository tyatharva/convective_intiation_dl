#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 01:30:00 UTC 2023

@author: Atharva Tyagi
"""

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

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
tempPath = './tmp/'
cdo = Cdo(tempdir=tempPath)
cdo.cleanTempDir()


class utils():
    
    @staticmethod
    def create_dir(folder_name):
        current_directory = os.getcwd()
        main_folder_path = os.path.join(current_directory, folder_name)
        os.makedirs(main_folder_path, exist_ok=True)
        backup_folder_path = os.path.join(main_folder_path, 'backup')
        os.makedirs(backup_folder_path, exist_ok=True)
        subfolders = ['bd02', 'bd11', 'rtma', 'hrrr', 'vil', 'rf-10']
        for subfolder in subfolders:
            subfolder_path = os.path.join(backup_folder_path, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)
    
    @staticmethod
    def list_files_s3(bucket, prefix):
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' in response:
            files = [obj['Key'] for obj in response['Contents']]
            return files
        else:
            return []



class mrms():
    
    delaytime = 3
    
    def mrms(dirname, product_long, product_short, mtime):
        
        modtime = mtime - timedelta(minutes=mrms.delaytime)
        if modtime.minute % 2 == 0:
            gettime = modtime
        else:
            gettime = modtime - timedelta(minutes=1)
        
        x = 0
        gettim2 = gettime + timedelta(minutes=2)
        while x < 30:
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
        while i < 90:
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
        settime_str = gettime.strftime("%Y-%m-%d,%H:%M:00,2min")
        
        settaxis = [
            "cdo",
            "-f", "nc4", "-r",
            f"settaxis,{settime_str}",
            f"./{dirname}/backup/{product_short}/{product_short}tmp.nc",
            f"./{dirname}/backup/{product_short}/{product_short}tmpp.nc"
        ]
        subprocess.run(settaxis)
        
        remap = [
            "cdo",
            "-f", "nc4",
            "remapnn,./perm/mygrid",
            f"./{dirname}/backup/{product_short}/{product_short}tmpp.nc",
            f"./{dirname}/backup/{product_short}.nc"
        ]
        subprocess.run(remap)
        
        remove = [f"rm ./{dirname}/backup/{product_short}/*.nc"]
        subprocess.run(remove, shell=True)
        
    def merge_mrms(dirname):
        ds1 = xr.open_dataset(f"./{dirname}/backup/vil.nc", chunks={'time': 1, 'lat': 2500, 'lon': 6000})
        ds2 = xr.open_dataset(f"./{dirname}/backup/rf-10.nc", chunks={'time': 1, 'lat': 2500, 'lon': 6000})
        merged_ds = xr.merge([ds1, ds2])
        merged_ds.to_zarr(f"./{dirname}/mrms.zarr", mode='w', consolidated=True)
        subprocess.run(f"rm ./{dirname}/backup/vil.nc", shell=True)
        subprocess.run(f"rm ./{dirname}/backup/rf-10.nc", shell=True)



class hrrr():
    
    delaytime = 55
    
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
    
    def hrrr(dirname, hrtime):
        DATES = pd.date_range(
            start=hrtime.strftime("%Y-%m-%d %H:00"),
            periods=3,
            freq="1H",
        )
        fxx=range(0,1)
        data = FastHerbie(DATES, model="hrrr", product="prs", fxx=fxx, max_threads=16,)
        data.download(searchString="(ABSV)|(4LFTX)|(CFNSF)|(CLWMR)|(CAPE)|(CIN)|(DPT)|(DLWRF)|(DSWRF)|(EFHL)|(FRICV)|(HGT)|(ICEC)|(LAND)|(LHTFL)|(MSTAV)|(MSLMA)|(HPBL)|(CNWAT)|(PWAT)|(PRES)|(RELV)|(SHTFL)|(SNOWC)|(LFTX)|(SFCR)|(TMP)|(UGRD)|(VGRD)|(VGTYP)|(DZDT)|(VVEL)|(0-0 m below ground)", max_threads=16, save_dir = f"./{dirname}/backup/")
        hrrr.mfilerdir_hrrr(f"./{dirname}/backup/hrrr/")
        
        tonc = [
            "bash", "-c",
            f"for file in ./{dirname}/backup/hrrr/*.grib2; do wgrib2 \"$file\" -nc4 -netcdf \"${{file%.grib2}}.nc\"; done"
        ]
        subprocess.run(tonc)
        
        path = f"./{dirname}/backup/hrrr/"
        pattern = f"{path}*.nc"
        files = glob.glob(pattern)
        files = sorted(files)
        for file in files:
            base_filename = os.path.splitext(os.path.basename(file))[0]
            output_filename = f"{base_filename}_remap.nc"
            cdo.remapnn('./perm/mygrid', input=f'-delname,HGT_surface {file}', options='-P 16 -f nc4 -r', output=os.path.join(path, output_filename))
            
        z = 0
        files = glob.glob(f"./{dirname}/backup/hrrr/*remap.nc")
        files = sorted(files)
        for file in files:
            ds = xr.open_dataset(file, chunks={'time': 1, 'lat': 2500, 'lon': 6000})
            if (z == 0):
                ds.to_zarr(f"./{dirname}/hrrr.zarr", consolidated=True)
                z+=1
            else:
                ds.to_zarr(f"./{dirname}/hrrr.zarr", consolidated=True, append_dim='time')
        
        remove = [f"rm ./{dirname}/backup/hrrr/*.nc"]
        subprocess.run(remove, shell=True)



class rtma():
    
    delaytime = 20
    
    def rtma(dirname, rtime):
        
        gettime = rtime
        gettime -= timedelta(minutes=rtma.delaytime)
        remain = gettime.minute % 15
        if remain != 0:
            gettime -= timedelta(minutes=remain)
        
        i = 0
        while i< 12:
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
        
        cdo.remapnn('./perm/mygrid', input='-chname,DPT_2maboveground,rtma_DPT_2maboveground,GUST_10maboveground,rtma_GUST_10maboveground,PRES_surface,rtma_PRES_surface,TMP_2maboveground,rtma_TMP_2maboveground,UGRD_10maboveground,rtma_UGRD_10maboveground,VGRD_10maboveground,rtma_VGRD_10maboveground -delname,HGT_surface,CEIL_cloudceiling,TCDC_entireatmosphere_consideredasasinglelayer_,VIS_surface,WDIR_10maboveground,WIND_10maboveground,SPFH_2maboveground -mergetime '+f"./{dirname}/backup/rtma/*.nc", options='-f nc4', output=f"./{dirname}/backup/rtma.nc")

        remove = [f"rm ./{dirname}/backup/rtma/*.nc"]
        subprocess.run(remove, shell=True)
        
        ds = xr.open_dataset(f"./{dirname}/backup/rtma.nc", chunks={'time': 1, 'lat': 2500, 'lon': 6000})
        ds.to_zarr(f"./{dirname}/rtma.zarr", mode='w', consolidated=True)
        subprocess.run(f"rm ./{dirname}/backup/rtma.nc", shell=True)



class goes():
    
    delaytime = 5
    
    def goes(dirname, product, bandnum, goestime):
        
        gettime = goestime
        i = 0
        while i < 36:
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
                f"for file in ./{dirname}/backup/bd11/*.nc; do cdo -f nc4 expr,'bright=(planck_fk2/(log((planck_fk1/Rad)+1))-planck_bc1)/planck_bc2;' \"$file\" \"${{file%.nc}}_tmp1.nc\" && gdalwarp -s_srs \"+proj=geos +h=35786023.0 +a=6378137.0 +b=6356752.31414 +f=0.0033528106647475126 +lon_0=-75.0 +sweep=x +no_defs\" -t_srs EPSG:4326 -r near \"${{file%.nc}}_tmp1.nc\" \"${{file%.nc}}_tmp2.nc\"; done"
            ]
        
        else:
            toref = [
                "bash", "-c",
                f"for file in ./{dirname}/backup/bd02/*.nc; do cdo -f nc4 expr,'2ref=kappa0*Rad;' \"$file\" \"${{file%.nc}}_tmp1.nc\" && gdalwarp -s_srs \"+proj=geos +h=35786023.0 +a=6378137.0 +b=6356752.31414 +f=0.0033528106647475126 +lon_0=-75.0 +sweep=x +no_defs\" -t_srs EPSG:4326 -r near \"${{file%.nc}}_tmp1.nc\" \"${{file%.nc}}_tmp2.nc\"; done"
            ]
        
        subprocess.run(toref)
        
        files = glob.glob(f"./{dirname}/backup/{product}/*_tmp2.nc")
        files = sorted(files)
        for file in files:
            gettime = gettime + timedelta(minutes=5)
            newname = gettime.strftime("%Y%m%d_%H%M_tmp3")
            cdo.settaxis(gettime.strftime("%Y-%m-%d,%H:%M:00,5min"), input=file, options='-f nc4 -r', output=f"./{dirname}/backup/{product}/{newname}.nc")
            
        if (product == "bd11"):
            cdo.remapnn('./perm/mygrid', input=f"-setunit,Kelvin -chname,Band1,{product} -mergetime "+f"./{dirname}/backup/{product}/*_tmp3.nc", options='-f nc4', output=f"./{dirname}/backup/{product}.nc")
        else:
            cdo.remapnn('./perm/mygrid', input=f"-chname,Band1,{product} -mergetime "+f"./{dirname}/backup/{product}/*_tmp3.nc", options='-f nc4', output=f"./{dirname}/backup/{product}.nc") # -z zip_2
        
        remove = [f"rm ./{dirname}/backup/{product}/*_tmp?.nc"]
        subprocess.run(remove, shell=True)
        
    def merge_goes(dirname):
        ds1 = xr.open_dataset(f"./{dirname}/backup/bd02.nc", chunks={'time': 1, 'lat': 2500, 'lon': 6000})
        ds2 = xr.open_dataset(f"./{dirname}/backup/bd11.nc", chunks={'time': 1, 'lat': 2500, 'lon': 6000})
        merged_ds = xr.merge([ds1, ds2])
        merged_ds.to_zarr(f"./{dirname}/goes.zarr", mode='w', consolidated=True)
        subprocess.run(f"rm ./{dirname}/backup/bd02.nc", shell=True)
        subprocess.run(f"rm ./{dirname}/backup/bd11.nc", shell=True)



if __name__ == "__main__":
    st = time.time()
    stdate_gb = datetime(2021, 10, 1)
    eddate_gb = datetime(2021, 10, 1)
    step_gb = timedelta(days=1)
    
    for i in range((eddate_gb - stdate_gb).days +1):
        cdo.cleanTempDir()
        
        date_cr = stdate_gb + i * step_gb
        hour_cr = np.random.randint(2, 24)
        minute_cr = np.random.randint(0, 60)
        datetime_cr = date_cr + timedelta(hours=hour_cr, minutes=minute_cr)
        hrrr_time = datetime_cr - timedelta(hours=2, minutes=hrrr.delaytime)
        goes_time = datetime_cr - timedelta(minutes=goes.delaytime)
        
        print(datetime_cr.strftime("%Y-%m-%d %H:%M"))
        dirName = datetime_cr.strftime("%Y%m%d_%H%M")
        utils.create_dir(dirName)

        tfm_mvil = multiprocessing.Process(target=mrms.mrms, args=(dirName, "VIL_00.50", "vil", datetime_cr, ))
        tfm_rf10 = multiprocessing.Process(target=mrms.mrms, args=(dirName, "Reflectivity_-10C_00.50", "rf-10", datetime_cr, ))
        tfm_rtma = multiprocessing.Process(target=rtma.rtma, args=(dirName, datetime_cr, ))
        tfm_hrrr = multiprocessing.Process(target=hrrr.hrrr, args=(dirName, hrrr_time, ))
        tfm_bd02 = multiprocessing.Process(target=goes.goes, args=(dirName, "bd02", 2, goes_time, ))
        tfm_bd11 = multiprocessing.Process(target=goes.goes, args=(dirName, "bd11", 11, goes_time, ))
        mrge_mrms = multiprocessing.Process(target=mrms.merge_mrms, args=(dirName, ))
        mrge_goes = multiprocessing.Process(target=goes.merge_goes, args=(dirName, ))
        
        tfm_hrrr.start()
        tfm_mvil.start()
        tfm_rf10.start()
        tfm_rtma.start()
        tfm_bd02.start()
        tfm_bd11.start()
        
        tfm_mvil.join()
        tfm_rf10.join()
        mrge_mrms.start()
        mrge_mrms.join()
        
        tfm_bd02.join()
        tfm_bd11.join()
        mrge_goes.start()
        mrge_goes.join()
        
        tfm_rtma.join()
        tfm_hrrr.join()
        shutil.rmtree(f"./{dirName}/backup/")

    
    et = time.time()
    ti = et - st
    print()
    print("done in ", ti, "seconds")
