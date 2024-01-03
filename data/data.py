#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:14:33 2023

@author: atyagi
"""

# This script retrieves data to train the model, each piece of data is for the past hour from the time specified in the filename of the data
# Each piece of data is around 25-30 gb, so make sure you have enough space, and also I think this script requires about 20-30 gb of ram (not sure though)
# Please read all of the comments before you run this script

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
        else:
            return []
    
    @staticmethod
    def elev_time(dirname, etime):
        etime -= timedelta(hours=2)
        for i in range(2):
            etime += timedelta(hours=1)
            etime_str = etime.strftime("%Y-%m-%d,%H:%M:00,5min")
            cdo.settaxis(f"{etime_str}", input="./perm/elev.nc", options="-f nc4 -r", output=f"./{dirname}/backup/elev/elev{i}.nc")
        etime -= timedelta(hours=1)
        etime_str = etime.strftime("%Y-%m-%d,%H:%M:00,5min")
        cdo.inttime(f"{etime_str}", input=f"-mergetime ./{dirname}/backup/elev/elev0.nc ./{dirname}/backup/elev/elev1.nc", options="-b F32 -f nc4 -r", output=f"./{dirname}/backup/elev.nc")
    
    @staticmethod
    def merge_ins(dirname):
        ds1 = xr.open_dataset(f"./{dirname}/backup/bd02.nc", chunks={'time': 1, 'lat': 2500, 'lon': 6000})
        ds2 = xr.open_dataset(f"./{dirname}/backup/bd11.nc", chunks={'time': 1, 'lat': 2500, 'lon': 6000})
        ds3 = xr.open_dataset(f"./{dirname}/backup/rtma.nc", chunks={'time': 1, 'lat': 2500, 'lon': 6000})
        ds4 = xr.open_dataset(f"./{dirname}/backup/hrrr.nc", chunks={'time': 1, 'lat': 2500, 'lon': 6000})
        ds5 = xr.open_dataset(f"./{dirname}/backup/elev.nc", chunks={'time': 1, 'lat': 2500, 'lon': 6000})
        merged_ds = xr.merge([ds1, ds2, ds3, ds4, ds5])
        merged_ds.to_zarr(f"./{dirname}/inputs.zarr", mode='w', consolidated=True)



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
        cdo.inttime(f"{itime},5min", input=f"-settaxis,{itime},2min ./{dirname}/backup/{product_short}/{product_short}tmp.nc", options='-f nc4 -r', output=f"./{dirname}/backup/{product_short}/{product_short}tmpp.nc")
        
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
        
    def merge_mrms(dirname):
        ds1 = xr.open_dataset(f"./{dirname}/backup/vil.nc", chunks={'time': 1, 'lat': 2500, 'lon': 6000})
        ds2 = xr.open_dataset(f"./{dirname}/backup/rf-10.nc", chunks={'time': 1, 'lat': 2500, 'lon': 6000})
        merged_ds = xr.merge([ds1, ds2])
        merged_ds.to_zarr(f"./{dirname}/target.zarr", mode='w', consolidated=True)
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
    
    def hrrr(dirname, htime):
        
        hrtime = htime - timedelta(hours=1, minutes=hrrr.delaytime)
        hrtime = hrtime.replace(minute=0)
        DATES = pd.date_range(
            start=hrtime.strftime("%Y-%m-%d %H:00"),
            periods=2,
            freq="1H",
        )
        
        fxx=range(0,1)
        data = FastHerbie(DATES, model="hrrr", product="prs", fxx=fxx, max_threads=16,)
        data.download(searchString="(0-0 m below ground)|((TMP|DPT|VVEL|UGRD|VGRD|ABSV):(([5-8][0,5]0)|(10[0,1][0,3])|(9[0,2,5,7][0,5])))|(CAPE)|(CIN)|(FRICV)|(MSLMA)|(RELV)|([U\|V]GRD:[1,8]0 m)|(SNOWC)|(ICEC)|(LAND)|((TMP|DPT):2 m)|(PWAT)|(HPBL)|(HGT:equilibrium level)|(HGT:level of adiabatic condensation from sfc)|(HGT:no_level)|(HGT:0C isotherm)|(LFTX)|(VGTYP)", max_threads=16, save_dir = f"./{dirname}/backup/")
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
        cdo.remapnn("./perm/mygrid", input=f"-settaxis,{stime} -inttime,{ltime} -mergetime {f1} {f2}", options='-b F32 -P 16 -f nc4 -r', output=f"./{dirname}/backup/hrrr.nc")
        
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
    
    delaytime = 5
    
    def goes(dirname, product, bandnum, gtime):
        
        gettime = gtime - timedelta(minutes=goes.delaytime)
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
            cdo.settaxis(gettime.strftime("%Y-%m-%d,%H:%M:00,5min"), input=file, options='-f nc4 -r', output=f"./{dirname}/backup/{product}/{newname}.nc")
        
        gtime -= timedelta(hours=1)
        time_str = gtime.strftime("%Y-%m-%d,%H:%M:00,5min")
        if (product == "bd11"):
            cdo.remapnn('./perm/mygrid', input=f"-settaxis,{time_str} -setunit,Kelvin -chname,Band1,{product} -mergetime ./{dirname}/backup/{product}/*_tmp3.nc", options='-b F32 -f nc4 -r', output=f"./{dirname}/backup/{product}.nc")
        else:
            cdo.remapnn('./perm/mygrid', input=f"-settaxis,{time_str} -chname,Band1,{product} -mergetime ./{dirname}/backup/{product}/*_tmp3.nc", options='-b F32 -f nc4 -r', output=f"./{dirname}/backup/{product}.nc")
        
        remove = [f"rm ./{dirname}/backup/{product}/*_tmp?.nc"]
        subprocess.run(remove, shell=True)
        


if __name__ == "__main__":
    st = time.time()
    stdate_gb = datetime(2023, 8, 8)    # Start date for getting data (inclusive)
    eddate_gb = datetime(2023, 8, 8)    # End date for getting data (inclusive)
    step_gb = timedelta(days=1)
    
    for i in range((eddate_gb - stdate_gb).days +1):
        cdo.cleanTempDir()
        
        date_cr = stdate_gb + i * step_gb
        hour_cr = np.random.randint(2, 24)
        minute_cr = np.random.randint(0, 12) * 5
        datetime_cr = date_cr + timedelta(hours=hour_cr, minutes=minute_cr)
        
        print(datetime_cr.strftime("%Y-%m-%d %H:%M"))
        dirName = datetime_cr.strftime("%Y%m%d_%H%M")
        utils.create_dir(dirName)

        tfm_mvil = multiprocessing.Process(target=mrms.mrms, args=(dirName, "VIL_00.50", "vil", datetime_cr, ))
        tfm_rf10 = multiprocessing.Process(target=mrms.mrms, args=(dirName, "Reflectivity_-10C_00.50", "rf-10", datetime_cr, ))
        tfm_rtma = multiprocessing.Process(target=rtma.rtma, args=(dirName, datetime_cr, ))
        tfm_hrrr = multiprocessing.Process(target=hrrr.hrrr, args=(dirName, datetime_cr, ))
        tfm_bd02 = multiprocessing.Process(target=goes.goes, args=(dirName, "bd02", 2, datetime_cr, ))
        tfm_bd11 = multiprocessing.Process(target=goes.goes, args=(dirName, "bd11", 11, datetime_cr, ))
        tfm_elev = multiprocessing.Process(target=utils.elev_time, args=(dirName, datetime_cr, ))
        mrge_mrms = multiprocessing.Process(target=mrms.merge_mrms, args=(dirName, ))
        mrge_file = multiprocessing.Process(target=utils.merge_ins, args=(dirName, ))
        
        tfm_hrrr.start()
        tfm_bd02.start()
        tfm_bd11.start()
        tfm_rtma.start()
        tfm_elev.start()
        tfm_rf10.start()
        tfm_mvil.start()
        
        tfm_mvil.join()
        tfm_rf10.join()
        mrge_mrms.start()
        mrge_mrms.join()
        
        tfm_elev.join()
        tfm_rtma.join()
        tfm_bd11.join()
        tfm_bd02.join()
        tfm_hrrr.join()
        
        mrge_file.start()
        mrge_file.join()
        shutil.rmtree(f"./{dirName}/backup/")
        et = time.time()
        ti = et - st
        print()
        print(f"{dirName} done in ", ti, "seconds")
        
        
    
    et = time.time()
    ti = et - st
    print()
    print("TOTAL: ", ti, "seconds")
