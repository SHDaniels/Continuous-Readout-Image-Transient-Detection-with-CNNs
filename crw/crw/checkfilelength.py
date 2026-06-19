import pandas as pd
import glob
import numpy as np
from tqdm import tqdm
import crw_config as config
import sys
import os
import subprocess
import gc
import resource

#lightcurve_ztf_20221020297847_000449_zr_c16_x_0029_4_1010_0182.csv

import os
import subprocess
import pandas as pd
import glob
from pathlib import Path

def get_file_info_batch(directory, batch_size=10000, name=None):
    """
    Efficiently get file info for millions of files using find + wc
    """
    all_files = []
    if name is None:
        name = "*csv"
    print(directory+name)

    # Use find to get all CSV files (bypasses argument limit)
    cmd = ['find', directory, '-name', name, '-type', 'f']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    files = result.stdout.strip().split('\n')
    
    print(f"Found {len(files)} files")
    
    # Process in batches to avoid memory issues
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(files)//batch_size)+1}")
        
        # Get line counts with wc
        wc_cmd = ['wc', '-l'] + batch
        wc_result = subprocess.run(wc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                   universal_newlines=True)
        #lightcurve_ztf_20221020148032_000686_zg_c12_x_0001_3_0860_1800
        #print(wc_result.stdout.strip().split('\n'))
        # Parse wc output
        for line in wc_result.stdout.strip().split('\n'):
            if line and 'total' not in line:
                path = line.lstrip().strip().split(" ")[-1]
                filename = os.path.basename(path).split("_")
                datetime = filename[2]
                run = filename[3]
                thisfilter = filename[4]
                choosechip = filename[5].replace("c","")
                sequencenumber = filename[7]
                amp = filename[8]
                x = filename[9]
                y = filename[10]
                parts = line.split()
                if len(parts) == 2:
                    count, path = parts
                    all_files.append({
                        'datetime': datetime,
                        'run': "%06d"%int(run),
                        'thisfilter': thisfilter,
                        'choosechip': "%02d"%int(choosechip),
                        'sequencenumber': "%04d"%int(sequencenumber),
                        'amp': amp,
                        'x': x,
                        'y': y,
                        'filename': os.path.basename(path),
                        'lines': int(count)-1,
                    })
    
    return pd.DataFrame(all_files)

def getfilesizeintodf(path='/lustre/lrspec/ZTF_data/lightcurves/1020/', batch_size=10000, name=None):
    df = get_file_info_batch(path, batch_size=batch_size, name=name)
    df.to_csv(config.ROOT + 'lightcurvefiles_info.csv', index=False)


def checklength(filename, total_lines, verbose=False):

    runsdir = config.runsdir
    print(filename.split("/")[-1].replace(".csv","").replace(".npy","").split("_")[-4:])
    datetime, run, thisfilter, choosechip = filename.split("/")[-1].replace(".csv","").replace(".npy","").split("_")[-4:]

    pattern = f"{runsdir}/lightcurve_ztf_{datetime}_{run}_{thisfilter}_c{choosechip}*"
    if verbose: print("pattern ", pattern)

    files = glob.glob(pattern)
    batch_size = 100
    original_lines = 0

    for i in tqdm(range(0, len(files), batch_size)):
        batch = files[i:i+batch_size]

        # Use xargs to run wc once with many files                                                                              
        cmd = ['xargs', '-0', 'wc', '-l']
        input_data = '\0'.join(batch).encode()
        #print(input_data)                                                                                                      
        result = subprocess.run(cmd, input=input_data,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
        #print(result)                                                                                                          
        output = result.stdout.decode().strip().split('\n')
        for line in output:
            if 'total' not in line and line.strip():
                original_lines += int(line.split()[0]) - 1
        gc.collect()
    if total_lines != original_lines:
        print(filename, total_lines, original_lines, "something is wrong!")
    #print(resource.getrlimit(resource.RLIMIT_NOFILE))

def checklength_fromdf(filename, total_lines, df, verbose=False):
    
    runsdir = config.runsdir

    datetime, run, thisfilter, choosechip = filename.split("/")[-1].replace(".csv","").split("_")[-4:]
    
    if verbose: print(f"{datetime} {run} {thisfilter} {choosechip}")
    #pattern = f"lightcurve_ztf_{datetime}_{run}_{thisfilter}_c{choosechip}*"
    #print("pattern ", pattern)
    select1 = df["datetime"] == int(datetime)
    select2 = df["run"] == int(run)
    select3 = df["thisfilter"] == thisfilter
    select4 = df["choosechip"] == int(choosechip)
    #print(df[select1 & select2 & select3 ].loc[:,"choosechip"].unique())
    #print(df["datetime"].dtype)
    original_lines = df[select1 & select2 & select3 & select4]["lines"].sum()


    if total_lines != original_lines:
        print(filename, total_lines, original_lines, "something is wrong!")
    #print(resource.getrlimit(resource.RLIMIT_NOFILE))  

def checklength_crwdata(filename, nfiles, ntimestamps, df, verbose=False):
    runsdir = config.runsdir

    datetime, run, thisfilter, choosechip = filename.split("/")[-1].replace(".npy","").split("_")[-4:]
    
    if verbose: print(f"{datetime} {run} {thisfilter} {choosechip}")
    #pattern = f"lightcurve_ztf_{datetime}_{run}_{thisfilter}_c{choosechip}*"
    #print("pattern ", pattern)
    select1 = df["datetime"] == int(datetime)
    select2 = df["run"] == int(run)
    select3 = df["thisfilter"] == thisfilter
    select4 = df["choosechip"] == int(choosechip)
    #print(df[select1 & select2 & select3 & select4])
    original_lines = df[select1 & select2 & select3 & select4].shape[0]    
    original_lengths = df[select1 & select2 & select3 & select4]["lines"].unique()
    if nfiles != original_lines:
        print(filename, nfiles, original_lines, "something is wrong! - number of files")
    if ntimestamps != original_lengths:
        print(filename, ntimestamps, original_lengths, "something is wrong! - length")
    return

def checklength_crwpos(filename, nfiles, df, verbose=False):
    runsdir = config.runsdir

    datetime, run, thisfilter, choosechip = filename.split("/")[-1].replace(".npy","").split("_")[-4:]

    if verbose:
        print(f"{datetime} {run} {thisfilter} {choosechip}")
    #pattern = f"lightcurve_ztf_{datetime}_{run}_{thisfilter}_c{choosechip}*"                                                     
    #print("pattern ", pattern)                                                                                                   
    select1 = df["datetime"] == int(datetime)
    select2 = df["run"] == int(run)
    select3 = df["thisfilter"] == thisfilter
    select4 = df["choosechip"] == int(choosechip)
    #print(df[select1 & select2 & select3 ].loc[:,"choosechip"].unique())                                                         
    #print(df["datetime"].dtype)                                                                                                 
    original_lines = df[select1 & select2 & select3 & select4].shape[0]
    if nfiles != original_lines:
        print(filename, nfiles, original_lines, "something is wrong! - number of pos files")

    return

def checklength_crwfeatures(filename, nfiles, df, verbose=False):
    runsdir = config.runsdir

    datetime, run, thisfilter, choosechip = filename.split("/")[-1].replace(".npy","").split("_")[-4:]

    if verbose:
        print(f"{datetime} {run} {thisfilter} {choosechip}")
    #pattern = f"lightcurve_ztf_{datetime}_{run}_{thisfilter}_c{choosechip}*"                
    #print("pattern ", pattern)                                                              
    select1 = df["datetime"] == int(datetime)
    select2 = df["run"] == int(run)
    select3 = df["thisfilter"] == thisfilter
    select4 = df["choosechip"] == int(choosechip)
    #print(df[select1 & select2 & select3 ].loc[:,"choosechip"].unique())                    
    #print(df["datetime"].dtype)                                                             
    original_lines = df[select1 & select2 & select3 & select4].shape[0]
    if nfiles != original_lines:
        print(filename, nfiles, original_lines, "something is wrong! - number of features")

    return

def search_file_sets(datetime,run,thisfilter,choosechip, verbose=False):
    runsdir = config.ROOT
    choosechip = "%02d"%int(choosechip)
    #choosechip.replace("c","")
    #print(runsdir + f"/ztf*{datetime}*{run}*{thisfilter}*{choosechip}*.csv")
    thelist = glob.glob(runsdir + f"/ztf*{datetime}*{run}*{thisfilter}*{choosechip}*.csv")
    if not  len(thelist)== 1: 
        if len(thelist) < 1:
            print ("missing ztf*csv", thelist)
        else:
            print("too many files")
    else:
        if verbose:
            print("you have all the lightcurves! Yay!!")


    thelist = glob.glob(runsdir + f"/crwdata*{datetime}*{run}*{thisfilter}*{choosechip}*.npy")
    if not  len(thelist)== 1:
        if len(thelist) < 1:
            print ("missing crwdata", thelist)
        else:
            print("too many files")
    else: 
        if verbose:
            print("you have all the data numpys! Yay!!")

    thelist = glob.glob(runsdir + f"/crwpos*{datetime}*{run}*{thisfilter}*{choosechip}*.npy")
    if not  len(thelist)== 1:
        if len(thelist) < 1:
            print ("missing crwpos", thelist)
        else:
            print("too many files")
    else: 
        if verbose:
            print("you have all the position numpys! Yay!!")
    
    thelist = glob.glob(runsdir + f"/crwfeat*{datetime}*{run}*{thisfilter}*{choosechip}*.npy")
    if not  len(thelist)== 2: 
        if len(thelist) < 2:
            print ("missing crwfeature", thelist)
        else:
            print("too many files")
    else:         
        if verbose:
            print("you have all the feature files! Yay!!")
    

if __name__ == "__main__":
    if len(sys.argv)>1:
        assert len(sys.argv) == 2, "pass the filename to check"
        filename = sys.argv[1]
        f = open(filename, "r")
        total_lines = sum(1 for line in f) - 1

    else:
        path = "*"
        path = "*20221020138241_000686_zi_03*"
        df = pd.read_csv(config.ROOT + "/lightcurvefiles_info.csv")
        dfgroup = df.groupby(["datetime","run","thisfilter","choosechip"]).count().reset_index()

        #for every lightcurve chech 
        for i in range(len(dfgroup)):
            datetime,run,thisfilter,choosechip = dfgroup.iloc[i:i+1,:4].values[0]
            search_file_sets(datetime,run,thisfilter,choosechip)

        nlcvs = dfgroup.shape[0]

        filelist = glob.glob(config.ROOT + "/ztf"+path+"csv")
        if len(filelist) != nlcvs:
            print(f"WARNING!! missing lightcurve csv files ({len(filelist)} vs {nlcvs})")
        else: 
            print("you have all the files lightcurve!, checking each file length\n")        
        
        for filename in tqdm(filelist): 
            f = open(filename, "r")
            total_lines = sum(1 for line in f) - 1            
            #datetime,run,thisfilter,choosechip,sequencenumber,amp,x,y,filename,lines,size_bytes
            #checklength(filename, total_lines)
            checklength_fromdf(filename, total_lines, df)
        
        filelist = glob.glob(config.ROOT + "/crwdata"+path+"npy")
        if len(filelist) != nlcvs:
            print(f"WARNING!! missing lightcurve numpy files  ({len(filelist)} vs {nlcvs})")
        else: 
            print("you have all the crwdata files!, checking for file length")
        for filename in tqdm(filelist):
            ar = np.load(filename)
            nfiles, ntimestamps = ar.shape[:2]
            #checklength(filename, total_lines)                                              
            checklength_crwdata(filename, nfiles, ntimestamps, df)

        filelist = glob.glob(config.ROOT + "/crwpos"+path+"npy")
        if len(filelist) != nlcvs:
            print("WARNING!! missing position numpy files")
        else:
            print("you have all the files crwpos!, checking each file length")
        for filename in tqdm(filelist):
            ar = np.load(filename)
            nfiles, nfeats = ar.shape[:2]
            #checklength_(filename, nfiles)                                             
            checklength_crwpos(filename, nfiles, df)

        filelist = glob.glob(config.ROOT + "/crwfeat"+path+"npy")
        if len(filelist) != nlcvs*2:
            print(f"WARNING!! missing lightcurve numpy files  ({len(filelist)} vs {nlcvs})")
        else:
            print("you have all the feature files!, checking each file length")
        for filename in tqdm(filelist):
            ar = np.load(filename)
            nfiles, nfeats = ar.shape[:2]
            #checklength_(filename, nfiles)                                             
            checklength_crwfeatures(filename, nfiles, df)

        print(df.columns)
    
        
