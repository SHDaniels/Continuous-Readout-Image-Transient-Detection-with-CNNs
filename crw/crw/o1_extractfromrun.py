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
from utils import *



def getchiplcv(choosechip = None, runsdir = None,
               thisfilter = None, run = None,
               datetime = None, extension = None,
               verbose = None, df_chips=None):
    
    if choosechip is None:
        choosechip = config.choosechip
    if runsdir is None:
        runsdir = config.runsdir
    if thisfilter is None:
        thisfilter = config.thisfilter
    if run is None:
        run = config.run
    if datetime is None:
        datetime = config.datetime
    if verbose is None:
        verbose = config.VERBOSE
    if extension is None:
        extension = config.extension

    print("dt", datetime, "r", run, "f", thisfilter, "c", choosechip)

    filedetails, lcvarrayfile, chipfilename, featuresfile, predictionfile, rootdir = config.makenames(datetime, run, 
                                                                                                      thisfilter, 
                                                                                                      choosechip)
    
    #print(filedetails, lcvarrayfile, chipfilename, featuresfile, predictionfile, rootdir)

    if os.path.isfile(chipfilename):
        print("File exists")
        return 
    else:
        print("Creating the file")

    if df_chips is None:
        pattern = f"{runsdir}/*{extension}"
    
        print(f"looking for lightcurves with pattern {pattern}")
        lcvs = glob.glob(pattern)

        if verbose:
            print(f"found {len(lcvs)} lightcurve files")

        dates = [l.split("/")[-1].split('_')[2] for l in lcvs]
        runs = [l.split("/")[-1].split('_')[3] for l in lcvs]
        filters = [l.split("/")[-1].split('_')[4] for l in lcvs]
        chip = [l.split("/")[-1].split('_')[5] for l in lcvs]

        df_chips = pd.DataFrame(columns=["name", "date", "run", "filter", "chip"],
                            data = np.array([lcvs, dates, runs, filters, chip]).T)


    pattern = f"{runsdir}/lightcurve_ztf_{datetime}_{run}_{thisfilter}_c{choosechip}*"
    if verbose:
        print(pattern)
    mychip = glob.glob(pattern)
    print(f"found {len(mychip)} files for chip {choosechip}")

    print("takes a couple of minutes")
    print(len(mychip))
    if len(mychip)>0:
        dfs = []  # Store DataFrames in a list
        for i,lcv in tqdm(enumerate(mychip)):
            _ = pd.read_csv(lcv, index_col=None)
            #if verbose:
            #    print(_)
            #sys.exit()
            #print(_.shape)
            #print(lcv.split("_")[7])
            _["sector"]=lcv.split("_")[-4]
            _["x"]=lcv.split("_")[-2]
            _["y"]=lcv.split("_")[-1].strip(".csv")
            dfs.append(_)  # Just add to list (no copying)

        # Single concatenation at the end
        df = pd.concat(dfs, ignore_index=True)

        df.to_csv(f"{chipfilename}", index=None)
        print(f"csv fifle printed to {chipfilename}")


if __name__ == "__main__":
    getchiplcv()
