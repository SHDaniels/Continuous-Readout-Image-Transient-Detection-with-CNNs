import pandas as pd
import glob
import numpy as np
from tqdm import tqdm
import crw_config as config
import sys




def getchiplcv(choosechip = None, runsdir = None,
               thisfilter = None, run = None,
               datetime = None, extension = None,
               verbose = None):
    
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

    print(datetime, run, thisfilter, choosechip)

    filedetails, lcvarrayfile, chipfilename, featuresfile, predictionfile, rootdir = config.makenames(datetime, run, thisfilter, choosechip)
    print(filedetails, lcvarrayfile, chipfilename, featuresfile, predictionfile, rootdir)

    print("\n\n\n\n\n")
    return
    

    pattern = f"{runsdir}/*{extension}"
    
    print(f"looking for lightcurves with pattern {pattern}")
    lcvs = glob.glob(pattern)

    if verbose:
        print(f"found {len(lcvs)} lightcurve files")

    dates = [l.split("/")[2].split('_')[2] for l in lcvs]
    runs = [l.split("/")[2].split('_')[3] for l in lcvs]
    filters = [l.split("/")[2].split('_')[4] for l in lcvs]
    chip = [l.split("/")[2].split('_')[5] for l in lcvs]
    
    df_chips = pd.DataFrame(columns=["name", "date", "run", "filter", "chip"],
                            data = np.array([lcvs, dates, runs, filters, chip]).T)



    print("LIGHTCURVES PER CHIP")
    print(df_chips.groupby(["date", "run", "filter", "chip"]).count())
    pattern = f"{runsdir}/lightcurve_ztf_{datetime}_{run}_{thisfilter}_c{choosechip}*"
    if verbose:
        print(pattern)
    mychip = glob.glob(pattern)
    print(f"found {len(mychip)} files for chip {choosechip}")

    print("takes a couple of minutes")

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
        
        #print(_.shape)
        if i == 0:
            df = _
        else:
            df = pd.concat([df, _])
            #print(df.shape)

    df.to_csv(f"{chipfilename}", index=None)
    print(f"csv fifle printed to {chipfilename}")


if __name__ == "__main__":
    getchiplcv()
