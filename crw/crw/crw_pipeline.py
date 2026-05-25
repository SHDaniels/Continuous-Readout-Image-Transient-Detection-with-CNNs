import crw_config as config
from o1_extractfromrun import getchiplcv as getchiplcv
from o2_savedataintoarray_crztf import savedataintoarray as savedataintoarray
from o3_save_simple_features_crztf import extractfeatures as extractfeatures
from o4_randomforest_simple_features_crztf import rffeatures_predict as rffeatures_predict
from argparse import ArgumentParser
import numpy as np
from utils import plotalllcs, loaddata, savehpflcs, binary_badlcvs, plotallfeatures


def launchpipeline(choosechip=None, runsdir=None, thisfilter=None,
                   run=None, datetime=None, silent=None):
    #read in configs
    
    X, Xpos, labels, filedetails, choosechip, runsdir, thisfilter, run, \
        datetime, extension, verbose  = loaddata(choosechip, runsdir, thisfilter, run,
                                                 datetime, silent)
    if X is None or Xpos is None:
        print("\n\n##################################")
        print("Step 1: save into a single csv")
        getchiplcv(choosechip, runsdir, thisfilter, run, datetime, extension, verbose)
        print("\n\n##################################")
        print("Step 2: save into a array")
        savedataintoarray(choosechip, runsdir, thisfilter, run, datetime, extension, verbose)
    
        X, Xpos, labels, filedetails, choosechip, runsdir, thisfilter, run, \
        datetime, extension, verbose  = loaddata(choosechip, runsdir, thisfilter, run,
                                                 datetime, silent)
    print("\n\n##################################")
    print("Step 3: extract features")
    print(f"there are {X.shape[0]} lightcurves to analyze on chip {choosechip}")

    if not labels is None:
        assert X.shape[0] == len(labels), "the labels and data are not the same length"
    extractfeatures(choosechip, runsdir, thisfilter, run, datetime,
                    extension, verbose)
    
    print("done feature extraction")
    return

    print("\n\n##################################")    
    print("Step 4: label lightcurves")
    filedetails, lcvarrayfile, chipfilename, featuresfile, predictionfile, rootdir = config.makenames(datetime, run, thisfilter, choosechip)
    labels = rffeatures_predict(choosechip, runsdir, thisfilter, run, datetime,
                       extension, verbose)
    _ = binary_badlcvs(X)
    binarylabels = ((_[:,0] < 150) & (_[:,1:].min(axis=1) > 20)).astype(int) * 2
    print(np.unique(binarylabels), np.unique(labels))
    labels = labels + binarylabels
    print(np.unique(labels))
    labels = np.max(np.dstack([labels, binarylabels]).squeeze(), axis=1).astype(int)
    features = np.load(open(featuresfile.replace("crwfeatures","crwfeatures_wwvt"), "rb"))
    plotallfeatures(features, labels=labels, filedetails=None)
    plotalllcs(X, labels, filedetails=filedetails)
    hpfX = savehpflcs(X=X, choosechip=choosechip, order=3, window=101, diagnostic_plots=True)    
    

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("-c", "--chip", dest="choosechip", default=None,
                    help="chip number(2 digits)")
    parser.add_argument("-d", "--runsdir", dest="runsdir", default=None,
                    help="chip number(2 digits)")
    parser.add_argument("-f", "--filter", dest="filter", default=None,
                    help="chip number(2 digits)")
    parser.add_argument("-r", "--run", dest="run", default=None,
                    help="chip number(2 digits)")
    parser.add_argument("-t", "--datetime", dest="datetime", default=None,
                    help="chip number(2 digits)")
    parser.add_argument("-q", "--quiet",
                    action="store_false", dest="silent", default=True,
                    help="don't print status messages to stdout")

    args = parser.parse_args()
    silent = args.silent
    choosechip =  args.choosechip
    runsdir = args.runsdir
    thisfilter = args.filter
    run = args.run
    datetime = args.datetime
            
    
    print(f"launchpipeline({choosechip}, {runsdir}, {thisfilter}, {run}, {datetime}, {silent})")
    launchpipeline(choosechip, runsdir, thisfilter, run, datetime, silent)
