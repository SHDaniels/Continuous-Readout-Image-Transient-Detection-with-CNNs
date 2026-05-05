from argparse import ArgumentParser

import numpy as np
import pylab as plt
import pandas as pd
from sklearn import preprocessing
import pywt
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.cm as cm
import crw_config as config
import sys

from utils import checkna, parallel_list_comprehension, diptst, lowpass_filter, binary_badlcvs

#from unidip import UniDip
#import unidip.dip as dip
C05trainingdata = False

def extractfeatures(choosechip=None, runsdir=None,
                    thisfilter=None, run=None,
                    datetime=None, extension=None,
                    verbose=None, diagnostic_plots=None):

    # If no arguments provided, try to load from config

    
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
  if diagnostic_plots is None:
    diagnostic_plots = config.DIAGNOSTICPLOTS
  
  filedetails, lcvarrayfile, chipfilename, featuresfile, predictionfile, rootdir = config.makenames(datetime, run, thisfilter, choosechip)
    
  NCl=  config.NCl
  ap_column = config.ap_column
  
  extension = config.extension
  building_trainingset = config.BUILDINGTRAININGSET
  wavelets = config.wavelets  
  wavelet = config.wavelet
  #'haar'  # Start with 'haar', then try 'db4'
  level = config.waveletlevel
  #5        # Choose an appropriate level (e.g., 5-8)    
  if verbose:
    print("opening", lcvarrayfile)
  Xsave = np.load(open(lcvarrayfile, "rb"), allow_pickle=True)
  X = Xsave[:,:,ap_column].copy()

  Nlc = X.shape[0]
  Nts = X.shape[1]
  if verbose:
      print(f"there are {Nlc} lightcurves and {Nts} time stamps in each")

  if diagnostic_plots:
      fig, axs = plt.subplots(6,5,figsize=(10,10))
      plt.title("full lightcurve")
      axs = axs.ravel()
      i = 0
      for j in np.random.choice(Nlc, 30, replace=False):
        axs[i].plot(X[j][:])
        axs[i].axis('off')
        i=i+1

      fig, axs = plt.subplots(6,5,figsize=(10,10))
      plt.title("10% lightcurve")        
      axs = axs.ravel()
      i = 0
      for j in np.random.choice(Nlc, 30, replace=False):
        axs[i].plot(X[j][:300])
        axs[i].axis('off')
        i=i+1

      fig, axs = plt.subplots(6,5,figsize=(10,10))
      plt.title("1% lightcurve")        
      axs = axs.ravel()
      i = 0
      for j in np.random.choice(Nlc, 30, replace=False):
        axs[i].plot(X[j][:30])
        axs[i].axis('off')
        i=i+1
      plt.show()


      fig, axs = plt.subplots(6,1,figsize=(10,10))
      plt.title("apertures")
      i = 0
      for j in np.random.choice(Nlc, 6, replace=False):
        axs[i].plot(Xsave[j,:,7] , alpha=0.5)
        axs[i].plot(Xsave[j,:,6] , alpha=0.5)
        axs[i].plot(Xsave[j,:,5] , alpha=0.5)
        axs[i].plot(Xsave[j,:,4], alpha=0.5)
        axs[i].axis('off')
        i+= 1
        
      plt.show()
      checkna(X)


      np.random.seed(362)

      fig, axs = plt.subplots(6,1,figsize=(10,10))
      plt.title("apertures")
      axs = axs.ravel()
      i = 0
      for j in np.random.choice(Nlc, 6, replace=False):
        axs[i].plot(Xsave[j,:,7][:300], alpha=0.5, label="flux_hap_5")

        axs[i].plot(Xsave[j,:,6][:300], alpha=0.5, label="flux_hap_4")
        axs[i].plot(Xsave[j,:,5][:300], alpha=0.5, label="flux_hap_3")
        axs[i].plot(Xsave[j,:,4][:300], alpha=0.5, label="flux_hap_2")

        axs[i].axis('off')
        i+= 1
        
      plt.legend()
      plt.show()


  if verbose:
    print(f"running Hartigan's test for all {Nlc} lvs")
    
  bimodality_score = parallel_list_comprehension(diptst, iter(X))
  #[diptst(x)[1] for x in tqdm(X)] #10 min for 800 lcvs
  #bimodality_score = [diptst(x) for x in tqdm(X)] #10 min for 800 lcvs     
  Xmean = X.mean(axis=1)
  Xstd = X.std(axis=1)
  pcs = [1, 5, 25, 50, 75, 95, 99]
  Xpc =  np.percentile(X, pcs, axis=1)

  features = np.vstack([Xmean, Xstd,  bimodality_score]).T

  labels = ["mean", "std", "Hartigan's"]
  labels = labels + ["snr"]
  labels = labels + ["1pc", "5pc"]
  labels = labels + ["25pc", "50pc", "75pc"]
  labels = labels + ["95pc", "99pc"]

  pd.DataFrame(data=labels).to_csv(f"{rootdir}crwlabels_.csv")
  print(f"saved labels in crwlabels_.csv")

  features = np.vstack([features.T, Xmean/Xstd[:,np.newaxis].T, Xpc]).T
  np.save(featuresfile, features)
  print(f"saved {features.shape} features in crwfeatures{filedetails}.npy")

  #features = np.load(open("crwfeatures{filedetails}.npy", "rb"))
  labelsdf = pd.read_csv(f"{rootdir}crwlabels_.csv", index_col=0)
  labels = [l for l in labelsdf["0"].values]
  if verbose:
    print(f"saving the following features")
    print(labels)
    
  features_scaled = preprocessing.scale(features, axis=0)
  features_scaled.std(axis=0)

  Xlp = lowpass_filter(X)

  # 2. The first element of coeffs is the Approximation at the deepest level
  approximation = np.array(
    [pywt.wavedec(x, wavelet, level=level)[-1] for x in Xlp])

  wmean = approximation.mean(axis=1)
  wmedian = np.median(approximation, axis=1)
  w95 = np.percentile(approximation, [95], axis=1)[0]

  print("adding wavelet features")

  
  features = np.vstack([features.T, wmean, wmedian, w95]).T
  labels = labels + ["wvlmean", "wvlmedian", "wvl95"]

  labelsdf = pd.read_csv(f"{rootdir}crwlabels_.csv", index_col=0)
  labelsdf = pd.concat([labelsdf,
                        pd.DataFrame(data=np.array(["wvlmean", "wvlmedian", "wvl95"]),
                                     index=np.arange(len(labelsdf),len(labelsdf)+3),
                                     columns=["0"])])

  label2num_dict = {l[0]:i for i,l in enumerate(labelsdf.values) if l[0] not in ["Hartigan's"] }
  label2num_dict["H"] = 2
  label2num_dict

  binarybad = binary_badlcvs(X)
  features = np.vstack([features.T, binarybad.T]).T
  labels = labels + ["digitize_middle", "digitize_bot", "digitize_top"]

  features_scaled = preprocessing.scale(features, axis=0)
  features_scaled.std(axis=0)
  
  fig, axs = plt.subplots(2,1, figsize=(6, 6))
  axs = axs.ravel()
  axs[0].boxplot(features)
  axs[0].set_xticks(range(1, features.shape[1]+1), labels)
  axs[0].set_xlabel('Feature Index')
  axs[0].set_ylabel('Original Value')
  axs[0].tick_params(axis='x', labelrotation=90)
  plt.grid(True)
  
  axs[1].boxplot(features_scaled)
  axs[1].set_xticks(range(1, features_scaled.shape[1]+1), labels)
  axs[1].set_xlabel('Feature Index')
  axs[1].set_ylabel('Scaled Value')
  axs[1].tick_params(axis='x', labelrotation=90)
  plt.grid(True)
  plt.savefig(f"{rootdir}/plots/features{filedetails}.pdf")

  if diagnostic_plots:
    plt.show()

    
    for i in range(X.shape[0]):
      if  bimodality_score[i] < 0.2:
        plt.figure(figsize=(15,1))
        plt.plot(X[i], color='k', label=f"{ bimodality_score[i]:.2f}")
        plt.title(i)
        plt.legend()
        plt.show()

    for i in range(X.shape[0]):
      if 0.2 <  bimodality_score[i] < 0.4:
        plt.figure(figsize=(15,1))
        plt.plot(X[i], color='k', label=f"{ bimodality_score[i]:.2f}")
        plt.title(i)
        plt.legend()
      plt.show()

  
  featuresfile = featuresfile.replace('crwfeatures', 'crwfeatures_wwvt')
  print(f"saved {features.shape} features in {featuresfile}")
  np.save(featuresfile, features)
  pd.DataFrame(data=labels).to_csv(f"{rootdir}crwlabels_wwvt_.csv")
  if verbose:
    print(f"saving the following features")
    print(labels)
    
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
            
    
    print(f"extractfeatures({choosechip}, {runsdir}, {thisfilter}, {run}, {datetime}, {silent})")
    extractfeatures(choosechip, runsdir, thisfilter, run, datetime, silent)
