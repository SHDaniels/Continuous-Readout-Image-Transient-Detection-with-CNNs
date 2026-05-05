

import numpy as np
import pylab as plt
import pandas as pd
import os
from sklearn import preprocessing
import pywt
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.cm as cm
import crw_config as config
import sys
from showlcv import plotlcv
from utils import checkna, parallel_list_comprehension, diptst, lowpass_filter
generalpath = "../outputs"
feature_threshold = None
feature_threshold = (("std", 1400, 0, 'gtr'), ("mean", 0, 0, "gtr")) 


def testrejections(choosechip=None, runsdir=None,
                   thisfilter=None, run=None,
                   datetime=None, extension=None, verbose=None):

  if all(arg is None for arg in [choosechip, runsdir,
                                   thisfilter, run, datetime, extension]):
      NCl=  config.NCl
      ap_column = config.ap_column

      verbose = config.VERBOSE
      diagnostic_plots = config.DIAGNOSTICPLOTS
      building_trainingset = config.BUILDINGTRAININGSET
      
      choosechip = config.choosechip
      runsdir = config.runsdir
      thisfilter = config.thisfilter
      run = config.run
      datetime = config.datetime
      extension = config.extension
      filedetails = config.filedetails
      tags = config.tags
      
      wavelet = config.wavelet
      #'haar'  # Start with 'haar', then try 'db4'
      level = config.waveletlevel
      #5        # Choose an appropriate level (e.g., 5-8)
      
      lcvarrayfile = config.lcvarrayfile
      chipfilename = config.chipfilename
      featuresfile = config.featuresfile

  else:
      lcvsavedfile = f"crwdata_{datetime}_{run}_{thisfilter}_{choosechip}.npy"
      chipfilename = f"ztfcrm_{datetime}_{run}_{thisfilter}_{choosechip}.csv"
      featuresfile = f"crwfeatures_{datetime}_{run}_{thisfilter}_{choosechip}.npy"

  labelsfile = generalpath + f"crwrejected_{datetime}_{run}_{thisfilter}_{choosechip}.npy" 
  if os.path.isfile(labelsfile):
    print("replace hard coded tags with stored label file")
    tags[f"_{datetime}_{run}_{thisfilter}_{choosechip}"] = np.load(open(labelsfile, "rb"))
  
  Xsave = np.load(open(lcvarrayfile, "rb"))
  X = Xsave[:,:,ap_column].copy()

  Nlc = X.shape[0]
  Nts = X.shape[1]

  featuresfile = featuresfile.replace('crwfeatures', 'crwfeatures_wwvt')
  print(f"reading features from {featuresfile}")
  features = np.load(open(featuresfile, "rb"))
  labelsdf = pd.read_csv(generalpath + "/" + "crwlabels_wwvt_.csv", index_col=0)
  labels = [labelsdf.iloc[i]["0"] for i in range(labelsdf.shape[0])]
  
  if  filedetails in tags:
          tag = tags[filedetails]
  else:
          tag = np.zeros(Nlc) 
          
  select = None
  seletind = []
  for i in range(features.shape[1]):
        color = tag

        plt.scatter(x=features[:,0], y=features[:,i], c=color, cmap='bwr', alpha=0.7)
        plt.title("if colors, R=1, B=0")
        
        if feature_threshold:
                
                ft = feature_threshold[0]
                print(ft)
                featurecolumn = int(labelsdf.index[labelsdf["0"] == ft[0]][0])
                if ft[3] == 'gtr':
                        select1 = ((features[:,featurecolumn] > ft[1]) &
                                  (tag == ft[2]))
                elif ft[3] == 'less':
                        select1 = ((features[:,featurecolumn] < ft[1]) &
                                  (tag == ft[2]))
                ft = feature_threshold[1]
                featurecolumn = int(labelsdf.index[labelsdf["0"] == ft[0]][0])
                if ft[3] == 'gtr':
                        select2 = ((features[:,featurecolumn] > ft[1]) &
                                  (tag == ft[2]))
                elif ft[3] == 'less':
                        select2 = ((features[:,featurecolumn] < ft[1]) &
                                  (tag == ft[2]))
                print(select1, select2)
                select = select1 & select2
                
                        
                #print((select & (tag[filedetails] == 0)).any())
                featureselect = features[select]
                #print(select,features[select])
                #print(featureselect[:,featurecolumn] > feature_threshold[1])
                #sys.exit()
                plt.scatter(x=featureselect[:,0], y=featureselect[:,i],
                            c=color[select], cmap='bwr', alpha=0.7, marker='x')
                       
                
        plt.xlabel(labels[0])
        plt.ylabel(labels[i])                   
        plt.show()
  if not select is None:
          selectind = [i for i in range(Nlc) if select[i]]
  
  if filedetails in tags.keys():
        if not select is None:
                print(selectind)
                for n in selectind:
                        plotlcv(n)
        else:
                for n in [i for i in range(Nlc) if tag[i]]:
                        print(n)
                        plotlcv(n)
                        

                
if __name__ == "__main__":
  testrejections()
