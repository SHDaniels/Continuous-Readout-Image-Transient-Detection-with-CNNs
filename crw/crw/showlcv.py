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

from utils import checkna, parallel_list_comprehension, diptst, lowpass_filter, loaddata




def plotlcv(n):

  X, Xpos, labels, filedetails, choosechip, runsdir, thisfilter, run, datetime, extension, verbose = loaddata()
  assert X not None, "the data file to load was not found"
    
  NCl=  config.NCl
  ap_column = config.ap_column
  
  verbose = config.VERBOSE
  diagnostic_plots = config.DIAGNOSTICPLOTS
  building_trainingset = config.BUILDINGTRAININGSET
  
  wavelet = config.wavelet
  #'haar'  # Start with 'haar', then try 'db4'
  level = config.waveletlevel
  #5        # Choose an appropriate level (e.g., 5-8)
  filedetails, lcvarrayfile, chipfilename, featuresfile, predictionfile, rootdir = config.makenames(datetime, run, thisfilter, choosechip)
      
  Xsave = np.load(open(lcvarrayfile, "rb"))
  X = Xsave[:,:,ap_column].copy()
  plt.plot(X[n])
  plt.title(n)
  plt.show()


  
if __name__ == "__main__":
  n = int(sys.argv[1])
  print(n)
  plotlcv(n)
