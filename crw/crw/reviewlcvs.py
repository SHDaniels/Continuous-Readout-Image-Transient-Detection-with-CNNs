import matplotlib
matplotlib.use('WebAgg')
matplotlib.rcParams['webagg.open_in_browser'] = False

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

import crw_config as config
import numpy as np
import plotly.express as px
#import ipywidgets as widgets

#from IPython.display import display
import sys
import os
import pandas as pd

from utils import hpf

class QuickReview:
    def __init__(self, series, lcvs):
        self.series = series
        self.reject_lightcurve = [0] * len(series)
        self.i = 0
        self.lcvs = lcvs

        """
        select = ["2979" in l for l in self.lcvs]
        self.series = self.series[select]
        self.reject_lightcurve = np.array(self.reject_lightcurve)[select]
        self.i = 0
        self.lcvs = self.lcvs[select]
        """

        self.fig, self.ax = plt.subplots(2,1,figsize=(10, 12))
        plt.subplots_adjust(bottom=0.2)
        print("done init")
        # Create buttons
        ax_prev = plt.axes([0.1, 0.05, 0.1, 0.075])
        ax_keep = plt.axes([0.3, 0.05, 0.1, 0.075])
        ax_reject = plt.axes([0.5, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.7, 0.05, 0.1, 0.075])
        
        self.btn_prev = Button(ax_prev, '<-')
        self.btn_keep = Button(ax_keep, 'Keep')
        self.btn_reject = Button(ax_reject, 'Reject')
        self.btn_next = Button(ax_next, '->')
        
        self.btn_prev.on_clicked(lambda x: self.move(-1))
        self.btn_keep.on_clicked(lambda x: self.decide(0))
        self.btn_reject.on_clicked(lambda x: self.decide(1))
        self.btn_next.on_clicked(lambda x: self.move(1))
        
        self.update_plot()
        plt.show()
    
    def move(self, step):
        self.i = (self.i + step) % len(self.series)
        self.update_plot()
    
    def decide(self, val):
        self.reject_lightcurve[self.i] = val
        self.move(1)
    
    def update_plot(self):
        
        self.ax[0].clear()
        y = self.series[self.i]
        self.ax[0].plot(y, alpha=0.5)
        y2 = hpf(np.arange(len(y)), y)
        self.ax[0].plot(y-y2+np.mean(y), alpha=0.8, color='r')
        self.ax[0].set_title(f"Series {self.i+1}/{len(self.series)} -  Status: {self.reject_lightcurve[self.i]} {self.lcvs[self.i]}")
        self.ax[1].clear()
        fft = np.fft.rfft(y) 
        power_spectrum = np.abs(fft)**2  
        freqs = np.fft.rfftfreq(len(y))
        self.ax[1].plot(freqs, power_spectrum, alpha=0.5)
        fft = np.fft.rfft(y-y2)  
        power_spectrum = np.abs(fft)**2  
        freqs = np.fft.rfftfreq(len(y))
        self.ax[1].plot(freqs, power_spectrum, alpha=0.8)
        
        self.ax[1].set_yscale('log')
        self.ax[1].set_xlim(0.01, 0.2) #1/100 pixels to 1/5 pixels scales
        self.ax[1].set_xticks([0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]) #1/100 pixels to 1/5 pixels scales
        self.ax[1].set_xticklabels(["1/100", "1/40", "1/20","1/13","1/10","1/6.67", "1/5"])

        plt.draw()

    def move(self, step):
        new_i = self.i + step
        if new_i >= len(self.series):
            print("Review complete! Closing window...")
            print("Final rejections:", self.reject_lightcurve)
            plt.close()
            self.fig_open = False
            return
        elif new_i < 0:
            new_i = 0
        
        self.i = new_i
        self.update_plot()
    
    def decide(self, val):
        self.reject_lightcurve[self.i] = val
        self.move(1)
    



def reviewlabels(filename=None,
                     verbose=None, diagnostic_plots=None, savemodel=False):

  lcvsfile = filename
  tagfile = filename.replace(".npy", f"_reviewed.npy")
  path = config.RFPATH

  NCl =  config.NCl
  ap_column = config.ap_column

  extension = config.extension
  building_trainingset = config.BUILDINGTRAININGSET
  wavelets = config.wavelets
  wavelet = config.wavelet
  #'haar'  # Start with 'haar', then try 'db4'                                     
  level = config.waveletlevel
  #5        # Choose an appropriate level (e.g., 5-8)                              

  rfthreshold = config.rfthreshold
  Nlc = 0
  features = []
  labels = []

  lcvslist = np.load(path + "/" + lcvsfile, allow_pickle=True)
  tags = np.load(path + "/" + tagfile)
  tagselect = np.zeros(len(tags), bool) * False
  rootdir = config.ROOT

  labelsfile  = f"{rootdir}crwlabels_.csv"
  print(f"reading labels from {labelsfile}")
  labelsdf_here = pd.read_csv(labelsfile, index_col=0)
  labels = np.array([labelsdf_here.iloc[i]["0"] for i in range(labelsdf_here.shape\
[0])])
  label2num_dict_here = {l[0]:i for i,l in enumerate(labelsdf_here.values)
                         if l[0] not in ["Hartigan's"]}
  label2num_dict_here["H"] = 2
  Nfeatures = len(labels)

  for i,lcv in enumerate(lcvslist):

    print(f"\n\n{lcv}\n\n")
    pieces = lcv.split(".")[0].split("_")
    datetime = pieces[2]
    run = pieces[3]
    thisfilter= pieces[4]
    choosechip = pieces[5].replace("c","")
    seq = int(pieces[7])

    x = int(pieces[9])
    y = int(pieces[10])
    print(choosechip, thisfilter, run,datetime, x, y)
    #choosechip, run, thisfilter, datetime = lcv.split("_")                        
    #lightcurve_ztf_20221020141806_000686_zi_c14_x_0003_4_0708_0764.csv'           
    if i == 250:
        plt.show()
        plt.close('all')
    try:

      pos = np.load(config.ROOT + "/" + f"crwpos_{datetime}_{run}_{thisfilter}_{choosechip}.npy")
    except FileNotFoundError:
      continue

    filedetails, lcvarrayfile, chipfilename, featuresfile, predictionfile, rootdir = config.makenames(datetime, run, thisfilter, choosechip)
    print(f"reading features from {featuresfile}")
    print(f"reading chipfilename {chipfilename}")
    features_here = np.load(open(featuresfile, "rb"))

    select = np.where((pos[:,0] == x) * (pos[:,1] == y))[0]
    print(len(select))
    try:
      print(select[seq])
    except IndexError:
      continue
    tagselect[i] = True
    features_here = features_here[select[seq]:select[seq]+1]
    assert len(features_here) == 1, "select did not work"
    Nlc += len(features_here)


    message = f"this code expects {Nfeatures} with wavelets set to {wavelets},"
    message += "got {np.array(features).shape[1]}"

    assert features_here.shape[1] == Nfeatures, message

    features.append(features_here)
    color = 'r' if tags[i] == 1 else 'k'
    pd.read_csv(config.LCVSPATH + "/" + lcvslist[i])[["flux_hap_2"]].plot(color=color)
    plt.title(f"{lcvslist[i]}\n{i}, label {tags[i]}")
    
  plt.show()





def reviewlcvs(filename=None, pedix=None,
               choosechip=None, runsdir=None,
               thisfilter=None, run=None,
               datetime=None, extension=None, verbose=None):

  print(filename, pedix)
  if pedix is None:
      pedix = "reviewed"
  else:
      pedix = "reviewed" + pedix
  if filename:
      outfile = filename.replace(".npy", f"_{pedix}.npy")
      print(f"labels will be saved in {outfile}")
      path = config.LCVSPATH
      ap_column = config.ap_column 
  elif all(arg is None for arg in [choosechip, runsdir,
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

      wavelet = config.wavelet
      #'haar'  # Start with 'haar', then try 'db4'
      level = config.waveletlevel
      #5        # Choose an appropriate level (e.g., 5-8)
      
      lcvarrayfile = config.lcvarrayfile
      chipfilename = config.chipfilename
      outfile = f"{config.ROOT}/crwrejected{filedetails}.npy"
  else:
      lcvsavedfile = f"crwdata_{datetime}_{run}_{thisfilter}_{choosechip}.npy"
      chipfilename = f"ztfcrm_{datetime}_{run}_{thisfilter}_{choosechip}.csv"
      outfile = f"{config.ROOT}/crwrejected{filedetails}.npy"

  if not filename:
      Xsave = np.load(open(lcvarrayfile, "rb"))
      print(lcvarrayfile)
      print(Xsave)
      X = Xsave[:,:,ap_column].copy()
  else:
      print(filename)
      lcvlist = np.load(filename)#open(filename),'rb')
      for i,lcv in enumerate(lcvlist):
          x = pd.read_csv(path+lcv)
          if i>0:
              Xsave=np.dstack([Xsave,x.values])
          else:
              Xsave = x.values

      Xsave = Xsave.transpose(2, 0, 1)
      X = Xsave[:,:,ap_column].copy()


  reviewer = QuickReview(X, lcvlist)#[:10])
  
  print("Final rejections:", reviewer.reject_lightcurve)

  if os.path.exists(outfile):
    response = input(f"{outfile} already exists. Overwrite? (y/n): ").lower()
    if response != 'y':
        print("Save cancelled.")
        exit()


  np.save(outfile, np.array(reviewer.reject_lightcurve))
  print(f"rejections saved in {outfile}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if len(sys.argv)>2:
            if len(sys.argv)>3:
                reviewlabels(sys.argv[1], sys.argv[2])
            else:
                reviewlcvs(sys.argv[1], sys.argv[2])
        else:
            reviewlcvs(sys.argv[1])
    else:
        reviewlcvs()
