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

def hpf(x, y, plot=False):
    from scipy.ndimage import filters
    from scipy.signal.windows import gaussian
    b = gaussian(1000, 20)
    #ga = filtfilt(b/b.sum(), [1.0], y)
    ga = filters.convolve1d(y, b/b.sum())
    if plot:
        plt.plot(x, ga, alpha=0.9, c="g", label="g")
    return ga


class QuickReview:
    def __init__(self, series, lcvs):
        self.series = series
        self.reject_lightcurve = [0] * len(series)
        self.i = 0
        self.lcvs = lcvs

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
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
        self.ax.clear()
        y = self.series[self.i]
        self.ax.plot(y, alpha=0.5)
        y2 = hpf(np.arange(len(y)), y)
        self.ax.plot(y-y2+np.mean(y), alpha=0.5, color='r')
        self.ax.set_title(f"Series {self.i+1}/{len(self.series)} -  Status: {self.reject_lightcurve[self.i]} {self.lcvs[self.i]}")
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
      ap_column = config.ap_column + 2
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
            reviewlcvs(sys.argv[1], sys.argv[2])
        else:
            reviewlcvs(sys.argv[1])
    else:
        reviewlcvs()
