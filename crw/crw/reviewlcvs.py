import crw_config as config
import numpy as np
import plotly.express as px
import ipywidgets as widgets

from IPython.display import display

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

class QuickReview:
    def __init__(self, series):
        self.series = series
        self.reject_lightcurve = [0] * len(series)
        self.i = 0
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.2)
        
        # Create buttons
        ax_prev = plt.axes([0.1, 0.05, 0.1, 0.075])
        ax_keep = plt.axes([0.3, 0.05, 0.1, 0.075])
        ax_reject = plt.axes([0.5, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.7, 0.05, 0.1, 0.075])
        
        self.btn_prev = Button(ax_prev, '←')
        self.btn_keep = Button(ax_keep, 'Keep')
        self.btn_reject = Button(ax_reject, 'Reject')
        self.btn_next = Button(ax_next, '→')
        
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
        self.ax.plot(self.series[self.i])
        self.ax.set_title(f'Series {self.i+1}/{len(self.series)} - Status: {self.reject_lightcurve[self.i]}')
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
    
    def update_plot(self):
        self.ax.clear()
        self.ax.plot(self.series[self.i])
        self.ax.set_title(f'Series {self.i+1}/{len(self.series)} - Status: {self.reject_lightcurve[self.i]}')
        plt.draw()


def reviewlcvs(choosechip=None, runsdir=None,
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

      wavelet = config.wavelet
      #'haar'  # Start with 'haar', then try 'db4'
      level = config.waveletlevel
      #5        # Choose an appropriate level (e.g., 5-8)
      
      lcvarrayfile = config.lcvarrayfile
      chipfilename = config.chipfilename

  else:
      lcvsavedfile = f"crwdata_{datetime}_{run}_{thisfilter}_{choosechip}.npy"
      chipfilename = f"ztfcrm_{datetime}_{run}_{thisfilter}_{choosechip}.csv"


      
  Xsave = np.load(open(lcvarrayfile, "rb"))
  X = Xsave[:,:,ap_column].copy()

  reviewer = QuickReview(X)
  
  print("Final rejections:", reviewer.reject_lightcurve)
  np.save(f"crwrejected{filedetails}.npy", np.array(reviewer.reject_lightcurve))
  print(f"rejections saved in crwrejected{filedetails}.npy")

if __name__ == '__main__':
    reviewlcvs()
