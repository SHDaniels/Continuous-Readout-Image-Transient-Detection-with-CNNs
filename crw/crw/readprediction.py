import crw_config as config
import pandas as pd
import numpy as np
import sys
from utils import getfromlcvname

def getprediction(lcv):
    datetime, run, thisfilter, choosechip, seq, x, y, a = getfromlcvname(lcv)

    try:
      pos = np.load(config.ROOT + "/" + f"crwpos_{datetime}_{run}_{thisfilter}_{choosechip}.npy") 
    except FileNotFoundError:
      print("missing position!")
      print(f"{lcv},0")
      return

    filedetails, lcvarrayfile, chipfilename, featuresfile, predictionfile, rootdir = config.makenames(datetime, run, thisfilter, choosechip)
    predictionsfile = featuresfile.replace("crwfeatures_", "crwprediction_")
    predictions = np.load(predictionfile)
    assert predictions.shape[0] == pos.shape[0], "positions as rf predictions must have the same number of enreies"

    #select the elements of the array corresponding to this star
    select = np.where((pos[:,0] == x) * (pos[:,1] == y) * (pos[:,3] == a))[0] 
    
    #fing the right run sequence
    try:
      seqindx = pos[select][:,2] == seq
    except IndexError:
      print("missing sequence!")
      print(f"{x} {y} {a} {seq}, {np.where((pos[:,0] == x) * (pos[:,1] == y) * (pos[:,3] == a))[0]}")
      print(f"{lcv},0") 
      return

    assert (pos[select][seqindx] == np.array([x,y,seq,a])).all()
    try:
      predictions[select][seqindx]
    except IndexError:
      print("missing prediction!")
      print(f"{lcv},0")     
      return

    print(f"{lcv},{predictions[select][seqindx][0]}")


if __name__ == "__main__":
    alllightcurves = pd.read_csv("alllightcurves", header=None)
    #print(alllightcurves.head())
    for i,lcv in alllightcurves.iterrows():
        #if i>10:
        #    break
        #print(lcv[0])
        getprediction(lcv[0])
