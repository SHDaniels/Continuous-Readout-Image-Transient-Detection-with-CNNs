import crw_config as config
import pandas as pd
import numpy as np
import sys

def getprediction(lcv):
    pieces = lcv.split(".")[0].split("_")
    datetime = pieces[2]
    run = pieces[3]
    thisfilter= pieces[4]
    choosechip = pieces[5].replace("c","")
    seq = int(pieces[7])
    x = int(pieces[9])
    y = int(pieces[10])
    a = int(pieces[8])

    try:
      pos = np.load(config.ROOT + "/" + f"crwpos_{datetime}_{run}_{thisfilter}_{choosechip}.npy") 
    except FileNotFoundError:
      print("missing position!")
      print(f"{lcv},0")
      return

    filedetails, lcvarrayfile, chipfilename, featuresfile, predictionfile, rootdir = config.makenames(datetime, run, thisfilter, choosechip)
    predictionsfile = featuresfile.replace("crwfeatures_", "crwprediction_")

    select = np.where((pos[:,0] == x) * (pos[:,1] == y) * (pos[:,3] == a))[0]
    
    predictions = np.load(predictionfile)
    assert predictions.shape[0] == pos.shape[0], "positions as rf predictions must have the same number of enreies"

    try:
      select[seq]
    except IndexError:
      print("missing sequence!")
      print(f"{lcv},0") 
      return
    try:
      predictions[select[seq]]
    except IndexError:
      print("missing prediction!")
      print(f"{lcv},0")     
      return

    print(f"{lcv},{predictions[select[seq]]}")


if __name__ == "__main__":
    alllightcurves = pd.read_csv("alllightcurves", header=None)
    #print(alllightcurves.head())
    for i,lcv in alllightcurves.iterrows():
        #if i>10:
        #    break
        #print(lcv[0])
        getprediction(lcv[0])
