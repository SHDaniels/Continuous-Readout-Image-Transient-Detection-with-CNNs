import numpy as npb
import os
#i20221020138241, 000686, zi, c01)
#20221020141806, 000686, zi, c01
#ztfcrm_20221020138241_000686_zi_01.csv
#20221020141806 
#686 zi 02
ROOT = "/lustre/lrspec/ZTF_data/outputs/1020/"
LCVSPATH = "/lustre/lrspec/ZTF_data/lightcurves/1020/"
RFPATH = "/lustre/lrspec/ZTF_data/random_forest_training/"
thisfilter = "zi"
run = "000686"
datetime = "20221020138241"
#_000686_zi_03"20221020141806" #"20221020138241" #"20221020141806" #"20221020138241" #"20221020148032" #"20221020148032"                                                                
choosechip = "03" #"05"      

ap_column = 4 #aperture is e.g. 4 -> 2x2+1 5 -> 3x2+1

#runsdir = "../selection_same_files_new_tracking"
#runsdir = "../selection"
runsdir = "/lustre/lrspec/ZTF_data/lightcurves/1020"
extension = ".csv"

NCl = 2
wavelets = True
wavelet = 'haar'
waveletlevel = 5
rfthreshold = 0.75#99


VERBOSE = True
DIAGNOSTICPLOTS = False
BUILDINGTRAININGSET = False

def makenames(datetime, run, thisfilter, choosechip):
  root = ROOT
  if not os.path.isdir(root):
    os.system("mkdir " + root)
  if not os.path.isdir(root + "/plots"):
    os.system("mkdir " + root + "/plots")
  if not os.path.isdir("../models"):
    os.system("mkdir ../models")

  filedetails = f"_{datetime}_{run}_{thisfilter}_{choosechip}"
  lcvarrayfile = f"{root}crwdata{filedetails}.npy"
  chipfilename = f"{root}ztfcrm{filedetails}.csv"
  featuresfile = f"{root}crwfeatures{filedetails}.npy"
  predictionfile = f"{root}crwprediction{filedetails}.npy"
  return filedetails, lcvarrayfile, chipfilename, featuresfile, predictionfile, root

filedetails, lcvarrayfile, chipfilename, featuresfile, predictionfile, root = makenames(datetime, run, thisfilter, choosechip)

    
