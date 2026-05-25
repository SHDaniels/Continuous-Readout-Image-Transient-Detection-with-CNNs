import numpy as np
import os
#i20221020138241, 000686, zi, c01)
#20221020141806, 000686, zi, c01
#ztfcrm_20221020138241_000686_zi_01.csv

ROOT = "/lustre/lrspec/ZTF_data/outputs/1020/"
LCVSPATH = "/lustre/lrspec/ZTF_data/lightcurves/1020/"
thisfilter = "zi"
run = "000686"
datetime = "20221020138241" #"20221020141806" #"20221020138241" #"20221020148032" #"20221020148032"                                                                
choosechip = "02" #"05"      


ap_column = 2 + 2

#runsdir = "../selection_same_files_new_tracking"
#runsdir = "../selection"
runsdir = "/lustre/lrspec/ZTF_data/lightcurves/1020"
extension = ".csv"

NCl = 2
wavelets = True
wavelet = 'haar'
waveletlevel = 5
rfthreshold = 0.5


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

tags = {}

if 1:
  #all(arg[0] == arg[1] for arg in zip(
  #  [thisfilter, run, datetime, choosechip, ap_column],
   #                         ["zg",  "000686", "20221020148032", "06", 5])):
    thischip = "06"
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"] = np.zeros(1290)
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1261] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1267] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1271] = 1    
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1276] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1278] = 1        
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1286] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][302] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][303] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][300] = 1
    for i in [661, 672, 676, 678, 680, 686, 688, 1136, 1137, 1138,
              689, 1266, 1280, 1282, 1284, 1289]:
      tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][i] = 1
    
    thischip = "05"
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"]  = np.load(open("crwrejected_20221020148032_000686_zg_05.npy", "rb"))
    """
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"] = np.zeros(810)
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][211:214] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][217] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][218] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][220] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][221] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][223] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][480] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][481:483] = 1    
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][484:486] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][487:493] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][494:496] = 1        
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][497:508] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][508:510] = 1	
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][570:600] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][212] = 1
    #tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][225] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][229] = 1
    #tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][235] = tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][237] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][264] = tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][239] = 1
    """

    thischip = "04"
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"] = np.zeros(1980)
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][450] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][451] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][452] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][453] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][454] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][455] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][456] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][457] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][458] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][459] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][460] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][461] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][462] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][463] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][464] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][465] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][466] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][467] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][468] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][469] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][470] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][471] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][472] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][473] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][474] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][475] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][476] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][477] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][478] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][479] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][508] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][510] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][511] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][512] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][513] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][514] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][515] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][516] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][517] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][518] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][519] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][520] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][521] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][522] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][523] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][524] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][525] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][526] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][527] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][528] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][529] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][530] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][531] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][532] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][533] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][534] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][535] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][536] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][537] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][538] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][539] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][578] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][930] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][931] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][932] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][933] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][934] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][935] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][936] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][937] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][938] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][939] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][940] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][941] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][942] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][943] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][944] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][945] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][946] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][947] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][948] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][949] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][950] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][951] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][952] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][953] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][954] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][955] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][956] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][957] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][958] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][959] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][977] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][979] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][980] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][989] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1052] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1053] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1054] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1055] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1057] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1058] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1059] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1061] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1062] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1063] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1064] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1065] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1066] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1067] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1068] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1069] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1070] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1071] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1072] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1073] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1074] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1075] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1076] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1077] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1078] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1079] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1345] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1560] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1561] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1562] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1568] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1569] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1570] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1571] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1572] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1585] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1714] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1724] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1725] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1736] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1770] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1771] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1772] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1773] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1774] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1775] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1776] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1777] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1778] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1779] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1780] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1781] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1782] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1783] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1784] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1785] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1786] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1787] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1788] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1789] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1790] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1792] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1793] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1794] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1795] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1796] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1797] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1798] = 1
    tags[f"_{datetime}_{run}_{thisfilter}_{thischip}"][1799] = 1

    
