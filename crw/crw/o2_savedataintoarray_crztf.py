
import numpy as np
import pandas as pd
import pandas as pd
from tqdm import tqdm
import glob
import crw_config as config

def savedataintoarray(choosechip = None, runsdir = None,
                      thisfilter = None, run = None, datetime = None,
                      extension = None,
                      verbose = None):

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

    filedetails, lcvarrayfile, chipfilename, featuresfile, predictionfile, rootdir = config.makenames(datetime, run, thisfilter, choosechip)



    #sometime we cannot read the csv file in pandas cause its too long
    #then save the file into two chunks with head and tail and read then concat
    if not chipfilename in glob.glob(chipfilename):
      df1 =  pd.read_csv(chipfilename.replace(".csv", "_1.csv"), index_col=None)
      df2 =  pd.read_csv(chipfilename.replace(".csv", "_2.csv"), index_col=None)
      df1.shape, df2.shape
      df = pd.concat([df1, df2])
    else:
      df =  pd.read_csv(chipfilename, index_col=None)
    print(f"shape of data file {df.shape}")
    
    print(f"number of lightcurves in {datetime} "
          f"{run} {thisfilter} {choosechip}: {df.groupby(['x','y','sector']).count()}")
      
    xs = df["x"].unique()
    ys = df["y"].unique()
    sec = df["sector"].unique()
    Xpossave = []
    Xsave = []#np.zeros((len(xs), len(ys), len(sec), 3080, 12))
    for i,x in tqdm(enumerate(xs)):
      for j,y in enumerate((df[df["x"]==x]["y"]).unique()):
        #print(y)
        for k,s in enumerate(df[(df["x"]==x) & (df["x"]==x)]["sector"].unique()):
          _ = df[(df["x"]==x) & (df["y"]==y) & (df["sector"]==s)]
          Xpossave.append(np.array([x,y]))
          Xsave.append(_)

    Xsave = np.array(Xsave)
    np.save(lcvarrayfile, Xsave)
    Xpossave = np.array(Xpossave)
    np.save(lcvarrayfile.replace("crwdata", "crwpos"), Xpossave)    
    print(f"lightcurves saved in {lcvarrayfile}")

if __name__ == "__main__":
  savedataintoarray()
