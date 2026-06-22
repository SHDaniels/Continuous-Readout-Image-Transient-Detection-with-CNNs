
import numpy as np
import pandas as pd
import pandas as pd
from tqdm import tqdm
import glob
import crw_config as config


### takes dataframes aggregated by common datetime, field (referred to as run) CCD (referred to as chip) and filter and saves them as numpy arays. Resulting arrays are shape Nstarstreaks Ntimestamps in image Nrows. Rows are star_id, x position, y position, timestamp, 4 flux columns at different apertures, actual x position, image in the sequence, amp, x, y => Nlc X 3080 X 13

### position file is saved as numpyarray containing x y s a => Nlc X 4

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


    if verbose:
        print(chipfilename)
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

    dfg = df.groupby(['x','y','sequence','amp']).count()
    print(f"number of lightcurves in {datetime} : {len(dfg)}")
    if verbose:
        print(f"{run} {thisfilter} {choosechip}: {dfg.iloc[:,:1]}")

    xs = df["x"].unique()
    ys = df["y"].unique()
    sec = df["sequence"].unique()
    amp = df["amp"].unique()

    Xpossave = []
    Xsave = []

    #np.zeros((len(xs), len(ys), len(sec), 3080, 12))
    #print(list(enumerate(xs)))

    a = df[(df["x"]==xs[0]) & (df["y"]==ys[0])]["amp"].unique()
    if len(a) > 1:
        print(f" WATNING: more than one amplifier at the same coordinates?? {df[(df['x']==xs[0]) & (df['y']==ys[0])]['amp']}")

    a = a[0]
    s = df[(df["x"]==xs[0]) & (df["y"]==ys[0])  & (df["amp"]==a)]["sequence"].unique()[0]

    shapehere = df[(df["x"]==xs[0]) & (df["y"]==ys[0]) & (df["amp"]==a) & (df["sequence"]==s)].shape
    print("xs", len(xs))
    for i,x in tqdm(enumerate(xs)):
        selectx = df["x"]==x
        for j,y in enumerate((df[selectx]["y"]).unique()):
            selectxy = (df["x"]==x) & (df["y"]==y)
            amps = df[(df["x"]==x) & (df["y"]==y)]["amp"].unique()

            for a in amps:
                selectxya = selectxy & (df["amp"]==a)

                for k,s in enumerate(np.sort(df[selectxya]["sequence"].unique())):
                    _ = df[selectxya & (df["sequence"]==s)]
                    if _.shape == shapehere:
                        #print(x,y,a,s _.shape)
                        Xpossave.append(np.array([x,y,s,a]))
                        Xsave.append(_)
                    else: 
                        print(f"warning: shape is wrong: expected {shapehere} got {_.shape}")
                    
    print("here")
    import sys
    print(f"Number of items in Xsave: {len(Xsave)}")
    print(f"Estimated memory: {sys.getsizeof(Xsave)} {sys.getsizeof(Xsave) / 1024**3:.1f} GB")
    Xsave = np.array(Xsave)
    print("here1")
    np.save(lcvarrayfile, Xsave)
    print("lcvarrayfile saved")    
    Xpossave = np.array(Xpossave)
    print("position file saved")
    np.save(lcvarrayfile.replace("crwdata", "crwpos"), Xpossave)    
    print(f"lightcurves saved in {lcvarrayfile}")

if __name__ == "__main__":
  savedataintoarray()
