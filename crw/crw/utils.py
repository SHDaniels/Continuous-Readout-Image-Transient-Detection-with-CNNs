import numpy as np
from tqdm import tqdm
import crw_config as config
import sys
import os
import subprocess
import gc
import resource 

from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

import matplotlib
matplotlib.use('WebAgg')
matplotlib.rcParams['webagg.open_in_browser'] = False
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import plotly.express as px

import pywt
from scipy.signal import savgol_filter

import crw_config as config


def checklength(filename):
    runsdir = config.runsdir
    f = open(filename, "r")
    total_lines = sum(1 for line in f) - 1
    f.close()
    del f
    gc.collect()
    #ztfcrm_20221020157697_000640_zi_14
    _, datetime, run, thisfilter, choosechip = filename.split("/")[-1].replace(".csv","").split("_")
    
    pattern = f"{runsdir}/lightcurve_ztf_{datetime}_{run}_{thisfilter}_c{choosechip}*"
    print("pattern ", pattern)

    files = glob.glob(pattern)
    batch_size = 100
    original_lines = 0

    for i in tqdm(range(0, len(files), batch_size)):
        batch = files[i:i+batch_size]
        
        # Use xargs to run wc once with many files
        cmd = ['xargs', '-0', 'wc', '-l']
        input_data = '\0'.join(batch).encode()
        #print(input_data)
        result = subprocess.run(cmd, input=input_data, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
        #print(result)
        output = result.stdout.decode().strip().split('\n')
        for line in output:
            if 'total' not in line and line.strip():
                original_lines += int(line.split()[0]) - 1
        #print(original_lines)
        #exit()
        gc.collect()        
    if total_lines != original_lines:
        print(filename, total_lines, original_lines, "something is wrong!")
    print(resource.getrlimit(resource.RLIMIT_NOFILE))


def checklcvs(df_chips):
    df_chips_groups = df_chips.groupby(["date", "run", "filter", "chip"]).count()
    print(df_chips_groups.head())
    nmissing = 0
    for ind in df_chips_groups.index:
        datetime, run, thisfilter, choosechip = ind
        choosechip = choosechip.replace('c','')

        filedetails, lcvarrayfile, chipfilename, \
            featuresfile, predictionfile, rootdir = config.makenames(datetime, run, 
                                                                     thisfilter, choosechip)
    
        #print("\t", chipfilename)

        if os.path.isfile(chipfilename):
            print("\t✔", ind, "nlcvs:", df_chips_groups.loc[ind].values[0])
        else:
            print("->", ind, "nlcvs:", df_chips_groups.loc[ind].values[0])
            nmissing += 1
    print(f"sill missing {nmissing} files")
    return 

def getalllcvs(runsdir, extension, verbose=False):

    pattern = f"{runsdir}/*{extension}"

    print(f"looking for lightcurves with pattern {pattern}")
    lcvs = glob.glob(pattern)

    print(f"found {len(lcvs)} lightcurve files")

    dates = [l.split("/")[-1].split('_')[2] for l in lcvs]
    runs = [l.split("/")[-1].split('_')[3] for l in lcvs]
    filters = [l.split("/")[-1].split('_')[4] for l in lcvs]
    chip = [l.split("/")[-1].split('_')[5] for l in lcvs]

    df_chips = pd.DataFrame(columns=["name", "date", "run", "filter", "chip"],
                            data = np.array([lcvs, dates, runs, filters, chip]).T)

    #print(df_chips)
    print("LIGHTCURVES PER CHIP")
    df_chips_groups = df_chips.groupby(["date", "run", "filter", "chip"]).count().rename({"name":"nlcvs"}, axis=1)
    if verbose:
        for i in df_chips_groups.index:
            print(df_chips_groups.loc[i])
    df_chips_groups.reset_index().to_csv(config.ROOT + "/allparams.csv")                                   
    return df_chips


def loaddata(choosechip=None, runsdir=None, thisfilter=None,
                   run=None, datetime=None, silent=None, loadX=True):
     #read in configs
    NCl=  config.NCl
    ap_column = config.ap_column

    verbose = config.VERBOSE
    diagnostic_plots = config.DIAGNOSTICPLOTS
    
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
    if silent:
        verbose = False
        diagnostic_plots = False
        
    extension = config.extension

    filedetails, lcvarrayfile, chipfilename, featuresfile, predictionfile, rootdir = config.makenames(datetime, run, thisfilter, choosechip)
    print("attempting to read ", lcvarrayfile)
    try:
      Xsave = np.load(lcvarrayfile, allow_pickle=True)
      print(Xsave.shape)
      X = Xsave[:,:,ap_column]
      outputs = [X]
    except FileNotFoundError:
      outputs = [None]
    try:
      Xpos = np.load(lcvarrayfile.replace("crwdata", "crwpos"), allow_pickle=True)
      outputs += [Xpos]
    except FileNotFoundError:
      outputs += [None]
    try:
      labels = np.load(f"{rootdir}crwprediction{filedetails}.npy", allow_pickle=True)
      outputs += [labels]
    except FileNotFoundError:
      outputs += [None]
    outputs += [filedetails]
    outputs += [choosechip, runsdir, thisfilter, run, datetime, extension, verbose]
    return outputs


def savehpflcs(X=None, choosechip=None, runsdir=None, thisfilter=None,
               run=None, datetime=None, verbose=None,
               order=3, window=101, diagnostic_plots=None):
    
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
    if diagnostic_plots is None:
      diagnostic_plots = config.DIAGNOSTICPLOTS
  
    filedetails, lcvarrayfile, chipfilename, featuresfile, predictionfile, rootdir = config.makenames(datetime, run, thisfilter, choosechip)
    if X is None:
      X, Xpos, labels, filedetails, choosechip, runsdir, thisfilter, run, datetime, extension, verbose = loaddata(choosechip=choosechip)
    Xmean = X.mean(axis=1, keepdims=True)
    hpflcs = highpass_filter(X, window=101, order=3)

    hpflcs = hpflcs + Xmean
    print(f"saving array into {lcvarrayfile.replace('crwdata', 'crwdataHP')}")
    print(lcvarrayfile.replace("crwdata", "crwdataHP"))
    np.save(lcvarrayfile.replace("crwdata", "crwdataHP"), hpflcs)

    if diagnostic_plots:
      fig, axs = plt.subplots(10,1,figsize=(10,10))
      axs = axs.ravel()
      for i in range(10):
        axs[i].plot(X[i], lw=0.1)
        axs[i].plot(hpflcs[i], alpha=0.8, lw=0.1)
      
      fig, axs = plt.subplots(10,1,figsize=(10,10))
      axs = axs.ravel()
      for i in range(10):
        axs[i].plot(X[i,:200], lw=1)
        axs[i].plot(hpflcs[i,:200], alpha=0.8, lw=1)
      plt.show()
      
      fig, axs = plt.subplots(2,1,figsize=(10,10))
      axs = axs.ravel()
      for i,j in enumerate([1269, 1272]):
        axs[i].plot(X[j], lw=0.1)
        axs[i].plot(hpflcs[j], alpha=0.8, lw=0.1)
      
      fig, axs = plt.subplots(2,1,figsize=(10,10))
      axs = axs.ravel()
      for i,j in enumerate([1269, 1272]):
        axs[i].plot(X[j,:200], lw=1)
        axs[i].plot(hpflcs[j,:200], alpha=0.8, lw=1)
      plt.show()
      
    return hpflcs


def checkna(x):
  assert np.isnan(x).sum()==0, "got nans here"


def parallel_list_comprehension(func, iterable, n_jobs=None):
    """
    Parallel version of list comprehension
    """
    import multiprocessing as mp
    
    if n_jobs is None:
        n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
        #n_jobs = mp.cpu_count()
    print(f"launching {n_jobs} jobs")
    
    with mp.Pool(n_jobs) as pool:
        return pool.map(func, iterable)


def process_iterable(data_iterable):
    for item in data_iterable:
        print(f"Processing: {item}")

  
def _gcm_(cdf, idxs):
    # Pre-allocate arrays to avoid repeated appending
    n = len(cdf)
    gcm = np.empty(n * 2)  # Over-allocate
    touchpoints = np.empty(n * 2, dtype=int)

    gcm[0] = cdf[0]
    touchpoints[0] = 0
    gcm_len, tp_len = 1, 1

    work_cdf = cdf
    work_idxs = idxs

    while len(work_cdf) > 1:
        distances = work_idxs[1:] - work_idxs[0]
        slopes = (work_cdf[1:] - work_cdf[0]) / distances

        minslope_idx = np.argmin(slopes)
        minslope = slopes[minslope_idx]
        minslope_idx += 1

        # Calculate extension directly into pre-allocated array
        extension_len = minslope_idx
        start_idx = gcm_len
        end_idx = gcm_len + extension_len

        gcm[start_idx:end_idx] = work_cdf[0] + distances[:minslope_idx] * minslope
        touchpoints[tp_len] = touchpoints[tp_len-1] + minslope_idx

        gcm_len = end_idx
        tp_len += 1

        work_cdf = work_cdf[minslope_idx:]
        work_idxs = work_idxs[minslope_idx:]

    return gcm[:gcm_len], touchpoints[:tp_len]

def _lcm_(cdf, idxs):
    g, t = _gcm_(1-cdf[::-1], idxs.max() - idxs[::-1])
    return 1-g[::-1], len(cdf) - 1 - t[::-1]

def _touch_diffs_(part1, part2, touchpoints):
    diff = np.abs((part2[touchpoints] - part1[touchpoints]))
    return diff.max(), diff

def diptst(dat, is_hist=False, numt=1000):
    """ diptest with pval """
    d, (_, idxs, left, _, right, _) = dip_fn(dat, is_hist)

    # Vectorized approach for uniform samples
    n_samples = idxs.shape[0]
    unifs = np.random.uniform(size=numt * n_samples).reshape([numt, n_samples])

    # Use list comprehension instead of apply_along_axis (often faster)
    unif_dips = np.array([dip_fn(unif, is_hist, True) for unif in unifs])

    pval = (np.sum(unif_dips >= d) + 1) / (numt + 1) if unif_dips.sum() > 0 else None

    return pval


def dip_fn(dat, is_hist=False, just_dip=False):
    """
        Compute the Hartigans' dip statistic either for a histogram of
        samples (with equidistant bins) or for a set of samples.
    """
    if is_hist:
        histogram = dat
        idxs = np.arange(len(histogram))
    else:
        # More efficient than digitize + bincount
        idxs, counts = np.unique(dat, return_counts=True)
        histogram = counts.astype(float)

    #special cases - should I remove them??
    # check for case 1<N<4 or all identical values
    if len(idxs) <= 4 or idxs[0] == idxs[-1]:
        left = []
        right = [1]
        d = 0.0
        return d if just_dip else (d, (None, idxs, left, None, right, None))

    cdf = np.cumsum(histogram, dtype=float)
    cdf /= cdf[-1]

    work_idxs = idxs
    work_histogram = np.asarray(histogram, dtype=float) / np.sum(histogram)
    work_cdf = cdf

    D = 0
    left = [0]
    right = [1]

    while True:
        left_part, left_touchpoints = _gcm_(work_cdf-work_histogram, work_idxs)
        right_part, right_touchpoints = _lcm_(work_cdf, work_idxs)

        d_left, left_diffs = _touch_diffs_(left_part,
                                           right_part, left_touchpoints)
        d_right, right_diffs = _touch_diffs_(left_part,
                                             right_part, right_touchpoints)

        if d_right > d_left:
            xr = right_touchpoints[d_right == right_diffs][-1]
            xl = left_touchpoints[left_touchpoints <= xr][-1]
            d = d_right
        else:
            xl = left_touchpoints[d_left == left_diffs][0]
            xr = right_touchpoints[right_touchpoints >= xl][0]
            d = d_left

        left_diff = np.abs(left_part[:xl+1] - work_cdf[:xl+1]).max()
        right_diff = np.abs(right_part[xr:]
                            - work_cdf[xr:]
                            + work_histogram[xr:]).max()

        if d <= D or xr == 0 or xl == len(work_cdf):
            the_dip = max(np.abs(cdf[:len(left)] - left).max(),
                          np.abs(cdf[-len(right)-1:-1] - right).max())
            if just_dip:
                return the_dip/2
            else:
                return the_dip/2, (cdf, idxs, left, left_part, right, right_part)
        else:
            D = max(D, left_diff, right_diff)

        work_cdf = work_cdf[xl:xr+1]
        work_idxs = work_idxs[xl:xr+1]
        work_histogram = work_histogram[xl:xr+1]

        left[len(left):] = left_part[1:xl+1]
        right[:0] = right_part[xr:-1]

        
def lowpass_filter(data, wavelet='db4', level=5):
    """
    Simple wavelet denoising example.
    """

    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")

    means = data.mean(axis=1)
    signal = data - means[:,np.newaxis]
    filtered_data = np.zeros_like(data, dtype=data.dtype)
    for i in tqdm(range(data.shape[0])):
      coeff = pywt.wavedec(signal[i], wavelet, level=level)
      sigma = np.median(np.abs(coeff[-level])) / 1
      uthresh = sigma * np.sqrt(2 * np.log(len(signal[i])))
      coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
      filtered_data[i, :] = pywt.waverec(coeff, wavelet)
    return filtered_data + means[:,np.newaxis]
        
def highopass_filter(data, window=101, order=3):
    """Apply Savitzky-Golay filter for smoothing or derivative"""

    return data - savgol_filter(data, window, order, axis=1)
     

  

class QuickReview:
    def __init__(self, series):
        self.series = series
        self.reject_lightcurve = [0]*len(series)
        self.i = 0
        
        self.nav = widgets.HBox([widgets.Button(d) for d in ['←','Keep','Reject','→']])
        for btn, cmd in zip(self.nav.children, [lambda x: self.move(-1), lambda x: self.decide(0), 
                                              lambda x: self.decide(1), lambda x: self.move(1)]):
            btn.on_click(cmd)
        
        self.update()
        display(widgets.VBox([self.nav]))
    
    def move(self, step): 
        self.i = (self.i + step) % len(self.series)
        px.line(y=self.series[self.i]).show()
    
    def decide(self, val):
        self.reject_lightcurve[self.i] = val
        self.move(1)
    
    def update(self): 
        self.move(0)


def PR(tag, labels, k):
  tagged = (tag==1).sum()
  for i in range(k):
    print(f"\trecall cluster {i} ({(labels==i).sum()})    % tagged: {(((tag>0) * (labels==i)).sum()) / tag.sum() * 100:.0f}%")
    print(f"\tprecision cluster {i} ({(((tag==0) * (labels==i)).sum())}) % not tagged: {(((tag==0) * (labels==i)).sum()) / (len(tag) - tag.sum()) * 100:.0f}%")
  if -1 in labels:
    i=-1
    print(f"\trecall cluster {i} ({(labels==i).sum()})    % tagged {(((tag>0) * (labels==i)).sum()) / tag.sum() * 100:.0f}%")
    print(f"\tprecision cluster {i} ({(((tag==0) * (labels==i)).sum())}) % not tagged: {(((tag==0) * (labels==i)).sum()) / (len(tag) - tag.sum()) * 100:.0f}% ")

  for i in range(k):
    n = ((tag==1) * (labels==i)).sum()
    print(f"of tagged objects {n} ({n * 1.0 / tagged * 100:.2f}%) are in cluster {i}")
  if -1 in labels:
    i=-1
    n = ((tag==1) * (labels==i)).sum()    
    print(f"of tagged objects {n} ({n * 1.0 / tagged * 100:.2f}%) are in cluster {i}\n\n\n")    


"""# Random forest"""

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def aggregate_rf_cv_weighted(X, y, n_splits=5, n_estimators=100, **rf_params):
    """
    Aggregate RF models with weights based on CV performance
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    weights = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        rf = RandomForestClassifier(n_estimators=n_estimators, **rf_params)

        rf.fit(X_train, y_train)
        models.append(rf)

        # Calculate weight based on validation performance
        y_pred = rf.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        weights.append(score)
        """
        print(accuracy_score(y_val, y_pred))
        print(precision_score(y_val, y_pred, zero_division=0))
        print(recall_score(y_val, y_pred, zero_division=0))
        print(f1_score(y_val, y_pred, zero_division=0))
        """

    # Normalize weights
    weights = np.array(weights) / np.sum(weights)
    print(f"Model weights: {weights}")

    return models, weights, skf

def predict_weighted(models, weights, X, aggregation='SOFT'):
    """
    Predict using weighted aggregation
    """
    all_predictions = []

    for model, weight in zip(models, weights):
        if aggregation == 'hard':
            pred = model.predict(X)
        else:
            pred = model.predict_proba(X)
        all_predictions.append(pred * weight)  # Apply weight

    if aggregation == 'hard':
        all_predictions = np.array(all_predictions)
        final_predictions = []
        for i in range(X.shape[0]):
            # Weighted voting
            votes = all_predictions[:, i]
            unique_classes = np.unique(votes)
            class_weights = []
            for cls in unique_classes:
                class_weight = np.sum(weights[votes == cls])
                class_weights.append(class_weight)
            final_predictions.append(unique_classes[np.argmax(class_weights)])
        return np.array(final_predictions)

    else:  # soft voting
        return np.sum(all_predictions, axis=0)



def binary_badlcvs(X):
  # -- normalize the lightcurves
  clp = np.array([i.clip(*np.sort(i)[[10, -10]]) for i in X])
  norm = (clp - clp.min(axis=1, keepdims=True)) / \
    (clp.max(axis=1, keepdims=True) -clp.min(axis=1, keepdims=True))

  # -- intialize the predictions array (True=has jumps)
  pred = []

  # -- loop through time series
  for lc in norm:
   
    # -- bin to 2D
    H, xedges, yedges = np.histogram2d(np.arange(len(lc)), lc, bins=(100, 100))

    # -- predict based on whether there is a "gap" in the "image" and also
    #    check that there are not isolated points at the edges (typically these
    #    are cosmic rays)
    pred.append(np.array([(H.T > 1)[40:60].sum(), (H.T > 0)[:40].sum(), (H.T > 0)[60:].sum()])
      #((H.T > 1)[40:60].sum() < 150) and
      # min((H.T > 0)[:40].sum(), (H.T > 0)[60:].sum()) > 20
      )

    # -- convert to numpy array
  return np.array(pred)


      
def plotalllcs(X, labels=None, filedetails=None):
  
  colors = [ "k", "r", "b", "g", "y"]
  if not labels is None:
    Ncl = min(len(np.unique(labels)), 5)
    print(f"there are {Ncl} distinct labels  (max 5)")
    for l in np.sort(np.unique(labels))[::-1]:
      inds = labels == l
      nlcs_in_label = inds.sum()
      print(f"{nlcs_in_label} objects labeled {l}")
      true_indices = np.where(inds)[0]
      nrows = int(nlcs_in_label / 4 + 0.5)

      if nrows <= 20:
        fig, axs = plt.subplots(nrows, 5, figsize=(10, 10))
        axs = axs.ravel()
        for i,lc in tqdm(enumerate(X[inds])):
          axs[i].plot(lc, color=colors[l], alpha=0.6, lw=0.1)
          axs[i].text(0.9, 0.9, f"{true_indices[i]}",
                      transform=axs[i].transAxes, 
                      ha='right', va='top',
                      fontsize=5)
          axs[i].axis('off')
          
        for i in range(i, len(axs)):
          axs[i].axis('off')
        valid = 'valid' if l == 0 else 'not_valid'

        plt.suptitle(f"{valid}")
        plt.savefig(f"{config.ROOT}/plots/allplots{filedetails}_{valid}.png")        
          
      else:
        valid = 'valid' if l == 0 else 'not valid'        
        left = nlcs_in_label
        start = 0
        for j,batch in enumerate(range(0, nlcs_in_label, 100)):
          print(f"plotting {batch+100} of {valid} {nlcs_in_label} lcvs")
          fig, axs = plt.subplots(20, 5, figsize=(10, 10))
          axs = axs.ravel()
          for i,lc in tqdm(enumerate(X[inds][batch:min(batch+100, nlcs_in_label)])):
            axs[i].plot(lc, color=colors[l], alpha=0.6, lw=0.1)
            axs[i].text(0.9, 0.9, f"{true_indices[i+batch]}",
                        transform=axs[i].transAxes, 
                        ha='right', va='top',
                        fontsize=5)
            axs[i].axis('off')
            left -= 1
            start += 1
          plt.suptitle(f"{valid}")
          for i in range(100):
            axs[i].axis('off')    

          plt.savefig(f"{config.ROOT}/plots/allplots{filedetails}_batch{j}_{valid}.png")
      plt.show()

  else:
    nlcs = int(X.shape[0] / 4 + 0.5)

    fig, axs = plt.subplots(nlcs, 4+1, figsize=(10, n/2))
    axs = axs.ravel()
    for i,lc in tqdm(enumerate(X)):
        axs[i].plot(lc, color="k", alpha=0.6)
        axs[i].axis('off')
    for i in range(i, len(axs)):
        axs[i].axis('off')
    plt.suptitle(f"unlabeled")

    if not filedetails is None:
      plt.savefig(f"{config.ROOT}/plots/allplots{filedetails}.png")
  plt.show()



      
def plotallfeatures(features, labels=None, filedetails=None):
  
  colors = [ "k", "r", "b", "g", "y"]
  j = 0
  for i,f in enumerate(features.T):
    fig, axs = plt.subplots(5, 4, figsize=(10, 10))
    axs = axs.ravel()
    print(i)
    for k,g in enumerate(features.T[i+1:]):
      for l in np.unique(labels):
          inds = labels == l
          axs[k].plot(f[inds], g[inds], '.',  color=colors[l], alpha=0.6, lw=0.1)
          
          #axs[i].axis('off')
    plt.show()

def interactive_plot_selector(data):
    choices, fig, ax = [], *plt.subplots(figsize=(15,5))
    Nlcv = data.shape[0]
    print("going to show {Nlcv} lightcurve, select the good ones and throw away the bad ones")
    plt.ion()
    for i, row in enumerate(data):
        ax.clear(), ax.plot(row), ax.set_title(f'lcv {i}/{Nlcv}: y/n?'), plt.draw(), plt.pause(0.1)
        choices.append(input('Accept? (y/n): ').lower() == 'y')
    plt.close()
    return choices

  

