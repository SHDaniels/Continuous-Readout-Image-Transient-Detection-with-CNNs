#save filtered versions of files

import pandas as pd
import numpy as np
from scipy.ndimage import filters
from scipy.signal.windows import gaussian
import os
import gc  # garbage collector


def hpf(x, y, plot=False):

    b = gaussian(1000, 27)
    #ga = filtfilt(b/b.sum(), [1.0], y)
    ga = filters.convolve1d(y, b/b.sum())
    # if plot:
    #     plt.plot(x, y - ga+ y.mean(), alpha=0.9, c="g", label="g")
    return y - ga + y.mean()





folder_path = "/lustre/lrspec/ZTF_data/lightcurves/1020/"
save_path = '/lustre/lrspec/ZTF_data/lightcurves/1020_filtered/'



# # List all the files in the folder
files = [f for f in os.listdir(folder_path)]  # Adjust this to your file format
for file in files:
    newname2 = save_path + file[:-4]+"_filtered_hap2.csv"
    newname3 = save_path + file[:-4]+"_filtered_hap3.csv"
    if not os.path.exists(newname2):
        try:
            lc = pd.read_csv(folder_path+file)
            filtered2 = hpf(lc.index.values, lc['flux_hap_2'], plot=False)
            filtered3 = hpf(lc.index.values, lc['flux_hap_3'], plot=False)
        
            filtered2.to_csv(newname2)
            filtered3.to_csv(newname3)
            del lc, filtered2, filtered3, newname2, newname3
            gc.collect()
        except:
            print("HELP HELP IM HAVING ISSUES")
            print(newname2)