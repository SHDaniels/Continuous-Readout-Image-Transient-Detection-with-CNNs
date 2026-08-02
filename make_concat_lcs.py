'''make time-adjusted concat light curves'''

import numpy as np
import os
import pandas as pd

folder_path = "/lustre/lrspec/ZTF_data/lightcurves/1020/"
new_folder_path = "/lustre/lrspec/ZTF_data/lightcurves/1020_time_concat/"


def make_big_array(filename, total_len = 3080*28):
    lightcurve_path = "/lustre/lrspec/ZTF_data/lightcurves/1020/"
    lcs_total = []
    
    for i in range(0, 29):
        new_filename = filename[:48]+f'{int(i):02}'+filename[50:]
        # filename_list.append(new_filename)
        lc = pd.read_csv(lightcurve_path+new_filename)['flux_hap_2']
        lcs_total = np.hstack((lcs_total,lc))
    closest_y = int(filename[-8:-4])
    lcs_adjust = lcs_total[closest_y:total_len+closest_y]
    return  lcs_adjust

# # List the files in the folder
all_files = [f for f in os.listdir(folder_path) if f.startswith('lightcurve')]  #this is all of them
all_files = np.array(all_files)
our_files = []

for file in all_files:
    if file[-20:-16]=='0000':
        our_files.append(file)

for file in our_files:
    try:
        new_array = make_big_array(file)
        new_filename = file[:-4]+'_time_adjusted_concat'
        np.save(new_folder_path+new_filename, new_array)
    except:
        print("had an issue: "+file)
        continue