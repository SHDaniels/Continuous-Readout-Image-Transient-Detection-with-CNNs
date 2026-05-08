'''
get dataframe of detections from cnn detections

'''

import review_findings_functions as rff
import os
import numpy as np
import pandas as pd
import gc
import sys

fpath_extractions_saveds = [
        "/lustre/lrspec/ZTF_data/science_extractions/actually_training/average/"]


for fpath_extractions_saved in fpath_extractions_saveds: 
    wid = 10
    abst_thr = .99
    
    files = [f for f in os.listdir(fpath_extractions_saved) if f.endswith("predsarray.npy")] 
    for i, file in enumerate(files):
        image_filename = os.fsdecode(file)
        filename_start = image_filename[:40]
        quad = image_filename[-16]
        path_to_save = fpath_extractions_saved+str(filename_start)+'_'+quad+'_alllist'
        
        if os.path.isfile( #if the one we want to make already exists
            path_to_save
        ):
            print("skipping ", filename_start)
            continue

        print(file)

        try: 
            predsarray = np.load(fpath_extractions_saved+image_filename, allow_pickle=True)
        except: #
            print("didn't load")
            print(fpath_extractions_saved+image_filename)
            continue

        alllist = rff.make_obj_list(predsarray, abst_thr, filename_start, quad)


        if type(alllist)==pd.DataFrame:
            alllist.to_csv(path_to_save)
        del predsarray
        del alllist
        gc.collect()