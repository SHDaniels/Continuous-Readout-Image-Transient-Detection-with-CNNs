'''
get dataframe of detections from cnn detections

'''

import review_findings_functions as rff
import os
import numpy as np
import pandas as pd
import gc


save_direc = "1020jan26"

fpath_extractions_saved = "/lustre/lrspec/ZTF_data/science_extractions/" + save_direc + "/"




if 1==1: #idk bestie i dont wanna delete the tab in all of them
    wid = 10
    abst_thr = .9999


    for i, file in enumerate(os.listdir(fpath_extractions_saved)):
        image_filename = os.fsdecode(file)
        filename_start = image_filename[:40]
        quad = image_filename[-16]
        path_to_save = fpath_extractions_saved+str(filename_start)+'_'+quad+'_alllist'
        
        if os.path.isfile( #if the one we want to make already exists
            path_to_save
        ):
            print("skipping ", filename_start)
            continue

        print(filename_start)

        try: 
            predsarray = np.load(fpath_extractions_saved+image_filename, allow_pickle=True)
        except: #
            continue

        alllist = rff.make_obj_list(predsarray, abst_thr, filename_start, quad)


        alllist.to_csv(path_to_save)
        del predsarray
        del alllist
        gc.collect()