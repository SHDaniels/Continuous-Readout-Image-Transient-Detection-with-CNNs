'''
get dataframe of detections from cnn detections

'''

import review_findings_functions as rff
import os
import numpy as np
import pandas as pd
import gc
import sys

abst_thr = float(sys.argv[2])

fpath_extractions_saveds = [
        #"/lustre/lrspec/ZTF_data/efficiency_testing/nosat_eff_data/full_image_implantations_pos/predsarrays_"+sys.argv[1]+"/"], 
"/lustre/lrspec/ZTF_data/efficiency_testing/nosat_eff_data/full_image_implantations_neg/predsarrays_"+sys.argv[1]+"/"]
#]


# save_direc = "1020jan26"

# fpath_extractions_saved = "/lustre/lrspec/ZTF_data/science_extractions/" + save_direc + "/"



# fpath_extractions_saveds = [
#         "/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_neg/presdarrays_flipped/", "/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_pos/presdarrays_flipped/"
# ]
# "/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_pos/presdarrays_both/",

# "/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_neg/presdarrays_both/",  
# fpath_extractions_saveds = [
#         "/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_pos/predsarrays_dup/", "/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_neg/predsarrays_dup/",
# ]


for fpath_extractions_saved in fpath_extractions_saveds: 
    wid = 10
    # abst_thr = .9999
    
    files = [f for f in os.listdir(fpath_extractions_saved) if f.endswith("predsarray.npy")] 
    for i, file in enumerate(files):
        image_filename = os.fsdecode(file)
        filename_start = image_filename[:40]
        quad = image_filename[-16]
        path_to_save = fpath_extractions_saved+str(filename_start)+'_'+quad+'_alllist_thr_'+str(abst_thr)
        
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


        alllist.to_csv(path_to_save)
        del predsarray
        del alllist
        gc.collect()