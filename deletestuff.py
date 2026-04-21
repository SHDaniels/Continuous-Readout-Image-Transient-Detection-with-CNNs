'''

gotta delete some predsarrays

'''

import os

folder_path = "/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/"
delete_path = '/lustre/lrspec/ZTF_data/science_extractions/1020april26/predsarrays_offset_sat_newhparam_b50_allstamps_satdumbsplit_folds_'

files = [f for f in os.listdir(folder_path) if f.endswith('predsarray.npy')] 
for file in files:
    for iteration in range(0,5):
        todelete = delete_path+str(iteration)+'/'+file
        if os.path.exists(todelete):
            os.remove(todelete) 