import numpy as np
import os

paths = ["/lustre/lrspec/ZTF_data/science_extractions/1020april26/predsarrays_offset_sat_newhparam_b50_allstamps_satdumbsplit_folds_0/",
         "/lustre/lrspec/ZTF_data/science_extractions/1020april26/predsarrays_offset_sat_newhparam_b50_allstamps_satdumbsplit_folds_1/",
         "/lustre/lrspec/ZTF_data/science_extractions/1020april26/predsarrays_offset_sat_newhparam_b50_allstamps_satdumbsplit_folds_2/",
         "/lustre/lrspec/ZTF_data/science_extractions/1020april26/predsarrays_offset_sat_newhparam_b50_allstamps_satdumbsplit_folds_3/",
         "/lustre/lrspec/ZTF_data/science_extractions/1020april26/predsarrays_offset_sat_newhparam_b50_allstamps_satdumbsplit_folds_4/",
]

savedir = "/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/"
for i, file in enumerate(os.listdir(paths[0])):
    if os.path.exists(os.path.join(savedir,file))==False: #see if it already exists
        if np.sum([os.path.exists(os.path.join(paths[i],file)) for i in range(1,5)])==4: 
            #make sure all 5 done
            arrays = []
            for path in paths:
                try:
                    arrays.append(np.load(path+file))
                except:
                    print("got an issue on "+path+file)
                    continue
            if len(arrays)==5:
                average = np.average(arrays, axis=0)
                np.save(os.path.join(savedir,file), average)
            else:
                print("didn't do the averaging for "+path+file)
