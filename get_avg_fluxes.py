'''dumb way to do this so i don't have to install anything'''

import os
import pandas as pd
import numpy as np

avg_fluxes = []
lc_folder = '/lustre/lrspec/ZTF_data/lightcurves/1020/'
for file in os.listdir(lc_folder):
    try:
        lc = pd.read_csv(os.path.join(lc_folder, file))['flux_hap_2']
        avg_fluxes.append([file, np.average(lc)])
    except:
        continue
                                                       
avg_fluxes = np.array(avg_fluxes)
np.save('/lustre/lrspec/ZTF_data/lightcurves/average_fluxes_1020', avg_fluxes)
                                                       
                                                       
                                                       