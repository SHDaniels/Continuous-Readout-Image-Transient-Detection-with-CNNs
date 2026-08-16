'''
let's get the lcs for the eff data - the original that were later implanted
'''

import os
import pandas as pd
import numpy as np

df_implants_pos = pd.read_csv("/lustre/lrspec/ZTF_data/efficiency_testing/nosat_eff_data/full_image_implantations_pos/df_implants_pos") #we used the same files for neg and pos

#idk i kinda wanna do it a different way....maybe i give up on this
