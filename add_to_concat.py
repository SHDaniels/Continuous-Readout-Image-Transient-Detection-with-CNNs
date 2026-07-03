'''
add some quantitites to concat
'''
import numpy as np
import pandas as pd

import review_findings_functions as rff


# concat_filtered = pd.read_csv('/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/concat_filtered3') 
#have both 


# concat_filtered = rff.add_snr_column(concat_filtered)
# concat_filtered = rff.add_distance_from_start(concat_filtered)
# concat_filtered.to_csv('/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/concat_filtered3') 

concat_filtered = pd.read_csv('/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/concat_filtered_with_rf') 
concat_filtered_with_rf = rff.add_rf_lightcurves(concat_filtered, '/lustre/lrspec/ZTF_data/random_forest_training/predictions_reupdated.csv')
concat_filtered_with_rf.to_csv('/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/concat_filtered_with_rf_predictions') 