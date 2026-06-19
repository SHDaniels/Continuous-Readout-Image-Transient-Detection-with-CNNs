#get filtered snrs

import pandas as pd
import numpy as np
import os
import gc  # garbage collector
import review_findings_functions as rff

# concat = pd.read_csv('/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/concat_with_band')

# concat2 = rff.add_snr_both_filtered_hap2(concat)

# concat2.to_csv('/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/concat_filtered2')
concat2 = pd.read_csv('/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/concat_filtered2')


concat3 = rff.add_snr_both_filtered_hap3(concat2)

concat3.to_csv('/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/concat_filtered3')