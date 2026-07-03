#get filtered snrs

import pandas as pd
import numpy as np
import os
import gc  # garbage collector
import review_findings_functions as rff

# concat = pd.read_csv('/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/concat_with_band')

# concat2 = rff.add_snr_both_filtered_hap2(concat)

# concat2.to_csv('/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/concat_filtered2')
# concat2 = pd.read_csv('/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/concat_filtered2')


# concat3 = rff.add_snr_both_filtered_hap3(concat2)

# concat3.to_csv('/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/concat_filtered3')


#for eff below here

lcfpath='/lustre/lrspec/ZTF_data/efficiency_testing/nosat_eff_data/full_image_implantations_pos/lightcurves_pos_filtered/'

concat = pd.read_csv('/lustre/lrspec/ZTF_data/efficiency_testing/nosat_eff_data/full_image_implantations_pos/df_both_matched_thr_0.99') 

concat2 = rff.add_snr_both_filtered_hap2(concat, lcfpath)

concat2.to_csv('/lustre/lrspec/ZTF_data/efficiency_testing/nosat_eff_data/full_image_implantations_pos/df_snrs')
# concat2 = pd.read_csv('/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/concat_filtered2')


concat3 = rff.add_snr_both_filtered_hap3(concat2, lcfpath)

concat3.to_csv('/lustre/lrspec/ZTF_data/efficiency_testing/nosat_eff_data/full_image_implantations_pos/df_snrs')

lcfpath='/lustre/lrspec/ZTF_data/efficiency_testing/nosat_eff_data/full_image_implantations_neg/lightcurves_neg_filtered/'


neg = pd.read_csv('/lustre/lrspec/ZTF_data/efficiency_testing/nosat_eff_data/full_image_implantations_neg/df_both_matched_thr_0.99') 

neg = rff.add_snr_both_filtered_hap2(neg, lcfpath)

neg.to_csv('/lustre/lrspec/ZTF_data/efficiency_testing/nosat_eff_data/full_image_implantations_neg/df_snrs')
# concat2 = pd.read_csv('/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/concat_filtered2')


neg = rff.add_snr_both_filtered_hap3(neg, lcfpath)

neg.to_csv('/lustre/lrspec/ZTF_data/efficiency_testing/nosat_eff_data/full_image_implantations_neg/df_snrs')