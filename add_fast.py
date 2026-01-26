'''
adding sobels/stamps to whole transients df

'''

import pandas as pd
import review_findings_functions as rff
import numpy as np

# df = pd.read_csv(
#     "/lustre/lrspec/ZTF_data/efficiency_testing/df_snr_4")

# df_updated = rff.add_sobel_ratio_fast(df)

# df_updated.to_csv("/lustre/lrspec/ZTF_data/efficiency_testing/df_snr_sobel")

# df = pd.read_csv(
#     "/lustre/lrspec/ZTF_data/efficiency_testing/df_snr_sobel")

# df_updated = rff.add_stamps3(df)

# df_updated.to_csv("/lustre/lrspec/ZTF_data/efficiency_testing/df_stamps")



# df = pd.read_csv(
#     "/lustre/lrspec/ZTF_data/efficiency_testing/df_stamps")

# df_updated = rff.add_normsobel_stuff(df)

# df_updated.to_csv("/lustre/lrspec/ZTF_data/efficiency_testing/df_stamps_and_sobel")

# df_eff = pd.read_csv(base_path+"df_snr_sobel_eff")
# # df_eff = pd.read_csv(

# df_updated_eff = rff.add_normsobel_stuff(df_eff, eff=True) 
    

# df_updated.to_csv("/lustre/lrspec/ZTF_data/efficiency_testing/df_eff_sobel")



                
        


# df_inf_nodups = pd.read_csv("/lustre/lrspec/ZTF_data/efficiency_testing/df_stamps_and_sobel") #there are dups but that's fine
# df_inf_nodups['quadrant']=df_inf_nodups['quad']

# df_inf_nodups = add_num_consecutive(df_inf_nodups)


# df_inf_nodups = add_satellite_flag(df_inf_nodups, new=True)

# df_inf_nodups.to_csv("/lustre/lrspec/ZTF_data/efficiency_testing/df_stamps_sobel_and_sat")

# df_inference = pd.read_csv("/lustre/lrspec/ZTF_data/efficiency_testing/df_stamps_sobel_and_sat")

# added = rff.find_dups(df_inference)

# added.to_csv("/lustre/lrspec/ZTF_data/efficiency_testing/df_inf_dups")

# df = pd.read_csv("/lustre/lrspec/ZTF_data/efficiency_testing/df_inf_dups")

# df = df[df['is_smaller_dup']==0]

# df_new = rff.add_num_consecutive(df)

# df_new = rff.add_sat_and_proximity(df_new)

# df_new.to_csv("/lustre/lrspec/ZTF_data/efficiency_testing/df_inf_sats")

df = pd.read_csv("/lustre/lrspec/ZTF_data/efficiency_testing/df_inf_sats")

df = rff.add_num_in_file_in_star(df)

df_new = rff.add_num_consecutive(df[df['num_in_file_in_star']<50])

df_new = rff.add_sat_and_proximity(df_new)

df_new.to_csv("/lustre/lrspec/ZTF_data/efficiency_testing/df_inf_sats2")