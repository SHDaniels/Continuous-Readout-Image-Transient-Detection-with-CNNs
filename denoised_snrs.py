import review_findings_functions as rff
import pandas as pd

df = pd.read_csv('/lustre/lrspec/ZTF_data/science_extractions/1020april26/average/concat_filtered_with_rf') 
df = rff.add_snr_both_denoised(df)
df.to_csv('/lustre/lrspec/ZTF_data/concat_filtered_denoised_rf')