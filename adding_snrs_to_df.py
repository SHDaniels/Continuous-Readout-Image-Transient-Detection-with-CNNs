import pandas as pd
import numpy as np
import review_findings_functions as rff
import importlib

df = pd.read_csv("/home/3155/inference_star_info_r12_bands")

df_plus_snr = rff.add_snr_whole_and_chunks(df)

df_plus_snr.to_csv('/home/3155/df_plus_snr')