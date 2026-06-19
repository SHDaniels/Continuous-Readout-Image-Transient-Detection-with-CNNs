import pandas as pd
import glob

# Read all chunk files
chunk_files = sorted(glob.glob('/lustre/lrspec/ZTF_data/outputs/1020/lightcurvefiles_info_chunk_*.csv'))
dfs = [pd.read_csv(f) for f in chunk_files]

# Combine
combined = pd.concat(dfs, ignore_index=True)

# Save
combined.to_csv('/lustre/lrspec/ZTF_data/outputs/1020/lightcurvefiles_info.csv', index=False)
print(f"Combined {len(combined)} records from {len(chunk_files)} chunks")
