import crw_config
import os
import sys
import pandas as pd
from crw_pipeline import launchpipeline


if __name__ == "__main__":
    allparams = pd.read_csv(crw_config.root + "/allparams.csv", dtype=str)
    
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    launchpipeline(choosechip=None, runsdir=None, thisfilter=None,
                       run=None, datetime=None, silent=False)





