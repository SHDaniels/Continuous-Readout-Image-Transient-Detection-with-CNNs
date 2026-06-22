import crw_config
import os
import sys
import pandas as pd
from crw_pipeline import launchpipeline


if __name__ == "__main__":
    allparams = pd.read_csv(crw_config.root + "/allparams.csv", dtype=str)
    
    if len(sys.argv) > 1 and sys.argv[1] == "slurm":

        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
        if task_id < allparams.shape[0]:
            chip = allparams.loc[task_id, "chip"].replace("c","")
            run  = allparams.loc[task_id, "run"]
            filt = allparams.loc[task_id, "filter"]
            dt = allparams.loc[task_id, "date"]
            if 1: #task_id >= 50 and task_id < 90:
                print(f"Task {task_id}: Processing date {dt} {run} {filt} {chip}")
                launchpipeline(choosechip=chip, runsdir=crw_config.runsdir, thisfilter=filt,
                       run=run, datetime=dt, silent=False)

        else:
            print(f"Task {task_id}: Nothing left to process (only {allparams.shape[0]} dates)")


    else:
        for i in allparams.index[:2]:
            chip = allparams.loc[i, "chip"].replace("c","")
            run  = allparams.loc[i, "run"]
            filt = allparams.loc[i, "filter"]
            dt = allparams.loc[i, "date"]
            
            print(chip, run, filt, dt)
            launchpipeline(choosechip=chip, runsdir=crw_config.runsdir, thisfilter=filt,
                       run=run, datetime=dt, silent=False)
            




