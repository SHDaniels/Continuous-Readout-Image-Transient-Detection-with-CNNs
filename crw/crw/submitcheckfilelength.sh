#!/bin/bash

#SBATCH --output=logs/lc_analysis_%A_%a.out
#SBATCH --error=logs/lc_analysis_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --array=0-199
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=standard

module load python/3.11.8

source /home/2073/initCRM.sh 
python3 -m pip install --user -r requirements.txt

python3 checkfilelengthslurm.py --chunk "$SLURM_ARRAY_TASK_ID" --total-chunks 200



