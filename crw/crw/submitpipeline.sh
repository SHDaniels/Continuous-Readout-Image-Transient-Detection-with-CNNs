#!/bin/bash
#SBATCH --job-name=ztf_extract
#SBATCH --output=logs/output_%A_%a.out
#SBATCH --error=logs/error_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --array=0-128%4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --partition=standard     # <-- REQUIRED ON DARWIN

# Load your environment (adjust for DARWIN)
source /home/2073/initCRM.sh 

# Load Python module
module load python/3.11.8

# Install all packages from requirements file
python3 -m pip install --user -r requirements.txt


# Run your script
python3 launchallpipeline.py slurm
#python3 launchallpipeline_onefile.py slurm
