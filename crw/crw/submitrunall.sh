#!/bin/bash
#SBATCH --job-name=ztf_extract
#SBATCH --output=logs/output_%A_%a.out
#SBATCH --error=logs/error_%A_%a.err
#SBATCH --cpus-per-task=4    # Each task gets 4 CPU cores instead of 1
#SBATCH --array=0-7%4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --partition=standard     # <-- REQUIRED ON DARWIN

# Load your environment (adjust for DARWIN)
source /home/2073/initCRM.sh 

# Load Python module
module load python/3.11.8

# Install all packages from requirements file
python3 -m pip install --user -r requirements.txt


# Run your script
python3 runallo1.py
