#!/bin/bash
#
#SBATCH --job-name=ddd-fast # Job name for tracking
#SBATCH --partition=gecko,falcon     # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10     # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=sbatch/joboutput_%j.out # Standard out from your job
#SBATCH --error=sbatch/joboutput_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

mkdir -p sbatch

## Execute your program(s) ##
module load CUDA
conda activate group
python3 attack.py