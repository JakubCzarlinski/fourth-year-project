#!/bin/bash
#
#SBATCH --job-name=jupyter     # Job name for tracking
#SBATCH --partition=falcon      # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --mem=60000            # 60GB RAM
#SBATCH --time=24:00:00        # Job time limit set to 12 hours
#
#SBATCH --output=jupyter2.log
#SBATCH --error=jupyter2.err

source /etc/profile.d/modules.sh
conda activate newenv
# python3 attack_robustness.py "original"
python3 attack_robustness.py "diffjpeg"
