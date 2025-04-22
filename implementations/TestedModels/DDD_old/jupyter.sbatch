#!/bin/bash
#
#SBATCH --job-name=jupyter     # Job name for tracking
#SBATCH --partition=falcon      # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --mem=60000            # 60GB RAM
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --output=jupyter.log
#SBATCH --error=jupyter.err
#SBATCH --mail-type=BEGIN # Get email when your allocation starts

source /etc/profile.d/modules.sh
module load CUDA
module load python-ml
conda activate myenv
cd ~/CS407/DDD/ # You might want to use the cd command here to change the working directory that jupyter notebook will use
jupyter notebook --ip=0.0.0.0 --port=11888 --no-browser