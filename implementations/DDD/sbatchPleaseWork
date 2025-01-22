#!/bin/bash
#
#SBATCH --job-name=DDDDJPEG     # Job name for tracking
#SBATCH --partition=eagle      # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --mem=60000            # 60GB RAM
#SBATCH --time=12:00:00        # Job time limit set to 12 hours
#
#SBATCH --output=jupyter.log
#SBATCH --error=jupyter.err
#SBATCH --mail-type=BEGIN # Get email when your allocation starts

source /etc/profile.d/modules.sh
module load CUDA
module load cs909-python
conda activate ddddjpeg
cd /dcs/large/u2103071/fourth-year-project/implementations/DDD # You might want to use the cd command here to change the working directory that jupyter notebook will use
jupyter notebook --ip=0.0.0.0 --port=11871 --no-browser


