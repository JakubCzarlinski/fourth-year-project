#!/bin/bash
#
#SBATCH --job-name=dddddddd # Job name for tracking
#SBATCH --partition=gecko     # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16     # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=/dcs/large/u2103071/fourth-year-project/implementations/DDD/ddddjpeg.out # Standard out from your job
#SBATCH --error=/dcs/large/u2103071/fourth-year-project/implementations/DDD/ddddjpeg.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

cd /dcs/large/u2103071/fourth-year-project/implementations/DDD
conda activate ddddjpeg
## Execute your program(s) ##
module load CUDA
python /dcs/large/u2103071/fourth-year-project/implementations/DDD/attack.py >> out.txt