#!/bin/bash
#SBATCH --job-name=make-bird-ds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=requeue-gpu
#SBATCH --time=0-02:00:00            # d-hh:mm:ss
#SBATCH --output=/scratch/tmp/%u/BirdBox/tmp/clean_data_%j.log
#SBATCH --mail-type=ALL

pip install --user -r $HOME/BirdBox/requirements.txt

# Project paths
CODE="$HOME/BirdBox/data"
DATA="$WORK/BirdBox/data/v1/dataset"

# TODO edit to take sbatch args for train/val/test 

# Run on train dataset
python "$CODE/data_cleaner.py" run\
  --frames-dir "$DATA/train" \
  --model-path "$CODE/acc99.37_e5_bird_filter_0901_0003.pth"
# Run on val dataset
python "$CODE/data_cleaner.py" run\
  --frames-dir "$DATA/val" \
  --model-path "$CODE/acc99.37_e5_bird_filter_0901_0003.pth"
# Run on test dataset
python "$CODE/data_cleaner.py" run\
  --frames-dir "$DATA/test" \
  --model-path "$CODE/acc99.37_e5_bird_filter_0901_0003.pth"

echo "end of dataset creation for Job "$SLURM_JOB_ID" :"
echo "date +%Y.%m.%d-%H:%M:%S"