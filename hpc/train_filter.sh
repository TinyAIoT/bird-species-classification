#!/bin/bash
#SBATCH --job-name=bird-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=requeue-gpu
#SBATCH --time=0-02:00:00            # d-hh:mm:ss
#SBATCH --output=/scratch/tmp/%u/BirdBox/tmp/train_%j.log
#SBATCH --job-name=TrainFilter
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akurkela@uni-muenster.de

pip install --user -r $HOME/BirdBox/requirements.txt

# Project paths
CODE="$HOME/BirdBox/data"
DATA="$WORK/BirdBox/data/v1"

# Run training
python "$CODE/data_cleaner.py" train \
  --workdir "$DATA" \
  --batch-size 64 \
  --workers 8 \
  --epochs 10 \
  --log-every 100

echo "end of Training for Job "$SLURM_JOB_ID" :"
echo "date +%Y.%m.%d-%H:%M:%S"