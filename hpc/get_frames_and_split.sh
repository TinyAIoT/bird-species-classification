#!/bin/bash
#SBATCH --job-name=make-bird-ds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=requeue-gpu
#SBATCH --time=0-02:00:00            # d-hh:mm:ss
#SBATCH --output=/scratch/tmp/%u/BirdBox/tmp/get_frames_split_ds_%j.log
#SBATCH --mail-type=ALL

pip install --user -r $HOME/BirdBox/requirements.txt

# Project paths
CODE="$HOME/BirdBox/data"
DATA="$WORK/BirdBox/data/v1"

# Run training
python "$CODE/get_frames.py"\
  --workdir "$DATA" \
  --workers 8 \

echo "end of dataset creation for Job "$SLURM_JOB_ID" :"
echo "date +%Y.%m.%d-%H:%M:%S"