#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=express
# try: normal ,express, long
#SBATCH --mem=18GB
#SBATCH --time=0-01:00:00
#SBATCH --job-name=birds_ds
#SBATCH --mail-type=ALL
#SBATCH --output /scratch/tmp/%u/BirdBox/tmp/prep_dataset_dss%j.log

# Load modules 
module purge
module load palma/2023a  GCC/12.3.0  OpenMPI/4.1.5
module load TensorFlow/2.13.0
module load scikit-learn/1.3.1
pip install --user matplotlib

# Code location in palma
code="$HOME"/BirdBox
wd="$WORK"/BirdBox
user_agent="bird-species-classification/1.0 (Uni Münster, Master Thesis; contact: akurkela@uni-muenster.de)"

# Creates a list with all stations
#python "$code"/data/get_stations.py --workdir $wd --user-agent $user_agent

# Get the movements for all stations
python "$code"/data/get_movements.py --workdir $wd --user-agent $user_agent

# Download the videos from the movements
python "$code"/data/get_videos.py --workdir $wd --user-agent $user_agent

# Split videos into frames
python "$code"/data/get_frames.py --workdir $wd
