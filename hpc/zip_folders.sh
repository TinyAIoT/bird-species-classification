#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --partition=express
# try: normal ,express, long
#SBATCH --mem=18GB
#SBATCH --time=0-02:00:00
#SBATCH --job-name=zip_birds_ds
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akurkela@uni-muenster.de
#SBATCH --output /scratch/tmp/%u/BirdBox/tmp/zip_folders_dss%j.log

wd="$WORK"/BirdBox/data/

tar -czf "$wd"/dataset_v2_train.tar.gz "$wd"/v2/dataset/train/ &
tar -czf "$wd"/dataset_v2_test.tar.gz "$wd"/v2/dataset/test/ &
tar -czf "$wd"/dataset_v2_val.tar.gz "$wd"/v2/dataset/val/ &
tar -czf "$wd"/dataset_v2_no_bird.tar.gz "$wd"/v2/dataset/no_bird/ &
wait