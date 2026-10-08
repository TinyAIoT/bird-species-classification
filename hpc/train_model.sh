#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # Total number of tasks
#SBATCH --gres=gpu:1                  # Request 4 GPUs
#SBATCH --cpus-per-task=4            # Number of CPU cores
# try --mem=16GB                    # Memory per node
#SBATCH --mem=28GB
#SBATCH --partition=d0giesek,gpu3090,gpu2080,gpu4090,gpuexpress-zen4

#SBATCH --time=0-12:00:00
#SBATCH --job-name=train_birdiary
#SBATCH --mail-type=ALL
#SBATCH --output /scratch/tmp/%u/tinybirdiary/tmp/training/train_birdiary_%j.log

#load modules 
module purge
# PyTorch/2.1.2-CUDA-12.1.1 
ml uv

# place of code in palma
code="$HOME"/tinybirdiary
wd="$WORK"/tinybirdiary

cd $wd

# Create and activate virtual environment with uv
uv venv .train-torch-venv 
source .train-torch-venv/bin/activate

# Install all Python packages with uv for consistency
uv pip install matplotlib torch torchvision torchaudio
uv pip install PyYAML tqdm scikit-learn einops seaborn torch_optimizer

# For debugging use: 
# pip install torchprofile 

# run code with flags
# run sbatch with flag --config path/to/config.yaml
python "$code"/model/training/run_transfer_learning_pipeline.py --working_dir $wd "$@"
echo "end of Training for Job "$SLURM_JOB_ID" :"
echo `date +%Y.%m.%d-%H:%M:%S`