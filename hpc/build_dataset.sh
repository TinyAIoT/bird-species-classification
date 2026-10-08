#!/bin/bash
#SBATCH --job-name=build-bird-ds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28GB
#SBATCH --partition=requeue-gpu
#SBATCH --requeue
#SBATCH --time=1-00:00:00            # d-hh:mm:ss
#SBATCH --output=/scratch/tmp/%u/tinybirdiary/tmp/build_dataset_%j.log
#SBATCH --mail-type=ALL

# Builds the dataset end to end with data/build_dataset.py.
# Every step is cached in the workdir, so a requeued/cancelled job just continues where it stopped.
#
# Usage:
#   sbatch hpc/build_dataset.sh                                   # all steps
#   sbatch hpc/build_dataset.sh --videos-per-species 20           # pilot
#   sbatch hpc/build_dataset.sh --steps filter assemble report    # only some steps
# All arguments are passed through to build_dataset.py (see python build_dataset.py --help).
#
# Downloading and frame extraction don't need a GPU. To not block a GPU for hours, run them as a CPU job
# and chain the GPU job after it, overriding the #SBATCH defaults on the command line:
#   jid=$(sbatch --parsable --partition=normal --gres=none hpc/build_dataset.sh --steps metadata split download frames)
#   sbatch --dependency=afterok:$jid hpc/build_dataset.sh --steps filter assemble report

set -euo pipefail

pip install --user -r "$HOME/tinybirdiary/requirements.txt"

# Project paths
CODE="$HOME/tinybirdiary"
WD="$WORK/tinybirdiary"
USER_AGENT=""

python "$CODE/data/build_dataset.py" \
  --workdir "$WD/data" \
  --user-agent "$USER_AGENT" \
  --filter-model "$CODE/data/filter-bird-background.pth" \
  --filter-batch-size 32 \
  --filter-workers 8 \
  --download-workers 4 \
  "$@"

echo "end of dataset creation for Job $SLURM_JOB_ID: $(date +%Y.%m.%d-%H:%M:%S)"
