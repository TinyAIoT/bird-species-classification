#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --partition=requeue-gpu
# Use one of these options:
#   all (non-private) zen3 gpus:      gpu2080,gpua100,gputitanrtx,gpu3090,gpuhgx
#   private gpus (zen4,zen3):         requeue-gpu
#   zen4 nodes:                       gpu4090,gpuh200
#   skylake nodes:                    gpuv100
#   express gpu:                      gpuexpress (max time of 2:00:00 only for short training)
#SBATCH --time=0-02:00:00
#SBATCH --job-name=quant_torch
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/tmp/%u/tinybirdiary/tmp/compression/quantize_torch_%j.log

# Quantizes a trained model for the ESP32-S3 with compression/esp-dl/quantize_torch_model.py.
#
# Usage:
#   sbatch hpc/quantize_torch.sh <config> [further arguments of quantize_torch_model.py]
#   sbatch hpc/quantize_torch.sh compression/esp-dl/configs/squeezenet_none.yaml
#   sbatch hpc/quantize_torch.sh compression/esp-dl/configs/squeezenet_layerwise.yaml       # grid search, see below
#   sbatch hpc/quantize_torch.sh compression/esp-dl/configs/squeezenet_layerwise.yaml --including_bias
# <config> is relative to the code directory (or absolute). Paths inside the config are relative to $wd.
# For configs with LayerwiseEqualization_quantization all combinations of the parameters below are run,
# every run adds a row to <model_name>_quant-metrics.csv. Other configs are run once.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: sbatch hpc/quantize_torch.sh <config> [arguments of quantize_torch_model.py]"
    exit 1
fi

# Project paths
code="$HOME"/tinybirdiary
wd="$WORK"/tinybirdiary

config="$1"
shift
if [[ "$config" != /* ]]; then
    config="$code/$config"
fi

# Equalization parameters of the grid search
opt_levels=( 1 2 )
iterations=( 10 )
value_thresholds=( 0.5 1.0 2.0 2.5 )

module purge
ml uv

# Create and activate virtual environment with uv
uv venv "$wd"/.quant-torch-venv
source "$wd"/.quant-torch-venv/bin/activate
uv pip install torch torchvision PyYAML pandas tqdm scikit-learn matplotlib seaborn onnxruntime esp-ppq

run() {
    python "$code"/compression/esp-dl/quantize_torch_model.py --config "$config" --working_dir "$wd" "$@"
}

if grep -qE '^quant_config:[[:space:]]*"?LayerwiseEqualization_quantization' "$config"; then
    for opt_level in "${opt_levels[@]}"; do
        for iteration in "${iterations[@]}"; do
            for value_threshold in "${value_thresholds[@]}"; do
                echo "=== opt_level=$opt_level iterations=$iteration value_threshold=$value_threshold"
                run --opt_level "$opt_level" --iterations "$iteration" --value_threshold "$value_threshold" "$@"
            done
        done
    done
else
    run "$@"
fi

echo "end of quantization for Job $SLURM_JOB_ID: $(date +%Y.%m.%d-%H:%M:%S)"
