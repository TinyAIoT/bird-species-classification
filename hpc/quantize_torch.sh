#!/bin/bash

##SBATCH --nodes=1
#SBATCH --ntasks=1                    # Total number of tasks
#SBATCH --gres=gpu:1                  # Request 2 GPUs
#SBATCH --cpus-per-task=4             # Number of CPU cores
#SBATCH --mem=10GB                    # Memory per node

#SBATCH --partition=requeue-gpu
# Use one of these options:
#   all (non-private) zen3 gpus:      gpu2080,gpua100,gputitanrtx,gpu3090,gpuhgx
#   private gpus (zen4,zen3):         requeue-gpu
#   zen4 nodes:                       gpu4090,gpuh200
#   skylake nodes:                    gpuv100
#   express gpu:                      gpuexpress (max time of 2:00:00 only for short training)

#SBATCH --time=0-01:00:00

#SBATCH --job-name=quant_torch

#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/tmp/%u/BirdBox/tmp/compression/quantize_torch_%j.log

#load modules 
module purge
ml palma/2023a 
ml foss/2023a 
ml PyTorch/2.1.2-CUDA-12.1.1
ml scikit-learn/1.3.1
ml matplotlib/3.7.2

pip install --user torchvision==0.16.2
pip install --user seaborn
pip install --user onnxruntime 
pip install --user esp-ppq

# place of code in palma
home="$HOME"
wd="$WORK"/BirdBox
code="$HOME"/BirdBox
log_path="$WORK"/BirdBox/tmp/compression/torch_quantize_"$SLURM_JOB_ID"
config="$WORK"/BirdBox/configs/compression/mbnv2.yaml

# Define an array of input models as parameters
# opt_level=( 1 2 )
# iterations=( 2)
# value_threshold=( 0.2 0.5 0.8 1 1.5 2)
# including_bias=( False True )
# bias_multiplier=( 0.5 )
# including_act=( False True )
# act_multiplier=( 0.5 )

opt_level=( 1 2 )
iterations=( 10 )
value_threshold=( 0.5 )

# Loop through the input models and convert them to ONNX format
for opt_level in "${opt_level[@]}"
do
    for iterations in "${iterations[@]}"
    do
        for value_threshold in "${value_threshold[@]}"
        do
            # for including_bias in "${including_bias[@]}"
            # do
            #     for bias_multiplier in "${bias_multiplier[@]}"
            #     do
            #         for including_act in "${including_act[@]}"
            #         do
            #             for act_multiplier in "${act_multiplier[@]}"
            #             do
                            python "$code"/compression/esp-dl/quantize_torch_model.py --opt_level $opt_level --iterations $iterations --value_threshold $value_threshold --working_dir $wd --config $config "$@"
                            # srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --mem=6GB --gres=gpu:1 python "$code"/compression/esp-dl/quantize_torch_model.py --opt_level $opt_level --iterations $iterations --value_threshold $value_threshold --working_dir $wd "$@" > "$log_path"_model_"$opt_level"_"$iterations"_"$value_threshold".log 2>&1 &
            #             done
            #         done
            #     done
            # done
        done
    done
done
wait


# Parallel Computing

# declare -a opt_levels=(1 1 )
# declare -a iterations=(2 10)
# declare -a value_thresholds=(0.8 0.8)

# # Loop through configurations and run the script
# for i in "${!opt_levels[@]}"; do
#     srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --mem=6GB --gres=gpu:1 python "$code"/compression/esp-dl/quantize_torch_model.py \
#         --opt_level ${opt_levels[$i]} \
#         --iterations ${iterations[$i]} \
#         --value_threshold ${value_thresholds[$i]} \
#         --working_dir $wd "$@" > "$log_path"_model_"${opt_levels[$i]}"_"${iterations[$i]}"_"${value_thresholds[$i]}".log 2>&1 &
# done

# # Wait for all background jobs to complete
# wait

echo "end of Training for Job "$SLURM_JOB_ID" :"
echo `date +%Y.%m.%d-%H:%M:%S`
