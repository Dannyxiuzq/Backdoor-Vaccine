#!/bin/bash

# Make sure to give execute permission to this script
# chmod +x run_inference.sh

# Redirect HuggingFace cache off /home (which is at 97% capacity on this host)
# onto /mnt/data, where the user has 2.8 TB free. Every script in this repo
# sources this file before doing GPU work, so this catches any incidental HF
# download (tokenizer fallbacks, datasets) even when base_model is already a
# local path.
export HF_HOME=/mnt/data/zengqixiu/hf_cache
mkdir -p "$HF_HOME/hub"

# Disable Weights & Biases telemetry — LlamaFactory's HF Trainer integration
# auto-enables it when the wandb package is importable, and it hard-fails the
# training run when no API key is configured. We don't need it here.
export WANDB_DISABLED=true

# Function to print memory status of all GPUs
print_gpu_memory() {
    echo "Current GPU memory usage:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
}

# Function to find an available GPU with sufficient free memory
find_available_gpu() {
    # Threshold: 38 GB free. Idle A100-40GB driver-reserves ~500 MB so a truly
    # free card reports ~40441 MB, never the full 40960. Set just below that so
    # any unoccupied card qualifies, but a card with another non-trivial job
    # (>2 GB used) is correctly skipped. Override via MIN_FREE_MEMORY env var.
    MIN_FREE_MEMORY="${MIN_FREE_MEMORY:-38000}"

    # Print all GPU memory status before selecting
    print_gpu_memory

    # Find a GPU with memory free >= 40GB
    AVAILABLE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
                    awk -v min_mem=$MIN_FREE_MEMORY '$2 >= min_mem {print $1, $2}' | \
                    sort -n -k2 -r | \
                    head -n 1 | \
                    awk '{print $1}')
    
    if [ -z "$AVAILABLE_GPU" ]; then
        echo "No GPU with at least 40 GB of free memory found. Exiting."
        exit 1
    else
        echo "Using GPU $AVAILABLE_GPU with sufficient memory."
    fi
}

# Respect an externally-pinned CUDA_VISIBLE_DEVICES so that parallel pipelines
# (e.g. Qwen on GPU 0 + llama3 on GPU 5) don't both auto-grab the same card.
# Only auto-select when CUDA_VISIBLE_DEVICES is unset OR empty.
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ] || [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    find_available_gpu
    export CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU
else
    echo "Respecting externally-set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (skipping auto-select)."
fi