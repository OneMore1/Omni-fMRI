#!/bin/bash

# Configuration
CONFIG_FILE=""
OUTPUT_DIR=""
NUM_GPUS=1  # Number of GPUs to use
MASTER_PORT=2955

# Optional: Resume from checkpoint
RESUME=""  # Set to checkpoint path to resume training

echo "Starting DDP training with $NUM_GPUS GPUs..."
echo "Config: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"

# Set environment variables
export CUDA_VISIBLE_DEVICES=3 # Adjust based on available GPUs
export OMP_NUM_THREADS=1

# Launch training with torchrun (recommended for PyTorch >= 1.10)
if [ -z "$RESUME" ]; then
    # Start from scratch
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        pretrain.py\
        --config $CONFIG_FILE \
        --output_dir $OUTPUT_DIR
else
    # Resume from checkpoint
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        pretrain.py \
        --config $CONFIG_FILE \
        --output_dir $OUTPUT_DIR \
        --resume $RESUME
fi

echo "Training completed!"

