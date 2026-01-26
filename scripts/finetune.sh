
# Set environment variables
export CUDA_VISIBLE_DEVICES=7  
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Configuration
CONFIG_FILE=""
NUM_GPUS=1  
MASTER_PORT=2953

# Optional: Output directory
OUTPUT_DIR=""

echo "Starting DDP fine-tuning with $NUM_GPUS GPUs..."
echo "Config: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"

if [ -z "$RESUME_CHECKPOINT" ]; then
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        finetune.py\
        --config $CONFIG_FILE \
        --output_dir $OUTPUT_DIR
else
    # Resume from checkpoint
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        finetune.py \
        --config $CONFIG_FILE \
        --output_dir $OUTPUT_DIR \
        --resume $RESUME_CHECKPOINT
fi

echo "Fine-tuning completed!"

