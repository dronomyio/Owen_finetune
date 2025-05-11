#!/bin/bash
set -e

# Check if NVIDIA GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA GPU not detected. This container requires GPU support."
    exit 1
fi

# Create output directory
mkdir -p /app/outputs

# Define default parameters
USE_LORA=true
USE_8BIT=true
EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=4
LEARNING_RATE=2e-4
MODEL_NAME="Qwen/Qwen2-0.5B"
SAVE_STEPS=200

# Parse environment variables if provided
if [ -n "$FINE_TUNING_EPOCHS" ]; then
    EPOCHS=$FINE_TUNING_EPOCHS
fi

if [ -n "$BATCH_SIZE" ]; then
    BATCH_SIZE=$BATCH_SIZE
fi

if [ -n "$GRAD_ACCUM_STEPS" ]; then
    GRAD_ACCUM=$GRAD_ACCUM_STEPS
fi

if [ -n "$LEARNING_RATE" ]; then
    LEARNING_RATE=$LEARNING_RATE
fi

if [ -n "$MODEL_PATH" ]; then
    MODEL_NAME=$MODEL_PATH
fi

if [ "${USE_LORA_TRAINING:-true}" = "false" ]; then
    USE_LORA=false
fi

if [ "${USE_8BIT_QUANTIZATION:-true}" = "false" ]; then
    USE_8BIT=false
fi

# Prepare command
CMD="python /app/scripts/train.py \
    --model_name_or_path $MODEL_NAME \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $EPOCHS \
    --save_steps $SAVE_STEPS"

# Add LoRA if enabled
if [ "$USE_LORA" = true ]; then
    CMD="$CMD --use_lora"
fi

# Add 8-bit quantization if enabled
if [ "$USE_8BIT" = true ]; then
    CMD="$CMD --use_8bit"
fi

# Add W&B if credentials exist
if [ -n "$WANDB_API_KEY" ]; then
    CMD="$CMD --use_wandb"
fi

echo "Starting fine-tuning with command:"
echo "$CMD"

# Execute the command
eval $CMD
