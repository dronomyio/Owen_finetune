#!/bin/bash
set -e

echo "Setting up Qwen2-0.5B Fine-tuning for Drone Commands"
echo "===================================================="

# Create project directories
mkdir -p data scripts outputs

# Create the dataset
echo "Step 1: Creating drone commands dataset..."
python3 create_dataset.py

# Check if dataset was created successfully
if [ -d "data/drone_commands_dataset" ]; then
    echo "✅ Dataset created successfully!"
else
    echo "❌ Failed to create dataset. Check for errors."
    exit 1
fi

# Copy script files to scripts directory
echo "Step 2: Setting up training scripts..."
cp train.py scripts/
cp inference.py scripts/
cp preprocess.py scripts/
cp run_finetuning.sh scripts/

# Make sure scripts are executable
chmod +x scripts/*.py scripts/*.sh

echo "Step 3: Setting up Docker environment..."
if ! command -v docker &> /dev/null; then
    echo "⚠️ Docker not found. Please install Docker to continue."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "⚠️ Docker Compose not found. Please install Docker Compose to continue."
    exit 1
fi

echo "✅ Setup complete!"
echo ""
echo "To start fine-tuning, run:"
echo "  docker-compose up"
echo ""
echo "After fine-tuning completes, the model will be saved to the 'outputs' directory."
echo "You can test the model using:"
echo "  docker run --gpus all -it --rm -v $(pwd)/outputs:/app/outputs qwen2-finetune python /app/scripts/inference.py --use_lora"
