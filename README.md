# Qwen2-0.5B Fine-tuning for Drone Commands

This repository contains a complete Docker setup for fine-tuning the Qwen2-0.5B model on drone command data. The model will learn to process natural language instructions and convert them to drone control commands.

## Features

- Automatic dataset generation from drone command templates
- Parameter-efficient fine-tuning using LoRA
- Memory-efficient training with 8-bit quantization
- GPU acceleration with NVIDIA Docker
- Interactive testing interface

## Prerequisites

- NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- Docker and Docker Compose
- NVIDIA Container Toolkit installed

## Quick Start

1. **Setup the project**

```bash
# Clone this repository
git clone <repository-url>
cd qwen2-drone-finetune

# Run the setup script
chmod +x setup.sh
./setup.sh
```

2. **Start fine-tuning**

```bash
docker-compose up
```

3. **Test the fine-tuned model**

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/outputs:/app/outputs \
  qwen2-finetune \
  python /app/scripts/inference.py --use_lora
```

## Dataset

The dataset is automatically generated using patterns from drone commands. It includes:

- Movement commands (forward, backward, left, right)
- Rotation commands (turn, yaw, rotate)
- Camera controls (take photo, record video)
- Navigation commands (return to home, land)
- System commands (check battery, status)
- Emergency procedures

Each command is formatted as an instruction pair:
- Input: Natural language command (e.g., "Move forward 5 meters")
- Output: Corresponding PX4 drone command (e.g., "PX4_CMD: navigator goto -n 5 0 0")

## Configuration

You can customize the fine-tuning process by editing the environment variables in `docker-compose.yml`:

```yaml
environment:
  - FINE_TUNING_EPOCHS=3       # Number of training epochs
  - BATCH_SIZE=4               # Batch size per GPU
  - GRAD_ACCUM_STEPS=4         # Gradient accumulation steps
  - LEARNING_RATE=2e-4         # Learning rate
  - USE_LORA_TRAINING=true     # Use LoRA for efficient fine-tuning
  - USE_8BIT_QUANTIZATION=true # Use 8-bit quantization
```

## Project Structure

```
.
├── Dockerfile              # Container definition
├── docker-compose.yml      # Docker Compose configuration
├── create_dataset.py       # Dataset creation script
├── setup.sh                # Setup script
├── scripts/                # Training and inference scripts
├── data/                   # Contains dataset
└── outputs/                # Will contain fine-tuned model
```

## Advanced Usage

### Custom Dataset

You can modify the dataset generation in `create_dataset.py` to include additional drone commands or different patterns.

### Exporting for Production

After fine-tuning, you can merge the LoRA adapters with the base model for deployment:

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/outputs:/app/outputs \
  qwen2-finetune \
  python -c "from peft import PeftModel; from transformers import AutoModelForCausalLM; model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B', device_map='auto', trust_remote_code=True); model = PeftModel.from_pretrained(model, '/app/outputs/final'); model.save_pretrained('/app/outputs/merged');"
```

## Troubleshooting

- **CUDA Out of Memory**: Reduce batch size or enable gradient accumulation
- **Container not detecting GPU**: Ensure NVIDIA Container Toolkit is installed correctly
- **Training too slow**: Try enabling DeepSpeed by adding `--deepspeed ds_config.json` to the training command

## License

This project is provided for educational purposes. The Qwen2 model is subject to its own license terms.
