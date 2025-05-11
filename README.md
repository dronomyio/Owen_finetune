# Qwen2-0.5B Fine-tuning for Drone Commands

This repository contains Docker setup for fine-tuning the Qwen2-0.5B model on drone command data to create a specialized model that can process natural language instructions and convert them to drone control commands.

## Project Structure

```
.
├── Dockerfile
├── docker-compose.yml
├── data/
│   └── drone_commands_raw.json     # Your existing data
├── scripts/
│   ├── train.py                    # Main training script
│   ├── preprocess.py               # Data preprocessing
│   ├── inference.py                # Testing the model
│   └── run_finetuning.sh           # Main entrypoint
└── outputs/                        # Will contain fine-tuned model (created during training)
```

## Prerequisites

- NVIDIA GPU with at least 8GB of VRAM
- Docker and Docker Compose
- NVIDIA Container Toolkit installed

## Setup

1. Clone this repository
2. Place your drone command dataset in the `data/` directory (or use the included script)
3. Build and run the Docker container

## Quick Start

### 1. Generate the Dataset

If you have the `paste.txt` file in the data directory, you can generate the dataset with:

```bash
mkdir -p data
python3 data/paste.txt > data/drone_commands_raw.json
```

### 2. Start Fine-tuning

```bash
docker-compose up
```

This will:
1. Build the Docker image
2. Process your data into the appropriate format
3. Fine-tune the Qwen2-0.5B model using efficient methods (LoRA and 8-bit quantization)
4. Save the resulting model to the `outputs/` directory

## Configuration

You can customize the fine-tuning process by editing environment variables in the `docker-compose.yml` file:

- `FINE_TUNING_EPOCHS`: Number of training epochs (default: 3)
- `BATCH_SIZE`: Batch size per GPU (default: 4)
- `GRAD_ACCUM_STEPS`: Gradient accumulation steps (default: 4)
- `LEARNING_RATE`: Learning rate (default: 2e-4)
- `USE_LORA_TRAINING`: Whether to use LoRA for efficient fine-tuning (default: true)
- `USE_8BIT_QUANTIZATION`: Whether to use 8-bit quantization (default: true)
- `WANDB_API_KEY`: Optional Weights & Biases API key for experiment tracking

## Testing the Fine-tuned Model

After fine-tuning completes, you can test your model using the provided inference script:

```bash
# For LoRA fine-tuned model
docker run --gpus all -it --rm \
  -v $(pwd)/outputs:/app/outputs \
  qwen2-finetune \
  python /app/scripts/inference.py --use_lora

# For full fine-tuned model
docker run --gpus all -it --rm \
  -v $(pwd)/outputs:/app/outputs \
  qwen2-finetune \
  python /app/scripts/inference.py
```

## Advanced Usage

### Fine-tuning on Custom Data

To fine-tune on your own dataset, create a JSON file with the following structure:

```json
[
  {
    "instruction": "Your instruction here",
    "output": "Expected model output"
  },
  ...
]
```

### Exporting for Inference

After fine-tuning, you can export the model for inference:

```bash
# Convert LoRA model to full model (optional)
docker run --gpus all -it --rm \
  -v $(pwd)/outputs:/app/outputs \
  qwen2-finetune \
  python -c "from peft import PeftModel; from transformers import AutoModelForCausalLM; model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B', device_map='auto', trust_remote_code=True); model = PeftModel.from_pretrained(model, '/app/outputs/final'); model.save_pretrained('/app/outputs/merged');"
```

## Performance Considerations

- **Memory Requirements**: With 8-bit quantization and LoRA, this should run on GPUs with 8GB VRAM.
- **Training Time**: On an NVIDIA RTX 3090, expect 1-3 hours for full training.

## Troubleshooting

- **Out of Memory Errors**: Reduce batch size or enable gradient accumulation
- **Slow Training**: Make sure your GPU is properly recognized by the container
- **Data Formatting Issues**: Check the format of your instruction dataset

## License

This project is provided for educational purposes. The Qwen2 model is subject to its own license terms.
# Owen_finetune
