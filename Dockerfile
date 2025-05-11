FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    transformers==4.38.0 \
    datasets==2.16.0 \
    accelerate==0.26.0 \
    bitsandbytes==0.42.0 \
    evaluate==0.4.1 \
    peft==0.6.0 \
    deepspeed==0.12.3 \
    tensorboard==2.15.1 \
    wandb==0.16.2 \
    sentencepiece==0.1.99 \
    huggingface-hub==0.20.2

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Copy training script and data
COPY scripts/ /app/scripts/
COPY data/ /app/data/

# Make scripts executable
RUN chmod +x /app/scripts/*.sh

# Default command
CMD ["/bin/bash", "/app/scripts/run_finetuning.sh"]
