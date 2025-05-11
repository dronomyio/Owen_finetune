#!/usr/bin/env python3
import os
import argparse
from datasets import load_from_disk
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-0.5B for drone commands")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="Path to the pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/app/data/drone_commands_dataset",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/app/outputs",
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Initial learning rate (after the potential warmup period) to use",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to use LoRA for parameter-efficient fine-tuning",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="Rank of the LoRA decomposition",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="Alpha parameter for LoRA scaling",
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Use 8-bit quantization for training",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Save checkpoint every X updates steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every X updates steps",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use Weights & Biases for logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="qwen2-drone-commands",
        help="W&B project name",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup wandb if needed
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project)
        except ImportError:
            print("W&B not installed, skipping wandb setup")
            args.use_wandb = False
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Load tokenizer
    print(f"Loading tokenizer for {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Define preprocessing function
    def preprocess_function(examples):
        return examples
    
    # Load model with quantization if required
    print(f"Loading model {args.model_name_or_path}")
    if args.use_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # Apply LoRA if required
    if args.use_lora:
        print("Applying LoRA adapters")
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            # Target the attention layers
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        prediction_loss_only=True,
        fp16=True,
        report_to="wandb" if args.use_wandb else "none",
        push_to_hub=False,
    )
    
    # Initialize data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model and tokenizer
    if args.use_lora:
        model.save_pretrained(f"{args.output_dir}/final")
    else:
        trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    
    print(f"Training completed. Model saved to {args.output_dir}/final")

if __name__ == "__main__":
    main()
