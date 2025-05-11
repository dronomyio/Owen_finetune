#!/usr/bin/env python3
import json
import os
import argparse
from datasets import Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess drone commands dataset for fine-tuning")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/app/data/drone_commands_raw.json",
        help="Path to raw drone commands JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/app/data/drone_commands_dataset",
        help="Directory to save processed dataset",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="qwen2",
        choices=["qwen2", "mistral", "llama"],
        help="Format template to use (default: qwen2)",
    )
    return parser.parse_args()

def format_qwen2_instruction(item):
    """Format data for Qwen2 instruction format"""
    instruction = item["instruction"]
    output = item["output"]
    return {
        "text": f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    }

def format_mistral_instruction(item):
    """Format data for Mistral instruction format"""
    instruction = item["instruction"]
    output = item["output"]
    return {
        "text": f"<s>[INST] {instruction} [/INST] {output}</s>"
    }

def format_llama_instruction(item):
    """Format data for Llama instruction format"""
    instruction = item["instruction"]
    output = item["output"]
    return {
        "text": f"<s>[INST] {instruction} [/INST] {output}</s>"
    }

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load raw data
    print(f"Loading raw data from {args.input_file}")
    with open(args.input_file, "r") as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data)} examples")
    
    # Format data based on selected format
    if args.format == "qwen2":
        format_func = format_qwen2_instruction
    elif args.format == "mistral":
        format_func = format_mistral_instruction
    elif args.format == "llama":
        format_func = format_llama_instruction
    else:
        raise ValueError(f"Unknown format: {args.format}")
    
    formatted_data = [format_func(item) for item in raw_data]
    
    # Create HuggingFace dataset
    dataset = Dataset.from_dict({"text": [item["text"] for item in formatted_data]})
    
    # Save the processed dataset
    dataset.save_to_disk(args.output_dir)
    
    print(f"Dataset saved to {args.output_dir}")
    print(f"Sample formatted example: \n{formatted_data[0]['text']}")

if __name__ == "__main__":
    main()
