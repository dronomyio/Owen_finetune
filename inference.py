#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Qwen2-0.5B")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/app/outputs/final",
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="Base model path or identifier",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether the model was fine-tuned with LoRA",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum length for generated text",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for text generation",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    except:
        print(f"Fallback to base model tokenizer: {args.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Load model
    print(f"Loading model")
    if args.use_lora:
        print(f"Loading base model: {args.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"Loading LoRA adapters from {args.model_dir}")
        model = PeftModel.from_pretrained(model, args.model_dir)
    else:
        print(f"Loading full fine-tuned model from {args.model_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Chat function
    def chat():
        print("\n=== Drone Command Interface ===")
        print("Type 'exit' to quit the chat")
        print("Enter a drone command instruction:")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # Format the input for Qwen2
            input_text = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    do_sample=(args.temperature > 0.1),
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract the assistant's response
            try:
                assistant_response = response.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0]
                print(f"\nDrone: {assistant_response}")
            except IndexError:
                print(f"\nDrone: {response}")
    
    # Start interactive chat
    chat()

if __name__ == "__main__":
    main()
