from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from transformers import GenerationConfig
from transformers import AutoTokenizer
import torch
from peft import PeftModel
import torch

import os
from peft import PeftModel
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_dir", type=str)
    parser.add_argument("--lora_dir", type=str)
    parser.add_argument("--llamacpp_dir", type=str, default="/data/repos/llama.cpp")
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()

def main():
    args = get_args()

    base_model_dir = args.base_model_dir
    lora_dir = args.lora_dir 
    merged_dir = os.path.join(
        lora_dir,
        "merged"
    )
    if args.device == 'auto':
        device_arg = { 'device_map': 'auto' }
    else:
        device_arg = { 'device_map': { "": "cpu" }}

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_arg['device_map']
    )

    model = PeftModel.from_pretrained(
        base_model,
        lora_dir,
        device_map={"": torch.device("cpu")}
    )

    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_dir,
        device_map="cpu"
    )

    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Model saved to {merged_dir}")

    cmd = f"bash ./quan_gguf.sh --llama_cpp_dir {args.llamacpp_dir} --model_dir {merged_dir}"
    os.system(cmd)

if __name__ == "__main__" :
    main()
