from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from transformers import GenerationConfig
from transformers import AutoTokenizer
import torch
from peft import PeftModel
import torch

import os
from peft import PeftModel

base_model_path = "/data/models/llama-3-8b-instruct/"
lora_dir = "/data/models/llama3_lora3"
merged_dir = os.path.join(
    lora_dir,
    "merged"
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="cpu"
)

model = PeftModel.from_pretrained(
    base_model,
    lora_dir,
    device_map={"": torch.device("cpu")}
)

model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    device_map="cpu"
)

model.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)
print(f"Model saved to {merged_dir}")