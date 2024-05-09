import os
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

model_init = "/data/models/llama-3-8b-instruct/" 
lora_dir = "/data/models/llama3_lora3"
merged_dir = os.path.join(
    lora_dir,
    "merged"
)
dataset_dir = "/data/datasets/ruozhiba-llama3"
tmp_out = os.path.join(
    lora_dir,
    "output"
)



#加载模型
max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_init, 
    max_seq_length = max_seq_length, 
    dtype = dtype,     
    load_in_4bit = load_in_4bit,  
)

#准备训练数据
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # 必须添加 EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # 必须添加EOS_TOKEN，否则无限生成
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

#hugging face数据集路径
dataset = load_dataset(dataset_dir, split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

#设置训练参数
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  
    loftq_config = None, 
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 20,
        # num_train_epochs = 1,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = tmp_out,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
    ),
)
#开始训练
print("Traing...")
trainer.train()

#保存微调模型
print(f"Saving lora to {lora_dir}")
model.to(torch.bfloat16)
model.save_pretrained(lora_dir) 

#合并模型，保存为16位hf
# print(f"Merging lora to {merged_dir}")
# model.save_pretrained_merged(
#     merged_dir,
#     tokenizer,
#     save_method = "merged_16bit",
# )


#合并模型，并量化成4位gguf
# model.save_pretrained_gguf(
#     "/data/models/llama3_lora/merged/gguf",
#     tokenizer,
#     quantization_method = "q4_k_m"
# )

# cpu合并
# python merge_peft.py \
#     --base_model_name_or_path="/home/jhu/dev/models/Llama-2-7b-chat-hf" \
#     --peft_model_path="/home/jhu/dev/models/llama2-7b-journal-finetune/checkpoint-500" \
#     --output_dir="/home/jhu/dev/models/llama2-7b-chat-merged"

# python convert-hf-to-gguf.py \
    # /data/models/llama3_lora/merged/ \
    # --outfile /data/models/llama3_lora/merged/llama3_ft_01.gguf

# ./quantize \
    # /data/models/llama3_lora/merged/llama3_ft_01.gguf \
    # /data/models/llama3_lora/merged/llama3_ft_Q4_K_M.gguf \
    # Q4_K_M

