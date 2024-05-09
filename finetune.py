import os
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_dir", type=str)
    parser.add_argument("--lora_dir", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    # parser.add_argument("--llamacpp_dir", type=str, default="/data/repos/llama.cpp")
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()

def main():
    args = get_args()

    model_init = args.base_model_dir
    lora_dir = args.lora_dir
    # merged_dir = os.path.join(
    #     lora_dir,
    #     "merged"
    # )
    dataset_dir = args.dataset_dir # "/data/datasets/ruozhiba-llama3"
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
            max_steps = args.max_steps,
            num_train_epochs = args.num_train_epochs,
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

if __name__ == "__main__" :
    main()
