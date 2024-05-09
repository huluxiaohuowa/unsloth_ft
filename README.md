# Use unsloth to train llama3 and quantize the model.

## Finetune
```bash
python finetune.py \
    --base_model_dir /data/models/llama-3-8b-instruct \
    --lora_dir /data/models/llama3_lora3 \
    --dataset_dir /data/datasets/ruozhiba-llama3 \
    --num_train_epochs 5
```

## Merge & quantize
```bash
python merge.py \
    --base_model_dir /data/models/llama-3-8b-instruct \
    --lora_dir /data/models/llama3_lora3 \
    --llamacpp_dir /data/repos/llama.cpp
```