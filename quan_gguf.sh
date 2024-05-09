#!/bin/bash

declare -A args

# 循环通过所有位置参数
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --llama_cpp_dir)
        args[llama_cpp_dir]="$2"
        shift # 过去参数名
        shift # 过去参数值
        ;;
        --model_dir)
        args[model_dir]="$2"
        shift # 过去参数名
        shift # 过去参数值
        ;;
        *)    # 如果遇到未知选项
        shift # 就单独过去参数
        ;;
    esac
done

echo ${args[llama_cpp_dir]}
echo ${args[model_dir]}


python ${args[llama_cpp_dir]}/convert-hf-to-gguf.py \
    ${args[model_dir]} \
    --outfile ${args[model_dir]}/bf16.gguf 

${args[llama_cpp_dir]}/quantize \
    ${args[model_dir]}/bf16.gguf \
    ${args[model_dir]}/Q4_K_M.gguf \
    Q4_K_M