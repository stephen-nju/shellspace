#!/usr/bin/bash

# 使用vllm 部署Qwen3-vl 模型
# 4B的模型
CUDA_VISIBLE_DEVICES=4 vllm serve /opt/nas/n/mmu/zhubin/DATA/models/anydoor_llm_0402 --served-model-name Qwen2-7B-Instruct --tensor-parallel-size 1 --port 8810 --max-model-len 32768

#30B的模型

# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /opt/tools/resource/easy_resource/model/Qwen3-VL-30B-A3B-Instruct/ --served-model-name Qwen3-VL-30B-A3B-Instruct \
# 	--max-model-len 128000 --mm-encoder-tp-mode data \
# 	--tensor-parallel-size 4 --port 8810 --async-scheduling --gpu-memory-utilization 0.90
