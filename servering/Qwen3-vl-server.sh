#!/usr/bin/bash

# 使用vllm 部署Qwen3-vl 模型
# 4B的模型
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /opt/tools/resource/easy_resource/model/Qwen3-VL-4B-Instruct/ --served-model-name Qwen3-vl-4B-Instruct --tensor-parallel-size 4 --port 8810 --async-scheduling

#30B的模型

# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /opt/tools/resource/easy_resource/model/Qwen3-VL-30B-A3B-Instruct/ --served-model-name Qwen3-VL-30B-A3B-Instruct \
# 	--max-model-len 128000 --mm-encoder-tp-mode data \
# 	--tensor-parallel-size 4 --port 8810 --async-scheduling --gpu-memory-utilization 0.90
