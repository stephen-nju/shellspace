#!/usr/bin/bash

# 使用vllm 部署Qwen3-vl 模型
# 4B的模型
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve /opt/tools/resource/easy_resource/model/Qwen3-4B-Instruct-2507 --served-model-name Qwen3-4B-Instruct-2507 --tensor-parallel-size 4 --port 8810 --max-model-len 32768

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /opt/tools/resource/easy_resource/model/Qwen3-32B/ --served-model-name Qwen3-32B --tensor-parallel-size 8 --port 8810 --max-model-len 32768 \
# 	--gpu-memory-utilization 0.90 --reasoning-parser deepseek_r1
#30B的模型

# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /opt/tools/resource/easy_resource/model/Qwen3-VL-30B-A3B-Instruct/ --served-model-name Qwen3-VL-30B-A3B-Instruct \
# 	--max-model-len 128000 --mm-encoder-tp-mode data \

# 	--tensor-parallel-size 4 --port 8810 --async-scheduling --gpu-memory-utilization 0.90

# CUDA_VISIBLE_DEVICES=0,1 vllm serve /opt/nas/n/models/Qwen/Qwen3-Embedding-8B/ --served-model-name Qwen3-Embedding-8B \
# 	--tensor-parallel-size 2 --port 8810 --gpu-memory-utilization 0.90
# 	--max-model-len 128000 --mm-encoder-tp-mode data \

# 使用sglang部署

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python -m sglang.launch_server --model-path /opt/nas/n/models/Qwen/Qwen3-32B --reasoning-parser qwen3 --tensor-parallel-size 8 --port 8810 --disable-overlap-schedule \
# 	--attention-backend fa3 --sampling-backend pytorch --chunked-prefill-size 2048 --max-running-requests 8 --mem-fraction-static 0.7 \
# 	--dtype bfloat16 --enable-fp32-lm-head

# # 使用tool
# CUDA_VISIBLE_DEVICES=6 vllm serve /opt/tools/resource/easy_resource/model/Qwen3-8B/ --served-model-name Qwen3-8B --port 8810 --max-model-len 32768 \
# 	--enable-auto-tool-choice --tool-call-parser hermes
