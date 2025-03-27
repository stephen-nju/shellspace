#!/bin/bash

CUDA_VISIBLE_DEVICES=4 API_PORT=7713 API_MODEL_NAME="magiclm-3B" python src/api.py \
	--model_name_or_path /opt/nas/p/zhubin/DATA/models/MagicLM-3B-Instruct-v0.1/ \
	--template honor \
	--trust_remote_code true \
	--finetuning_type full
