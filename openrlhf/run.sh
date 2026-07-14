#!/usr/bin/bash

export DATE=$(date "+%m%d")
echo "training scripts date ${DATE}"
export dataset=/opt/nas/p/mmu/zb/DATA/raw_data/hf_data/OpenRLHF/preference_dataset_mixture2_and_safe_pku/

./openrlhf_train.sh --do_train --do_eval --algorithm_estimator dpo \
	--name="${DATE}_Qwen3-4B_test_ep2_lr2e5" \
	--model_name_or_path /opt/tools/resource/easy_resource/model/Qwen3-4B \
	--dataset $dataset
