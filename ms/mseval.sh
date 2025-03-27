#!/usr/bin/bash

Usage() {
	cat <<EOF
Usage:eval dataset
EOF
}

export PROJECT_PATH=/opt/nas/p/zhubin/code/ms-swift
cd ${PROJECT_PATH}
export eval_dataset
export adapters
export finetuning_type
export model
options=$(getopt -l "help,finetuning_type:,adapters:,model:,eval_dataset:" -o "h:" -a -- "$@")

eval set -- "$options"
# echo "$options"
while true; do
	case "$1" in
	-h | --help)
		Usage
		exit 0
		;;
	--finetuning_type)
		shift
		finetuning_type="$1"
		;;
	--adapters)
		shift
		adapters="$1"
		;;
	--model)
		shift
		model="$1"
		;;
	-d | --eval_dataset)
		shift
		eval_dataset="$1"
		;;
	--)
		shift
		break
		;;
	esac
	shift
done

if [[ $finetuning_type == "lora" ]]; then

	NPROC_PER_NODE=8 \
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
	swift infer --adapters ${adapters} \
		--val_dataset $eval_dataset \
		--infer_backend pt \
		--temperature 0 \
		--max_batch_size 32 \
		--max_new_tokens 512 \
		--system 'You are a helpful assistant.'
elif [[ $finetuning_type == "full" ]]; then
	NPROC_PER_NODE=8 \
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
	swift infer --model ${model} \
		--val_dataset $eval_dataset \
		--infer_backend pt \
		--temperature 0 \
		--max_batch_size 32 \
		--max_new_tokens 512 \
		--system 'You are a helpful assistant.'
fi