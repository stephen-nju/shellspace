#!/usr/bin/bash

Usage() {
	cat <<EOF
Usage:eval dataset
EOF
}

export PROJECT_PATH=/opt/nas/n/zb/code/Llmtrain/
cd ${PROJECT_PATH}
export hoststr='node12 slots=8'
export model_name_or_path
export adapter_name_or_path
export eval_dataset
export finetuning_type=lora
export template=qwen
export output_dir
export output_name=output
export batch_size=4
options=$(getopt -l "help,model_name_or_path:,hoststr:,adapter_name_or_path:,eval_dataset:,finetuning_type:,template:,output_name:,batch_size:
" -o "h:d:t:n:m:g:" -a -- "$@")

eval set -- "$options"
# echo "$options"
while true; do
	case "$1" in
	-h | --help)
		Usage
		exit 0
		;;
	-m | --model_name_or_path)
		shift
		model_name_or_path="$1"
		;;
	--adapter_name_or_path)
		shift
		adapter_name_or_path="$1"
		;;
	-t | --template)
		shift
		template="$1"
		;;
	--finetuning_type)
		shift
		finetuning_type="$1"
		;;
	-d | --eval_dataset)
		shift
		eval_dataset="$1"
		;;
	--hoststr)
		shift
		hoststr="$1"
		;;
	--output_name)
		shift
		output_name="$1"
		;;
	--batch_size)
		shift
		batch_size="$1"
		;;
	--)
		shift
		break
		;;
	esac
	shift
done

optional_params=()
if [[ -z $output_dir ]]; then
	if [[ -n $adapter_name_or_path ]]; then
		output_dir=${adapter_name_or_path}
		optional_params+=(--adapter_name_or_path ${adapter_name_or_path})
	else
		output_dir=${model_name_or_path}
	fi
fi

output_dir=${output_dir}/$output_name

attrun \
	--hoststr="${hoststr}" \
	torchrun \
	--nproc_per_node=\$nproc_per_node --nnodes=\$nnodes --node_rank=\$node_rank --master_addr=\$master_addr \
	src/train.py \
	--stage sft \
	--model_name_or_path ${model_name_or_path} \
	--trust_remote_code true \
	--resize_vocab false \
	--do_predict \
	--eval_dataset ${eval_dataset} \
	--template ${template} \
	--finetuning_type ${finetuning_type} \
	--output_dir ${output_dir} \
	--cutoff_len 4096 \
	--max_new_tokens 512 \
	--do_sample false \
	--per_device_eval_batch_size ${batch_size} \
	--predict_with_generate \
	--overwrite_output_dir true \
	"${optional_params[@]}"

# attrun \
# 	--hoststr="${hoststr}" \
# 	torchrun \
# 	--nproc_per_node=\$nproc_per_node --nnodes=\$nnodes --node_rank=\$node_rank --master_addr=\$master_addr \
# 	src/train.py \
# 	--stage sft \
# 	--model_name_or_path ${model_name_or_path} \
# 	--trust_remote_code true \
# 	--resize_vocab false \
# 	--do_eval \
# 	--eval_dataset ${eval_dataset} \
# 	--template ${template} \
# 	--finetuning_type ${finetuning_type} \
# 	--output_dir ${output_dir} \
# 	--cutoff_len 4096 \
# 	--max_new_tokens 512 \
# 	--do_sample false \
# 	--per_device_eval_batch_size 2 \
# 	--overwrite_output_dir true \
# 	"${optional_params[@]}"
