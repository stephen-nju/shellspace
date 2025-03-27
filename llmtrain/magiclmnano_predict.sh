Usage() {
	cat <<EOF
Usage:eval dataset
EOF
}

export PROJECT_PATH=/opt/nas/p/zhubin/code/LLaMA-Factory/
cd ${PROJECT_PATH}
export hostfile=/opt/nas/p/zhubin/code/LLaMA-Factory/config/hostfile
export include="nlp-nlp-sum-0"
export model_name_or_path=/opt/nas/p/zhubin/DATA/models/honor2_5b_patched_tokenizer/
export adapter_name_or_path
export eval_dataset
export finetuning_type=lora
export template=honor
export temperature=2
export output_dir
wandb offline

options=$(getopt -l "help,model_name_or_path:,hoststr:,adapter_name_or_path:,eval_dataset:,finetuning_type:,template:,output_dir:,temperature:,include:" -o "h:d:t:n:m:g:" -a -- "$@")

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
	--predict_with_generate)
		shift
		predict_with_generate="$1"
		;;
	--temperature)
		shift
		temperature="$1"
		;;
	--output_dir)
		shift
		output_dir="$1"
		;;
	--include)
		shift
		include="$1"
		;;
	--)
		shift
		break
		;;
	esac
	shift
done

attrun \
	--hostfile ${hostfile} \
	--include "${include}" \
	torchrun \
	--nproc_per_node=\$nproc_per_node --nnodes=\$nnodes --node_rank=\$node_rank --master_addr=\$master_addr \
	src/train.py \
	--stage sft \
	--model_name_or_path ${model_name_or_path} \
	--adapter_name_or_path ${adapter_name_or_path} \
	--resize_vocab true \
	--do_predict \
	--eval_dataset ${eval_dataset} \
	--template ${template} \
	--finetuning_type ${finetuning_type} \
	--output_dir ${output_dir} \
	--cutoff_len 4069 \
	--max_new_tokens 512 \
	--do_sample true \
	--temperature ${temperature} \
	--per_device_eval_batch_size 4 \
	--predict_with_generate
