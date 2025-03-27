#!/usr/bin/bash
Usage() {
	cat <<EOF
Usage: train magiclm nano
-m --model_name_or_path     base model name or path
-n  --name 				    runing experiment name
-h  --help                  display help
-e  --epoch                 num train epochs
-l  --lr					learning rate
-b  --bs					train batch size
-d  --dataset               train dataset
EOF
}

export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export NCCL_IB_DISABLE=$NCCL_IB_DISABLE
export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX
export NCCL_IB_TC=160
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=$NCCL_IB_HCA  #腾讯云H800服务器，可以将此参数注释掉
export NCCL_ALGO=Ring

export DS_ENV_FILE=/opt/nas/p/zhubin/code/Llmtrain/.deepspeed_env
export PROJECT_PATH=/opt/nas/p/zhubin/code/ms-swift
export HF_HOME=/opt/local/data/

cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export DS_CONFIG_STAGE_3=${PROJECT_PATH}/config/deepspeed/zero_stage3_config.json
export DS_CONFIG_STAGE_2=${PROJECT_PATH}/config/deepspeed/zero_stage2_config.json
export WANDB_PROJECT="ms-swift"

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

export name
export dataset=()
export stage
export lr=2e-5
export epochs=3
export template
export finetuning_type=lora
export batch_size=4
export hostfile=/opt/nas/p/zhubin/code/Llmtrain/config/hostfile
export include
export gradient_accumulation_steps=1
export model_name_or_path=/opt/nas/p/zhubin/DATA/models/honor2_5b_patched_tokenizer/
export resize_vocab=false
export save_strategy=steps
export save_steps=5000
export save_total_limit=2
export do_train=false
export do_eval=false
export logging_steps=5
export max_length=2048
export warmup_ratio=0.03
#lora fintuning
export lora_rank=32
export lora_target=()
export lora_alpha=$((2 * ${lora_rank}))
export lora_dropout=0.0
#LoRA+
export loraplus_lr_ratio
export loraplus_lr_embedding=1e-6
# dpo parameter
export pref_loss=simpo
export pref_beta=0.1
export simpo_gamma=0.5
export ddp_timeout=180000000
# neftune
export neftune_noise_alpha
## eval
export eval_dataset
export eval_steps
export eval_strategy=no

options=$(getopt -l "help,do_train,do_eval,stage:,model_name_or_path:,name:,epochs:,lr:,batch_size:,template:,\
finetuning_type:,dataset:,max_length:,include:,resize_vocab:,gradient_accumulation_steps:,eval_dataset:,eval_strategy:,eval_steps:,\
pref_loss:,pref_beta:,simpo_gamma:,ddp_timeout:,neftune_noise_alpha:,hostfile:,\
lora_rank:,lora_alpha:,lora_target:,lora_dropout:,loraplus_lr_ratio:,loraplus_lr_embedding:,\
save_steps:,save_total_limit:,logging_steps:,warmup_ratio:,save_strategy:" -o "e:l:d:b:n:m:g:" -a --n "$0" -- "$@")

eval set -- "$options"
# echo "$options"

while true; do
	case "$1" in
	-h | --help)
		Usage
		exit 0
		;;
	--do_train)
		do_train=true
		;;
	--do_eval)
		do_eval=true
		;;
	-m | --model_name_or_path)
		shift
		model_name_or_path="$1"
		;;
	--stage)
		shift
		stage="$1"
		;;
	-e | --epochs)
		shift
		epochs="$1"
		;;
	-l | --lr)
		shift
		lr="$1"
		;;
	-b | --batch_size)
		shift
		batch_size="$1"
		;;
	-t | --template)
		shift
		template="$1"
		;;
	--finetuning_type)
		shift
		finetuning_type="$1"
		;;
	-g | --gradient_accumulation_steps)
		shift
		gradient_accumulation_steps="$1"
		;;
	-d | --dataset)
		shift
		IFS=',' read -ra DATA_ARRAY <<<"$1"
		dataset+=("${DATA_ARRAY[@]}")
		;;
	-n | --name)
		shift
		name="$1"
		;;
	--hostfile)
		shift
		hostfile="$1"
		;;
	--include)
		shift
		include="$1"
		;;
	--save_total_limit)
		shift
		save_total_limit="$1"
		;;
	--save_strategy)
		shift
		save_strategy="$1"
		;;
	--save_steps)
		shift
		save_steps="$1"
		;;
	--logging_steps)
		shift
		logging_steps="$1"
		;;
	--max_length)
		shift
		max_length="$1"
		;;
	--warmup_ratio)
		shift
		warmup_ratio="$1"
		;;
	--pref_loss)
		shift
		pref_loss="$1"
		;;
	--pref_beta)
		shift
		pref_beta="$1"
		;;
	--simpo_gamma)
		shift
		simpo_gamma="$1"
		;;
	--neftune_noise_alpha)
		shift
		neftune_noise_alpha="$1"
		;;
	--resize_vocab)
		shift
		resize_vocab="$1"
		;;
	--lora_rank)
		shift
		lora_rank="$1"
		;;
	--lora_alpha)
		shift
		lora_alpha="$1"
		;;
	--lora_target)
		shift
		IFS=',' read -ra ARRAY <<<"$1"
		lora_target+=("${ARRAY[@]}")
		;;
	--lora_dropout)
		shift
		lora_dropout="$1"
		;;
	--loraplus_lr_ratio)
		shift
		lorap_lr_ratio="$1"
		;;
	--loraplus_lr_embedding)
		shift
		lorap_emb_lr="$1"
		;;
	--eval_dataset)
		shift
		eval_dataset="$1"
		;;
	--eval_strategy)
		shift
		eval_strategy="$1"
		;;
	--eval_steps)
		shift
		eval_steps="$1"
		;;
	--)
		shift
		break
		;;
	esac
	shift
done

optional_params=()
# 处理python脚本中default等于None的选项
if [[ ! $neftune_noise_alpha ]]; then
	echo "optinonal paramas neftune_noise_alpha is null"
else
	optional_params+=(--neftune_noise_alpha ${neftune_noise_alpha})
fi

if [[ -n $lorap_lr_ratio ]]; then
	optional_params+=(--lorap_lr_ratio ${lorap_lr_ratio})
	if [[ -n $lorap_emb_lr ]]; then
		optional_params+=(--lorap_emb_lr $lorap_emb_lr)
	fi
fi

if [[ $do_eval == true ]]; then
	optional_params+=(--val_dataset ${eval_dataset})
	optional_params+=(--eval_steps ${eval_steps})
fi

if [[ -n $template ]];then
	optional_params+=(--template $template)
fi


export OUTPUT_DIR=/opt/nas/p/zhubin/saved_checkpoint/$name

mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "working directory=$(pwd)"
echo "${dataset[@]}"

if [[ $finetuning_type == "lora" ]]; then
	nproc_per_node=8
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
		NPROC_PER_NODE=$nproc_per_node \
		swift sft \
		--do_train ${do_train} \
		--do_eval ${do_eval} \
		--model ${model_name_or_path} \
		--train_type ${finetuning_type} \
		--dataset "${dataset[@]}" \
		--num_train_epochs ${epochs} \
		--per_device_train_batch_size ${batch_size} \
		--learning_rate ${lr} \
		--lora_rank ${lora_rank} \
		--lora_alpha ${lora_alpha} \
		--target_modules "${lora_target[@]}" \
		--max_length ${max_length} \
		--warmup_ratio ${warmup_ratio} \
		--output_dir ${OUTPUT_DIR} \
		--gradient_accumulation_steps ${gradient_accumulation_steps} \
		--save_strategy ${save_strategy} \
		--save_steps ${save_steps} \
		--save_total_limit ${save_total_limit} \
		--logging_steps 5 \
		--torch_dtype bfloat16 \
		"${optional_params[@]}" \
		--system 'You are a helpful assistant.' \
		--gradient_checkpointing_kwargs '{"use_reentrant": false}'
elif [[ $finetuning_type == "full" ]]; then
	nproc_per_node=8
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
		NPROC_PER_NODE=$nproc_per_node \
		swift sft \
		--do_train ${do_train} \
		--do_eval ${do_eval} \
		--model ${model_name_or_path} \
		--train_type ${finetuning_type} \
		--dataset "${dataset[@]}" \
		--num_train_epochs ${epochs} \
		--per_device_train_batch_size ${batch_size} \
		--learning_rate ${lr} \
		--max_length ${max_length} \
		--warmup_ratio ${warmup_ratio} \
		--output_dir ${OUTPUT_DIR} \
		--gradient_accumulation_steps ${gradient_accumulation_steps} \
		--save_strategy ${save_strategy} \
		--save_steps ${save_steps} \
		--save_total_limit ${save_total_limit} \
		--logging_steps 5 \
		--torch_dtype bfloat16 \
		"${optional_params[@]}" \
		--system 'You are a helpful assistant.' \
		--gradient_checkpointing_kwargs '{"use_reentrant": false}'
fi