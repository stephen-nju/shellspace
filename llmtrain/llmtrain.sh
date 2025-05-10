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
export NCCL_IB_HCA=$NCCL_IB_HCA #腾讯云H800服务器，可以将此参数注释掉
export NCCL_ALGO=Ring

export DS_ENV_FILE=/opt/nas/p/zhubin/code/Llmtrain/.deepspeed_env
export PROJECT_PATH=/opt/nas/p/zhubin/code/Llmtrain/
export HF_HOME=/opt/local/data/

cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export DS_CONFIG_STAGE_3=${PROJECT_PATH}/config/deepspeed/zero_stage3_config.json
export DS_CONFIG_STAGE_2=${PROJECT_PATH}/config/deepspeed/zero_stage2_config.json
export WANDB_PROJECT="Llmtrain"

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

export name
export seed
export dataset
export stage
export lr=2e-5
export epochs=3
export template=qwen
export finetuning_type=full
export batch_size=4
export weight_decay=0
export max_grad_norm=1
export hostfile=/opt/nas/p/zhubin/code/Llmtrain/config/hostfile
export include
export gradient_accumulation_steps=1
export model_name_or_path
export resize_vocab=false
export save_strategy=steps
export save_steps=5000
export save_total_limit=2
export do_train=false
export do_eval=false
export logging_steps=5
export cutoff_len=2048
export warmup_ratio=0.05
#lora fintuning
export lora_rank=32
export lora_target=all
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

## enable_thinking
export enable_thinking=false
export enable_liger_kernel=false
export flash_attn="auto"
options=$(getopt -l "help,do_train,do_eval,stage:,model_name_or_path:,name:,epochs:,lr:,batch_size:,template:,\
finetuning_type:,dataset:,cutoff_len:,include:,resize_vocab:,gradient_accumulation_steps:,eval_dataset:,eval_strategy:,eval_steps:,\
pref_loss:,pref_beta:,simpo_gamma:,ddp_timeout:,neftune_noise_alpha:,hostfile:,weight_decay:,max_grad_norm:,flash_attn:,\
lora_rank:,lora_alpha:,lora_target:,lora_dropout:,loraplus_lr_ratio:,loraplus_lr_embedding:,seed:,enable_thinking:,enable_liger_kernel:,\
save_steps:,save_total_limit:,logging_steps:,warmup_ratio:,save_strategy:" -o "e:l:d:b:n:m:g:" -a -- "$@")

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
		dataset="$1"
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
	--cutoff_len)
		shift
		cutoff_len="$1"
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
		lora_target="$1"
		;;
	--lora_dropout)
		shift
		lora_dropout="$1"
		;;
	--loraplus_lr_ratio)
		shift
		loraplus_lr_ratio="$1"
		;;
	--loraplus_lr_embedding)
		shift
		loraplus_lr_embedding="$1"
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
	--weight_decay)
		shift
		weight_decay="$1"
		;;
	--max_grad_norm)
		shift
		max_grad_norm="$1"
		;;
	--seed)
		shift
		seed="$1"
		;;
	--enable_thinking)
		shift
		enable_thinking="$1"
		;;
	--enable_liger_kernel)
		shift
		enable_liger_kernel="$1"
		;;
	--flash_attn)
		shift
		flash_attn="$1"
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

if [[ -n $loraplus_lr_ratio ]]; then
	optional_params+=(--loraplus_lr_ratio ${loraplus_lr_ratio})
fi

if [[ ${do_eval} = true ]]; then
	optional_params+=(--eval_dataset ${eval_dataset})
	optional_params+=(--eval_strategy ${eval_strategy})
	optional_params+=(--eval_steps ${eval_steps})
fi

deepspeed_params=()
if [[ -n $include ]]; then
	deepspeed_params+=(--include $include)
fi

if [[ $enable_thinking = true ]]; then
	echo ">>>> thinking is enabled, pay attention to your dataset"
fi

export OUTPUT_DIR=/opt/nas/p/zhubin/saved_checkpoint/$name
export WANDB_DIR=$OUTPUT_DIR/logs

mkdir -p ${OUTPUT_DIR}
mkdir -p ${WANDB_DIR}

echo "wandb dir=$WANDB_DIR"
echo "working directory=$(pwd)"

deepspeed --hostfile=$hostfile --master_port=${MASTER_PORT} "${deepspeed_params[@]}" --no_local_rank \
	src/train.py \
	--deepspeed ${DS_CONFIG_STAGE_2} \
	--stage ${stage} \
	--seed ${seed} \
	--run_name $name \
	--template ${template} \
	--do_train ${do_train} \
	--do_eval ${do_eval} \
	--model_name_or_path $model_name_or_path \
	--trust_remote_code true \
	--resize_vocab $resize_vocab \
	--use_fast_tokenizer false \
	--report_to wandb \
	--overwrite_output_dir \
	--overwrite_cache \
	--dataset ${dataset} \
	--cutoff_len ${cutoff_len} \
	--output_dir ${OUTPUT_DIR} \
	--num_train_epochs ${epochs} \
	--overwrite_cache \
	--finetuning_type ${finetuning_type} \
	--lora_rank ${lora_rank} \
	--lora_target ${lora_target} \
	--lora_dropout ${lora_dropout} \
	--lora_alpha ${lora_alpha} \
	--loraplus_lr_embedding ${loraplus_lr_embedding} \
	--warmup_ratio ${warmup_ratio} \
	--logging_steps ${logging_steps} \
	--lr_scheduler_type cosine \
	--per_device_train_batch_size ${batch_size} \
	--per_device_eval_batch_size ${batch_size} \
	--gradient_accumulation_steps ${gradient_accumulation_steps} \
	--weight_decay ${weight_decay} \
	--max_grad_norm ${max_grad_norm} \
	--pref_beta ${pref_beta} \
	--pref_loss ${pref_loss} \
	--simpo_gamma ${simpo_gamma} \
	--preprocessing_num_workers 16 \
	--save_strategy ${save_strategy} \
	--save_steps ${save_steps} \
	--save_total_limit ${save_total_limit} \
	--learning_rate ${lr} \
	--ddp_timeout ${ddp_timeout} \
	--bf16 true \
	--enable_thinking ${enable_thinking} \
	--enable_liger_kernel ${enable_liger_kernel} \
	--flash_attn ${flash_attn} \
	"${optional_params[@]}" \
	2>&1 | tee ${OUTPUT_DIR}/train.log
