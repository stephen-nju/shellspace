Usage() {
	cat <<EOF
Usage: train magiclm nano
EOF
}

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=160
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7
export NCCL_ALGO=Ring

export PROJECT_PATH=/home/jovyan/zhubin/code/Llmtrain/
export PYTHONPATH=/home/jovyan/zhubin/DATA/models/honor_2_5b_patched_tokenizer:$PYTHONPATH
cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export DS_CONFIG_STAGE_3=${PROJECT_PATH}/config/deepspeed/zero_stage3_config.json
export DS_CONFIG_STAGE_2=${PROJECT_PATH}/config/deepspeed/zero_stage2_config.json
export WANDB_PROJECT="MagicLM_Nano"

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

export name
export dataset
export stage=rm
export lr=1e-6
export epochs=1
export template=honor
export finetuning_type=full
export batch_size=1
export include="node12"
export gradient_accumulation_steps=1
export model_name_or_path=/home/jovyan/zhubin/code/DataGeneration/reward_model/internlm2-20b-reward-patched/
export adapter_name_or_path
export resize_vocab=true
export save_strategy=steps
export save_steps=10
export save_total_limit=10
export do_train=false
export do_eval=false
export logging_steps=5
export cutoff_len=1024
export warmup_ratio=0.03

#lora fintuning
export lora_rank=32
export lora_alpha=$((2 * ${lora_rank}))
export lora_target=all

#LoRA+
export loraplus_lr_ratio
export loraplus_lr_embedding=1e-6

# dpo parameter
export pref_loss=sigmoid
export pref_beta=0.1
export simpo_gamma=0.5
export ddp_timeout=180000000
# neftune
export neftune_noise_alpha

## eval
export eval_dataset
export eval_steps
export eval_strategy=no
## generate
export max_new_tokens=512
export top_k=0
export top_p=0.9

options=$(getopt -l "help,do_train,do_eval,stage:,model_name_or_path:,adapter_name_or_path:,name:,epochs:,lr:,batch_size:,template:,\
finetuning_type:,dataset:,cutoff_len:,include:,resize_vocab:,gradient_accumulation_steps:,eval_dataset:,eval_strategy:,eval_steps:,\
pref_loss:,pref_beta:,simpo_gamma:,ddp_timeout:,neftune_noise_alpha:,max_new_tokens:,top_p:,top_k:,\
lora_rank:,lora_target:,lora_alpha:,loraplus_lr_embedding:,loraplus_lr_ratio:,\
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
	--adapter_name_or_path)
		shift
		adapter_name_or_path="$1"
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
	--lora_target)
		shift
		lora_target="$1"
		;;
	--lora_alpha)
		shift
		lora_alpha="$1"
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
	--max_new_tokens)
		shift
		max_new_tokens="$1"
		;;
	--top_p)
		shift
		top_p="$1"
		;;
	--top_k)
		shift
		top_k="$1"
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

if [[ $do_eval = true ]]; then
	optional_params+=(--eval_dataset ${eval_dataset})
	optional_params+=(--eval_steps ${eval_steps})
fi
# 判断变量非空
if [[ -n $loraplus_lr_ratio ]]; then
	optional_params+=(--loraplus_lr_ratio ${loraplus_lr_ratio})
fi

if [[ -n ${adapter_name_or_path} ]]; then
	optional_params+=(--adapter_name_or_path ${adapter_name_or_path})
fi

export OUTPUT_DIR=/home/jovyan/zhubin/saved_checkpoint/$name
export WANDB_DIR=$OUTPUT_DIR/logs
export HOSTFILE=/home/jovyan/zhubin/code/Llmtrain/config/hostfile
mkdir -p ${OUTPUT_DIR}
mkdir -p ${WANDB_DIR}

echo "wandb dir=$WANDB_DIR"
wandb offline
export WANDB_MODE=offline
deepspeed --hostfile=${HOSTFILE} --include=${include} --master_port=${MASTER_PORT} --no_local_rank \
	src/train.py \
	--deepspeed ${DS_CONFIG_STAGE_2} \
	--stage ${stage} \
	--run_name ${name} \
	--pref_beta ${pref_beta} \
	--pref_loss ${pref_loss} \
	--simpo_gamma ${simpo_gamma} \
	--template ${template} \
	--do_train ${do_train} \
	--do_eval ${do_eval} \
	--eval_strategy ${eval_strategy} \
	--model_name_or_path $model_name_or_path \
	--resize_vocab true \
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
	--lora_alpha ${lora_alpha} \
	--loraplus_lr_embedding ${loraplus_lr_embedding} \
	--warmup_ratio ${warmup_ratio} \
	--logging_steps ${logging_steps} \
	--lr_scheduler_type cosine \
	--per_device_train_batch_size ${batch_size} \
	--per_device_eval_batch_size ${batch_size} \
	--gradient_accumulation_steps ${gradient_accumulation_steps} \
	--preprocessing_num_workers 16 \
	--save_strategy ${save_strategy} \
	--save_steps ${save_steps} \
	--save_total_limit ${save_total_limit} \
	--learning_rate ${lr} \
	--ddp_timeout ${ddp_timeout} \
	--bf16 true \
	--plot_loss true \
	--max_new_tokens ${max_new_tokens} \
	--top_p ${top_p} \
	--top_k ${top_k} \
	"${optional_params[@]}" \
	2>&1 | tee ${OUTPUT_DIR}/train.log
