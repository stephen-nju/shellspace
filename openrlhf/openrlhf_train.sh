Usage() {
	cat <<EOF
Usage: train ppo 
-m --model_name_or_path     base model name or path
-n  --name 				    runing experiment name
-h  --help                  display help
-e  --epoch                 num train epochs
-l  --lr					learning rate
-b  --bs					train batch size
-d  --dataset               train dataset
EOF
}

#环境变量配置

export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export NCCL_IB_DISABLE=$NCCL_IB_DISABLE
export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX
export NCCL_IB_TC=160
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=$NCCL_IB_HCA #腾讯云H800服务器，可以将此参数注释掉
export NCCL_ALGO=Ring
export MKL_THREADING_LAYER=GNU
export HYDRA_FULL_ERROR=1
export RAY_IGNORE_VERSION_MISMATCH=True
export PROJECT_PATH=/opt/nas/p/mmu/zb/code/OpenRLHF/
export HF_HOME=/opt/local/data/

cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export WANDB_PROJECT="OpenRLHF_trian"

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

export module
export name
export do_train
export do_eval
export dataset
export eval_dataset
export model_name_or_path
export lr=2e-5
export batch_size=128
export epochs=5
export ppo_mini_batch_size=64
export ppo_micro_batch_size_per_gpu=8
export max_prompt_length=512
export max_response_length=1024
export algorithm_estimator=dpo
export rollout_n=4

#dpo 配置
export dpo_beta=0.1
## 机器配置
export include
export ds_zero_stage=2
export nnodes=1
export n_gpus_per_node=8
export hostfile=/opt/nas/p/mmu/zb/code/OpenRLHF/config/hostfile

LONG_OPTS="help,do_train,do_eval,name:,dataset:,eval_dataset:,epochs:,algorithm_estimator:,model_name_or_path:,lr:,batch_size:,rollout_n:,\
ppo_mini_batch_size:,ppo_micro_batch_size_per_gpu:,reward_model:,max_prompt_length:,max_response_length:,\
dpo_beta:,\
include:,ds_zero_stage:,nnodes:,n_gpus_per_node:,"

options=$(getopt -l "${LONG_OPTS}" -o "h:" -a -- "$@")
eval set -- "$options"

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
	--lr)
		shift
		lr="$1"
		;;
	--name)
		shift
		name="$1"
		;;
	--algorithm_estimator)
		shift
		algorithm_estimator="$1"
		;;
	--model_name_or_path)
		shift
		model_name_or_path="$1"
		;;
	--dataset)
		shift
		dataset="$1"
		;;
	--eval_dataset)
		shift
		eval_dataset="$1"
		;;
	--batch_size)
		shift
		batch_size"$1"
		;;
	--epochs)
		shift
		epochs="$1"
		;;
	--max_prompt_length)
		shift
		max_prompt_length="$1"
		;;
	--max_response_length)
		shift
		max_response_length="$1"
		;;
	--ppo_mini_batch_size)
		shift
		ppo_mini_batch_size="$1"
		;;
	--ppo_micro_batch_size_per_gpu)
		shift
		ppo_micro_batch_size_per_gpu="$1"
		;;
	--rollout_n)
		shift
		rollout_n="$1"
		;;
	--nnodes)
		shift
		nnodes="$1"
		;;
	--n_gpus_per_node)
		shift
		n_gpus_per_node="$1"
		;;
	--include)
		shift
		include="$1"
		;;
	--ds_zero_stage)
		shift
		ds_zero_stage="$1"
		;;
	--dpo_beta)
		shift
		dpo_beta="$1"
		;;
	--)
		shift
		break
		;;
	esac
	shift
done

export OUTPUT_DIR=/opt/nas/p/mmu/zb/saved_checkpoint/$name
export WANDB_DIR=$OUTPUT_DIR/loggs

mkdir -p ${OUTPUT_DIR}
mkdir -p ${WANDB_DIR}

echo "wandb dir=$WANDB_DIR"
echo "working directory=$(pwd)"

dpo_optional=()
if [[ ${algorithm_estimator} = "dpo" ]]; then
	# 该参数需要优先设置
	dpo_optional+=(--module "openrlhf.cli.train_dpo")
fi

if [[ -n ${dpo_beta} ]]; then
	dpo_optional+=(--beta ${dpo_beta})
fi

deepspeed_params=()
if [[ -n $include ]]; then
	deepspeed_params+=(--include $include)
fi

max_len=$((max_prompt_length + max_response_length))

deepspeed --master_port=${MASTER_PORT} "${deepspeed_params[@]}" --no_local_rank \
	"${dpo_optional[@]}" \
	--save_path ${OUTPUT_DIR} \
	--save_steps -1 \
	--logging_steps 1 \
	--eval_steps -1 \
	--train_batch_size ${batch_size} \
	--micro_train_batch_size 1 \
	--pretrain ${model_name_or_path} \
	--bf16 \
	--max_epochs ${epochs} \
	--max_len ${max_len} \
	--zero_stage ${ds_zero_stage} \
	--learning_rate ${lr} \
	--dataset ${dataset} \
	--apply_chat_template \
	--chosen_key chosen \
	--rejected_key rejected \
	--attn_implementation flash_attention_2 \
	--load_checkpoint \
	--packing_samples \
	--gradient_checkpointing
