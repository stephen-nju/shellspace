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

export PROJECT_PATH=/opt/nas/p/zb/code/RLtrain/
export HF_HOME=/opt/local/data/

cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export WANDB_PROJECT="RLtrian"

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

export name
export do_train
export do_eval
export dataset
export eval_dataset
export model_name_or_path
export batch_size
export ppo_batch_size
export max_prompt_length=512
export max_response_length=1024
export algorithm_estimator=grpo

options=$(getopt -l "help,do_train,do_eval,name:,dataset:,eval_dataset:,model_name_or_path:,batch_size:,reward_model:,max_prompt_length:,max_reponse_length:," -o "h" -a -- "$@")
eval set -- "$options"
while true; do
	case "$1" in
	-h | --help) Usage exit 0 ;;
	--do_train)	do_train=true ;;
	--do_eval)	do_eval=true ;;
	--model_name_or_path) shift model_name_or_path="$1" ;;
	--name) shift name=$1 ;;
	--dataset) shift dataset=$1 ;;
	--eval_dataset) shift eval_dataset=$1 ;;
	--batch_size) shift batch_size=$1 ;;
	--max_prompt_length) shift max_prompt_length=$1 ;;
	--max_response_length) shift max_reponse_length=$1 ;;
	--)
		shift
		break
		;;
	esac
	shift
done


export OUTPUT_DIR=/opt/nas/p/zb/saved_checkpoint/$name
export WANDB_DIR=$OUTPUT_DIR/loggs

mkdir -p ${OUTPUT_DIR}
mkdir -p ${WANDB_DIR}

echo "wandb dir=$WANDB_DIR"
echo "working directory=$(pwd)"

python3 -m verl.trainer.main_ppo \
	algorithm.adv_estimator=grpo \
	data.train_files=$dataset \
	data.val_files=$eval_dataset \
	data.train_batch_size=$batch_size \
	data.max_prompt_length=$max_prompt_length \
	data.max_response_length=$max_reponse_length \
	data.filter_overlong_prompts=True \
	data.truncation='error' \
	actor_rollout_ref.model.path=$model_name_or_path \
	actor_rollout_ref.actor.optim.lr=1e-6 \
	actor_rollout_ref.model.use_remove_padding=True \
	actor_rollout_ref.actor.ppo_mini_batch_size=256 \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
	actor_rollout_ref.actor.use_kl_loss=True \
	actor_rollout_ref.actor.kl_loss_coef=0.001 \
	actor_rollout_ref.actor.kl_loss_type=low_var_kl \
	actor_rollout_ref.actor.entropy_coeff=0 \
	actor_rollout_ref.model.enable_gradient_checkpointing=True \
	actor_rollout_ref.actor.fsdp_config.param_offload=False \
	actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
	actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
	actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
	actor_rollout_ref.rollout.name=vllm \
	actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
	actor_rollout_ref.rollout.n=5 \
	actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
	actor_rollout_ref.ref.fsdp_config.param_offload=True \
	+actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
	algorithm.use_kl_in_reward=False \
	trainer.critic_warmup=0 \
	trainer.logger=['console'] \
	trainer.project_name='RLtrain' \
	trainer.experiment_name=$name \
	trainer.n_gpus_per_node=8 \
	trainer.nnodes=1 \
	trainer.save_freq=20 \
	trainer.test_freq=5 \
	trainer.total_epochs=15 $@
