#!/bin/bash
#
# RL 训练默认值
#

apply_rl_defaults() {
	# ── 训练 ───────────────────────────────────────────────────────────
	LR="${LR:-1e-6}"
	BATCH_SIZE="${BATCH_SIZE:-8}"
	NUM_EPOCH="${NUM_EPOCH:-1}"
	NUM_ROLLOUT="${NUM_ROLLOUT:-1}"

	# ── Rollout ───────────────────────────────────────────────────────
	N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
	ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
	ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
	ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-32000}"
	ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
	ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
	ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"

	# ── 算法 ──────────────────────────────────────────────────────────
	ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo}"
	EPS_CLIP="${EPS_CLIP:-0.2}"
	EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
	KL_LOSS_COEF="${KL_LOSS_COEF:-0.00}"
	KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"
	USE_KL_LOSS="${USE_KL_LOSS:-true}"
	ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
	NORMALIZE_ADVANTAGES="${NORMALIZE_ADVANTAGES:-true}"
	GAMMA="${GAMMA:-1.0}"
	LAMBD="${LAMBD:-1.0}"

	# ── 集群 ─────────────────────────────────────────────────────────
	NNODES="${NNODES:-1}"
	N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
	ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
	ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
	ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
	ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"

	# ── 优化器 ────────────────────────────────────────────────────────
	OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-adam}"
	LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
	WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
	ADAM_BETA1="${ADAM_BETA1:-0.9}"
	ADAM_BETA2="${ADAM_BETA2:-0.98}"
	CLIP_GRAD="${CLIP_GRAD:-1.0}"

	# ── 并行 ─────────────────────────────────────────────────────────
	TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
	SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-true}"
	PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
	CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
	EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-1}"
	EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"

	# ── 重计算 ───────────────────────────────────────────────────────
	RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
	RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
	RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"

	# ── 动态 batch ───────────────────────────────────────────────────
	USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-true}"
	MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-9216}"
	LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-256}"
	GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"

	# ── 内存 / Dropout ───────────────────────────────────────────────
	ATTENTION_DROPOUT="${ATTENTION_DROPOUT:-0.0}"
	HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-0.0}"
	ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"

	# ── TIS ─────────────────────────────────────────────────────────
	USE_TIS="${USE_TIS:-true}"
	TIS_CLIP="${TIS_CLIP:-2.0}"

	# ── SGLang ─────────────────────────────────────────────────────
	SGLANG_ROUTER_IP="${SGLANG_ROUTER_IP:-${RAY_NODE_IP:-127.0.0.1}}"
	SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-21000}"

	# ── 数据 ─────────────────────────────────────────────────────────
	INPUT_KEY="${INPUT_KEY:-prompt}"
	LABEL_KEY="${LABEL_KEY:-label}"
	METADATA_KEY="${METADATA_KEY:-metadata}"

	# ── Multi-turn ReTool ─────────────────────────────────────────────
	CUSTOM_GENERATE_FUNCTION_PATH="${CUSTOM_GENERATE_FUNCTION_PATH:-miles.rollout.generate_hub.multi_turn.generate}"
	GENERATE_TOOL_SPECS_PATH="${GENERATE_TOOL_SPECS_PATH:-examples.retool_v2.tool_sandbox.tool_specs}"
	GENERATE_EXECUTE_TOOL_FUNCTION_PATH="${GENERATE_EXECUTE_TOOL_FUNCTION_PATH:-examples.retool_v2.tool_sandbox.execute_tool}"
	GENERATE_TOOL_CALL_PARSER="${GENERATE_TOOL_CALL_PARSER:-qwen25}"
	GENERATE_MAX_TURNS="${GENERATE_MAX_TURNS:-16}"
	GENERATE_MULTI_SAMPLES="${GENERATE_MULTI_SAMPLES:-false}"
	LOG_MULTI_TURN="${LOG_MULTI_TURN:-true}"
	BALANCE_DATA="${BALANCE_DATA:-true}"
	APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-true}"
	CUSTOM_RM_PATH="${CUSTOM_RM_PATH:-examples.retool_v2.tool_sandbox.reward_func}"

	# ── Flags ────────────────────────────────────────────────────────
	LOG_PASS_RATE="${LOG_PASS_RATE:-true}"
	COLOCATE="${COLOCATE:-false}"

	# ── 路径 ────────────────────────────────────────────────────────
	PROJECT_PATH="${PROJECT_PATH:-/opt/nas/p/mmu/zb/code/shellspace/miles}"
	MILES_DIR="${MILES_DIR:-/opt/nas/p/mmu/zb/code/shellspace/miles/repo/miles}"
	MEGATRON_DIR="${MEGATRON_DIR:-/opt/nas/p/mmu/zb/code/shellspace/miles/repo/Megatron-LM}"
	SAVE_DIR_BASE="${SAVE_DIR_BASE:-/opt/nas/n/mmu/zhubin/saved_checkpoint/}"
	TRAIN_SCRIPT="${TRAIN_SCRIPT:-train.py}"

	# ── Ray ────────────────────────────────────────────────────────
	MASTER_ADDR="${MASTER_ADDR:-${RAY_HEAD_IP}}"
	DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-30}"

	# ── Wandb ──────────────────────────────────────────────────────
	WANDB_PROJECT="${WANDB_PROJECT:-RLtrain}"
	WANDB_GROUP="${WANDB_GROUP:-${NAME:-RLtrain}}"
	WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
	USE_WANDB="${USE_WANDB:-true}"
	DISABLE_WANDB_RANDOM_SUFFIX="${DISABLE_WANDB_RANDOM_SUFFIX:-false}"
	# ── 日志 ────────────────────────────────────────────────────────
	LOG_LEVEL="${LOG_LEVEL:-INFO}"
	LOG_FILE="${LOG_FILE:-/dev/null}"
	DRY_RUN="${DRY_RUN:-false}"

	# ── 路径推断 ───────────────────────────────────────────────────
	PROMPT_DATA="${PROMPT_DATA:-${DATASET}}"
	SAVE_DIR="${SAVE_DIR:-${SAVE_DIR_BASE}/${NAME}/checkpoints}"
	SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
	WANDB_DIR="${WANDB_DIR:-${SAVE_DIR}/logs}"

	# ── Eval ────────────────────────────────────────────────────────
	USE_EVAL="${USE_EVAL:-true}"
	EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
	EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA:-}"
	N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-4}"
	EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-8192}"
	EVAL_TOP_P="${EVAL_TOP_P:-1.0}"
}
