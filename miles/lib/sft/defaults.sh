#!/bin/bash
#
# SFT 训练默认值
#

apply_sft_defaults() {
	# ── 基础路径 ───────────────────────────────────────────────
	MILES_DIR="${MILES_DIR:-/opt/nas/p/mmu/zb/code/shellspace/slime/repo/ProRL/slime}"
	MEGATRON_DIR="${MEGATRON_DIR:-/opt/nas/p/mmu/zb/code/shellspace/slime/repo/ProRL/Megatron-LM}"

	# ── 模型参数默认值 ─────────────────────────────────────────
	MODEL_ARGS_ROTARY_BASE="${MODEL_ARGS_ROTARY_BASE:-5000000}"
	ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
	ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"

	# ── SFT 专用 rollout ──────────────────────────────────────
	ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-miles.rollout.sft_rollout.generate_rollout}"

	# ── 数据 (parquet 格式) ───────────────────────────────────
	INPUT_KEY="${INPUT_KEY:-messages}"

	# ── SFT 训练参数 ─────────────────────────────────────────
	NUM_EPOCH="${NUM_EPOCH:-3}"
	GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
	ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"

	# ── SFT 模式固定参数 ─────────────────────────────────────
	LOSS_TYPE="${LOSS_TYPE:-sft_loss}"
	CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-true}"
	DISABLE_COMPUTE_ADVANTAGES_AND_RETURNS="${DISABLE_COMPUTE_ADVANTAGES_AND_RETURNS:-true}"
	DEBUG_TRAIN_ONLY="${DEBUG_TRAIN_ONLY:-true}"

	# ── 并行配置 (TP=1 per 参考脚本) ────────────────────────
	TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
	SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-true}"
	PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
	CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
	EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-1}"
	EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"

	# ── 重计算 ───────────────────────────────────────────────
	RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
	RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
	RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"

	# ── 动态 batch ──────────────────────────────────────────
	USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-true}"
	MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"

	# ── 优化器 (cosine + warmup) ────────────────────────────
	OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-adam}"
	LR="${LR:-1e-5}"
	LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
	MIN_LR="${MIN_LR:-1e-6}"
	LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.1}"
	WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
	ADAM_BETA1="${ADAM_BETA1:-0.9}"
	ADAM_BETA2="${ADAM_BETA2:-0.95}"
	CLIP_GRAD="${CLIP_GRAD:-1.0}"

	# ── Dropout / 精度 ──────────────────────────────────────
	ATTENTION_DROPOUT="${ATTENTION_DROPOUT:-0.0}"
	HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-0.0}"
	ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"

	# ── 集群 ────────────────────────────────────────────────
	ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"

	# ── Ray ─────────────────────────────────────────────────
	RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
	RAY_NUM_GPUS="${RAY_NUM_GPUS:-8}"
	MASTER_ADDR="${MASTER_ADDR:-${RAY_HEAD_IP}}"
	RAY_DASHBOARD_HOST="${RAY_DASHBOARD_HOST:-0.0.0.0}"
	RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

	# ── Wandb ───────────────────────────────────────────────
	USE_WANDB="${USE_WANDB:-true}"
	WANDB_PROJECT="${WANDB_PROJECT:-slime-dev}"
	WANDB_GROUP="${WANDB_GROUP:-retool-sft}"

	# ── 路径 ───────────────────────────────────────────────
	SAVE_DIR="${SAVE_DIR:-/opt/nas/n/mmu/zhubin/saved_checkpoint/retool_sft/${NAME}}"
	WANDB_DIR="${WANDB_DIR:-${SAVE_DIR}/logs}"
	SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
}
