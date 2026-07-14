#!/bin/bash
#
# SFT 训练命令构建
#

build_sft_train_cmd() {
	local cmd="${PYTHON_BIN:-python3} ${MILES_DIR}/train_async.py"

	# ── 集群 ────────────────────────────────────────────────
	cmd+=" --actor-num-nodes ${ACTOR_NUM_NODES}"
	cmd+=" --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE}"

	# ── 模型架构参数 ────────────────────────────────────────
	[ -n "${MODEL_ARGS:-}" ] && cmd+=" ${MODEL_ARGS}"

	# ── Checkpoint ──────────────────────────────────────────
	cmd+=" --hf-checkpoint ${HF_CHECKPOINT}"
	[ -n "${REF_LOAD:-}" ] && cmd+=" --ref-load ${REF_LOAD}"
	[ -n "${LOAD_DIR:-}" ] && cmd+=" --load ${LOAD_DIR}"
	cmd+=" --save ${SAVE_DIR} --save-interval ${SAVE_INTERVAL}"
	[ -n "${ROTARY_BASE:-}" ] && cmd+=" --rotary-base ${ROTARY_BASE}"

	# ── SFT Rollout ─────────────────────────────────────────
	[ "${APPLY_CHAT_TEMPLATE:-false}" = "true" ] && cmd+=" --apply-chat-template"
	cmd+=" --rollout-num-gpus ${ROLLOUT_NUM_GPUS}"
	cmd+=" --rollout-function-path ${ROLLOUT_FUNCTION_PATH}"
	cmd+=" --prompt-data ${PROMPT_DATA}"
	cmd+=" --input-key ${INPUT_KEY}"
	[ -n "${LABEL_KEY:-}" ] && cmd+=" --label-key ${LABEL_KEY}"
	[ -n "${METADATA_KEY:-}" ] && cmd+=" --metadata-key ${METADATA_KEY}"
	cmd+=" --rollout-shuffle"
	cmd+=" --num-epoch ${NUM_EPOCH}"
	cmd+=" --rollout-batch-size ${ROLLOUT_BATCH_SIZE}"
	cmd+=" --global-batch-size ${GLOBAL_BATCH_SIZE}"

	# ── SFT Loss 配置 ───────────────────────────────────────
	cmd+=" --loss-type ${LOSS_TYPE}"
	[ "$CALCULATE_PER_TOKEN_LOSS" = "true" ] && cmd+=" --calculate-per-token-loss"
	[ "$DISABLE_COMPUTE_ADVANTAGES_AND_RETURNS" = "true" ] && cmd+=" --disable-compute-advantages-and-returns"
	[ "$DEBUG_TRAIN_ONLY" = "true" ] && cmd+=" --debug-train-only"

	# ── 并行 ────────────────────────────────────────────────
	cmd+=" --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE}"
	[ "$SEQUENCE_PARALLEL" = "true" ] && cmd+=" --sequence-parallel"
	cmd+=" --pipeline-model-parallel-size ${PIPELINE_MODEL_PARALLEL_SIZE}"
	cmd+=" --context-parallel-size ${CONTEXT_PARALLEL_SIZE}"
	cmd+=" --expert-model-parallel-size ${EXPERT_MODEL_PARALLEL_SIZE}"
	cmd+=" --expert-tensor-parallel-size ${EXPERT_TENSOR_PARALLEL_SIZE}"

	# ── 重计算 ──────────────────────────────────────────────
	cmd+=" --recompute-granularity ${RECOMPUTE_GRANULARITY}"
	cmd+=" --recompute-method ${RECOMPUTE_METHOD}"
	cmd+=" --recompute-num-layers ${RECOMPUTE_NUM_LAYERS}"

	# ── 动态 batch ─────────────────────────────────────────
	[ "$USE_DYNAMIC_BATCH_SIZE" = "true" ] && cmd+=" --use-dynamic-batch-size"
	cmd+=" --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}"

	# ── 优化器 ──────────────────────────────────────────────
	cmd+=" --optimizer ${OPTIMIZER_TYPE}"
	cmd+=" --lr ${LR}"
	cmd+=" --lr-decay-style ${LR_DECAY_STYLE}"
	[ -n "${MIN_LR:-}" ] && cmd+=" --min-lr ${MIN_LR}"
	[ -n "${LR_WARMUP_FRACTION:-}" ] && cmd+=" --lr-warmup-fraction ${LR_WARMUP_FRACTION}"
	cmd+=" --weight-decay ${WEIGHT_DECAY}"
	cmd+=" --adam-beta1 ${ADAM_BETA1}"
	cmd+=" --adam-beta2 ${ADAM_BETA2}"
	cmd+=" --clip-grad ${CLIP_GRAD}"

	# ── 精度 / Dropout ─────────────────────────────────────
	cmd+=" --attention-dropout ${ATTENTION_DROPOUT}"
	cmd+=" --hidden-dropout ${HIDDEN_DROPOUT}"
	cmd+=" --accumulate-allreduce-grads-in-fp32"
	cmd+=" --attention-softmax-in-fp32"
	cmd+=" --attention-backend ${ATTENTION_BACKEND}"

	# ── Wandb ────────────────────────────────────────────────
	if [ "${USE_WANDB:-false}" = "true" ]; then
		cmd+=" --use-wandb --wandb-project ${WANDB_PROJECT}"
		[ -n "${WANDB_GROUP:-}" ] && cmd+=" --wandb-group ${WANDB_GROUP}"
		[ -n "${WANDB_MODE:-offline }" ] && cmd+=" --wandb-mode ${WANDB_MODE}"
		cmd+=" --wandb-dir ${WANDB_DIR}"
		[ -n "${WANDB_KEY:-}" ] && cmd+=" --wandb-key ${WANDB_KEY}"
	fi

	echo "$cmd"
}
