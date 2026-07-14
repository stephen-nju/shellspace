#!/bin/bash
#
# RL 训练命令构建
#

build_rl_train_cmd() {
	local cmd="${PYTHON_BIN:-python3} ${MILES_DIR}/${TRAIN_SCRIPT}"

	# ── 集群 ───────────────────────────────────────────────────────
	cmd+=" --actor-num-nodes ${ACTOR_NUM_NODES}"
	cmd+=" --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE}"
	cmd+=" --rollout-num-gpus ${ROLLOUT_NUM_GPUS}"
	cmd+=" --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}"

	# ── 模型架构 ───────────────────────────────────────────────────
	[ -n "${MODEL_ARGS:-}" ] && cmd+=" ${MODEL_ARGS}"

	# ── Checkpoint ─────────────────────────────────────────────────
	cmd+=" --hf-checkpoint ${HF_CHECKPOINT}"
	[ -n "${REF_LOAD:-}" ] && cmd+=" --ref-load ${REF_LOAD}"
	[ -n "${LOAD_DIR:-}" ] && cmd+=" --load ${LOAD_DIR}"
	cmd+=" --save ${SAVE_DIR} --save-interval ${SAVE_INTERVAL:-1000}"
	[ -n "${SAVE_HF:-}" ] && cmd+=" --save-hf ${SAVE_HF}"
	cmd+=" --update-weights-interval 1"

	# ── Multi-turn ReTool ────────────────────────────────────────────
	cmd+=" --custom-generate-function-path ${CUSTOM_GENERATE_FUNCTION_PATH}"
	cmd+=" --generate-tool-specs-path ${GENERATE_TOOL_SPECS_PATH}"
	cmd+=" --generate-execute-tool-function-path ${GENERATE_EXECUTE_TOOL_FUNCTION_PATH}"
	cmd+=" --generate-tool-call-parser ${GENERATE_TOOL_CALL_PARSER}"
	cmd+=" --generate-max-turns ${GENERATE_MAX_TURNS}"
	[ "${GENERATE_MULTI_SAMPLES}" = "true" ] && cmd+=" --generate-multi-samples"
	[ "${LOG_MULTI_TURN}" = "true" ] && cmd+=" --log-multi-turn"
	[ "${BALANCE_DATA}" = "true" ] && cmd+=" --balance-data"
	[ "${APPLY_CHAT_TEMPLATE}" = "true" ] && cmd+=" --apply-chat-template"
	cmd+=" --custom-rm-path ${CUSTOM_RM_PATH}"
	[ "${COLOCATE}" = "true" ] && cmd+=" --colocate"

	# ── Rollout ────────────────────────────────────────────────────
	# 当 MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1 时不传 rollout_function_path，避免与 custom_generate_function_path 冲突
	[ -n "${ROLLOUT_FUNCTION_PATH:-}" ] && [ "${MILES_EXPERIMENTAL_ROLLOUT_REFACTOR:-0}" != "1" ] && cmd+=" --rollout-function-path ${ROLLOUT_FUNCTION_PATH}"
	[ -n "${CUSTOM_CONFIG_PATH:-}" ] && cmd+=" --custom-config-path ${CUSTOM_CONFIG_PATH}"

	cmd+=" --prompt-data ${PROMPT_DATA}"
	cmd+=" --input-key ${INPUT_KEY} --label-key ${LABEL_KEY} --metadata-key ${METADATA_KEY}"
	cmd+=" --rollout-shuffle --reward-key score"
	cmd+=" --num-rollout ${NUM_ROLLOUT} --num-epoch ${NUM_EPOCH}"
	cmd+=" --rollout-batch-size ${ROLLOUT_BATCH_SIZE} --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}"
	cmd+=" --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}"
	cmd+=" --rollout-max-prompt-len ${ROLLOUT_MAX_PROMPT_LEN}"
	cmd+=" --rollout-temperature ${ROLLOUT_TEMPERATURE}"
	cmd+=" --rollout-top-p ${ROLLOUT_TOP_P} --rollout-top-k ${ROLLOUT_TOP_K}"

	# ── 并行 ───────────────────────────────────────────────────────
	cmd+=" --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE}"
	[ "${SEQUENCE_PARALLEL:-false}" = "true" ] && cmd+=" --sequence-parallel"
	cmd+=" --pipeline-model-parallel-size ${PIPELINE_MODEL_PARALLEL_SIZE}"
	cmd+=" --context-parallel-size ${CONTEXT_PARALLEL_SIZE}"
	cmd+=" --expert-model-parallel-size ${EXPERT_MODEL_PARALLEL_SIZE}"
	cmd+=" --expert-tensor-parallel-size ${EXPERT_TENSOR_PARALLEL_SIZE}"

	# ── 重计算 ────────────────────────────────────────────────────
	cmd+=" --recompute-granularity ${RECOMPUTE_GRANULARITY}"
	cmd+=" --recompute-method ${RECOMPUTE_METHOD}"
	cmd+=" --recompute-num-layers ${RECOMPUTE_NUM_LAYERS}"

	# ── 动态 batch ────────────────────────────────────────────────
	cmd+=" --use-dynamic-batch-size"
	cmd+=" --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}"
	cmd+=" --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE}"
	cmd+=" --global-batch-size ${GLOBAL_BATCH_SIZE}"

	# ── 算法 ──────────────────────────────────────────────────────
	cmd+=" --advantage-estimator ${ADVANTAGE_ESTIMATOR}"
	[ "${NORMALIZE_ADVANTAGES:-false}" = "true" ] && cmd+=" --normalize-advantages"
	[ "${USE_TIS:-false}" = "true" ] && cmd+=" --use-tis --tis-clip ${TIS_CLIP}"
	[ "${USE_KL_LOSS:-false}" = "true" ] && cmd+=" --use-kl-loss --kl-loss-coef ${KL_LOSS_COEF} --kl-loss-type ${KL_LOSS_TYPE}"
	cmd+=" --entropy-coef ${ENTROPY_COEF}"
	cmd+=" --eps-clip ${EPS_CLIP}"
	[ -n "${EPS_CLIP_HIGH:-}" ] && cmd+=" --eps-clip-high ${EPS_CLIP_HIGH}"
	cmd+=" --gamma ${GAMMA} --lambd ${LAMBD}"

	# ── 优化器 ────────────────────────────────────────────────────
	cmd+=" --optimizer ${OPTIMIZER_TYPE}"
	cmd+=" --lr ${LR} --lr-decay-style ${LR_DECAY_STYLE} --weight-decay ${WEIGHT_DECAY}"
	cmd+=" --adam-beta1 ${ADAM_BETA1} --adam-beta2 ${ADAM_BETA2} --clip-grad ${CLIP_GRAD}"
	[ "${OVERRIDE_OPT_PARAM_SCHEDULER:-false}" = "true" ] && cmd+=" --override-opt-param-scheduler"

	# ── 精度 ──────────────────────────────────────────────────────
	cmd+=" --attention-dropout ${ATTENTION_DROPOUT}"
	cmd+=" --hidden-dropout ${HIDDEN_DROPOUT}"
	cmd+=" --accumulate-allreduce-grads-in-fp32"
	cmd+=" --attention-softmax-in-fp32"
	cmd+=" --attention-backend ${ATTENTION_BACKEND}"
	cmd+=" --no-gradient-accumulation-fusion"
	[ "${LOG_PASS_RATE}" = "true" ] && cmd+=" --log-passrate"

	# ── Wandb ─────────────────────────────────────────────────────
	if [ "${USE_WANDB:-false}" = "true" ]; then
		cmd+=" --use-wandb --wandb-project ${WANDB_PROJECT}"
		[ -n "${WANDB_GROUP:-}" ] && cmd+=" --wandb-group ${WANDB_GROUP}"
		[ -n "${WANDB_MODE:-}" ] && cmd+=" --wandb-mode ${WANDB_MODE}"
		cmd+=" --wandb-dir ${WANDB_DIR}"
		[ -n "${WANDB_KEY:-}" ] && cmd+=" --wandb-key ${WANDB_KEY}"
		[ "${DISABLE_WANDB_RANDOM_SUFFIX:-false}" = "true" ] && cmd+=" --disable-wandb-random-suffix"
	fi

	# ── SGLang ────────────────────────────────────────────────────
	[ -n "${SGLANG_MEM_FRACTION_STATIC:-}" ] && cmd+=" --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC}"
	[ -n "${SGLANG_CONTEXT_LENGTH:-}" ] && cmd+=" --sglang-context-length ${SGLANG_CONTEXT_LENGTH}"

	# ── Eval ───────────────────────────────────────────────────────
	if [ "${USE_EVAL}" = "true" ]; then
		cmd+=" --eval-interval ${EVAL_INTERVAL}"
		[ -n "${EVAL_PROMPT_DATA:-}" ] && cmd+=" --eval-prompt-data ${EVAL_PROMPT_DATA}"
		cmd+=" --n-samples-per-eval-prompt ${N_SAMPLES_PER_EVAL_PROMPT}"
		cmd+=" --eval-max-response-len ${EVAL_MAX_RESPONSE_LEN}"
		cmd+=" --eval-top-p ${EVAL_TOP_P}"
	fi

	echo "$cmd"
}
