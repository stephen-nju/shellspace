#!/bin/bash
#
# 训练命令分派器
# 根据 TRAIN_MODE 调用对应模块的 build 函数
#

# Polar 命令构建（保留在此，其他两个已迁至各模块）
build_polar_train_cmd() {
	local cmd="${PYTHON_BIN:-python3} ${MILES_DIR}/train_async.py"

	cmd+=" --actor-num-nodes 1"
	cmd+=" --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE}"
	cmd+=" --rollout-num-gpus ${ROLLOUT_NUM_GPUS}"
	cmd+=" --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}"

	[ -n "${MODEL_ARGS:-}" ] && cmd+=" ${MODEL_ARGS}"
	cmd+=" --hf-checkpoint ${HF_CHECKPOINT}"
	[ -n "${REF_LOAD:-}" ] && cmd+=" --ref-load ${REF_LOAD}"
	[ -n "${LOAD_DIR:-}" ] && cmd+=" --load ${LOAD_DIR}"
	cmd+=" --save ${SAVE_DIR} --save-interval ${SAVE_INTERVAL:-10}"
	cmd+=" --update-weights-interval 1"

	cmd+=" --rollout-function-path slime_bridge.rollout.generate_rollout_polar_async"
	cmd+=" --custom-rm-path slime_bridge.reward.reward_func"
	cmd+=" --custom-reward-post-process-path slime_bridge.reward_post_process.post_process_rewards"
	[ -n "${CUSTOM_CONFIG_PATH:-}" ] && cmd+=" --custom-config-path ${CUSTOM_CONFIG_PATH}"
	cmd+=" --data-source-path slime_bridge.data_source.CeilEpochRolloutDataSourceWithBuffer"

	cmd+=" --prompt-data ${PROMPT_DATA}"
	cmd+=" --input-key prompt --label-key label --metadata-key metadata"
	cmd+=" --rollout-shuffle --reward-key score"
	cmd+=" --num-epoch ${NUM_EPOCH:-1}"
	cmd+=" --rollout-batch-size ${ROLLOUT_BATCH_SIZE:-4}"
	cmd+=" --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-16}"
	cmd+=" --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN:-16000}"
	cmd+=" --rollout-max-prompt-len ${ROLLOUT_MAX_PROMPT_LEN:-32000}"
	cmd+=" --rollout-temperature ${ROLLOUT_TEMPERATURE:-1.0}"
	cmd+=" --rollout-top-p ${ROLLOUT_TOP_P:-1.0}"
	cmd+=" --rollout-top-k ${ROLLOUT_TOP_K:--1}"
	cmd+=" --dynamic-history --num-steps-per-rollout 1"

	cmd+=" --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE:-2}"
	cmd+=" --sequence-parallel"
	cmd+=" --pipeline-model-parallel-size 1"
	cmd+=" --context-parallel-size 1"
	cmd+=" --expert-model-parallel-size 1"
	cmd+=" --expert-tensor-parallel-size 1"

	cmd+=" --recompute-granularity full --recompute-method uniform --recompute-num-layers 1"
	cmd+=" --use-dynamic-batch-size"
	cmd+=" --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-60000}"
	cmd+=" --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE:-256}"
	cmd+=" --distributed-timeout-minutes ${DISTRIBUTED_TIMEOUT_MINUTES:-30}"

	cmd+=" --advantage-estimator grpo --normalize-advantages"
	cmd+=" --use-tis"
	cmd+=" --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl"
	cmd+=" --entropy-coef 0.0"
	cmd+=" --eps-clip ${EPS_CLIP:-0.2}"
	[ -n "${EPS_CLIP_HIGH:-}" ] && cmd+=" --eps-clip-high ${EPS_CLIP_HIGH}"
	cmd+=" --gamma ${GAMMA:-1.0}"
	cmd+=" --clip-grad ${CLIP_GRAD:-1.0}"

	cmd+=" --optimizer ${OPTIMIZER_TYPE:-adam}"
	cmd+=" --lr ${LR:-1e-6} --lr-decay-style ${LR_DECAY_STYLE:-constant}"
	cmd+=" --weight-decay ${WEIGHT_DECAY:-0.1}"
	cmd+=" --adam-beta1 ${ADAM_BETA1:-0.9} --adam-beta2 ${ADAM_BETA2:-0.98}"

	cmd+=" --attention-dropout ${ATTENTION_DROPOUT:-0.0}"
	cmd+=" --hidden-dropout ${HIDDEN_DROPOUT:-0.0}"
	cmd+=" --accumulate-allreduce-grads-in-fp32"
	cmd+=" --attention-softmax-in-fp32"
	cmd+=" --attention-backend ${ATTENTION_BACKEND:-auto}"
	cmd+=" --no-gradient-accumulation-fusion"

	cmd+=" --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC:-0.8}"
	cmd+=" --sglang-context-length ${SGLANG_CONTEXT_LENGTH}"
	cmd+=" --sglang-tool-call-parser ${SGLANG_TOOL_CALL_PARSER:-qwen3_coder}"
	cmd+=" --router-policy ${SGLANG_ROUTER_POLICY:-round_robin}"
	cmd+=" --sglang-router-port ${SGLANG_ROUTER_PORT}"

	if [ "${USE_WANDB:-false}" = "true" ]; then
		cmd+=" --use-wandb --wandb-project ${WANDB_PROJECT:-polar-swegym-grpo}"
		cmd+=" --wandb-group ${WANDB_GROUP:-${NAME:-polar_default}}"
		[ -n "${WANDB_MODE:-}" ] && cmd+=" --wandb-mode ${WANDB_MODE}"
	fi

	echo "$cmd"
}

# 分派入口（由 rltrain.sh 调用）
build_train_cmd() {
	case "${TRAIN_MODE}" in
	rl)     build_rl_train_cmd ;;
	sft)    build_sft_train_cmd ;;
	polar)  build_polar_train_cmd ;;
	*)      log_error "Unknown TRAIN_MODE: ${TRAIN_MODE}"; return 1 ;;
	esac
}
