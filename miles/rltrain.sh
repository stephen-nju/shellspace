#!/bin/bash
#
# Slime Training Launcher — 极简入口
#
# 用法:
#   ./rltrain.sh --train-mode rl   -n <name> -m <model> -d <data>
#   ./rltrain.sh --train-mode sft  --hf-checkpoint <hf_path> --prompt-data <parquet>
#   ./rltrain.sh --train-mode polar -n <name> --use_polar ... (向后兼容)
#   ./rltrain.sh --train-mode sft --dry-run ... (只打印命令)
#

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ── 始终导入的基础模块 ──────────────────────────────────────────
source "${SCRIPT_DIR}/lib/base/colors.sh"
source "${SCRIPT_DIR}/lib/base/log.sh"
source "${SCRIPT_DIR}/lib/base/cleanup.sh"
source "${SCRIPT_DIR}/lib/base/hardware.sh"
source "${SCRIPT_DIR}/lib/base/env.sh"
source "${SCRIPT_DIR}/lib/args.sh"
source "${SCRIPT_DIR}/lib/model_config.sh"
source "${SCRIPT_DIR}/lib/utils.sh"
source "${SCRIPT_DIR}/lib/ray.sh"
source "${SCRIPT_DIR}/lib/config.sh"
source "${SCRIPT_DIR}/lib/megatron.sh"

# ── 参数解析 ───────────────────────────────────────────────────
parse_args "$@" || {
	case $? in
	10) usage; exit 0 ;;
	*)  usage; exit 1 ;;
	esac
}

# 默认 TRAIN_MODE，向后兼容
TRAIN_MODE="${TRAIN_MODE:-rl}"

# ── 按需导入训练模块 ───────────────────────────────────────────
case "${TRAIN_MODE}" in
rl)
	source "${SCRIPT_DIR}/lib/rl/defaults.sh"
	source "${SCRIPT_DIR}/lib/rl/validate.sh"
	source "${SCRIPT_DIR}/lib/rl/cmd.sh"
	;;
sft)
	source "${SCRIPT_DIR}/lib/sft/defaults.sh"
	source "${SCRIPT_DIR}/lib/sft/validate.sh"
	source "${SCRIPT_DIR}/lib/sft/cmd.sh"
	;;
polar)
	source "${SCRIPT_DIR}/lib/polar.sh"
	source "${SCRIPT_DIR}/lib/rl/defaults.sh"
	source "${SCRIPT_DIR}/lib/rl/validate.sh"
	source "${SCRIPT_DIR}/lib/rl/cmd.sh"
	cleanup_add_hook polar_cleanup
	;;
*)
	log_error "Unknown TRAIN_MODE: ${TRAIN_MODE}"
	exit 1
	;;
esac

# ============================================================================
# 使用说明
# ============================================================================

usage() {
	cat <<EOF
${SCRIPT_NAME:-rltrain.sh} - Slime Training Launcher

Usage: ${SCRIPT_NAME:-rltrain.sh} [OPTIONS]

Required (RL):
    -n, --name NAME                实验名称
    -m, --model_name_or_path PATH  HF 模型路径
    -d, --dataset PATH             训练数据 (jsonl)

Required (SFT):
    --hf-checkpoint PATH            HF 权重路径
    --prompt-data PATH              parquet 数据路径

Train mode:
    --train-mode MODE               rl | sft | polar  (默认: rl)

Examples:
    RL:  ${SCRIPT_NAME:-rltrain.sh} --train-mode rl -n exp -m /opt/qwen2.5-72b -d data.jsonl
    SFT: ${SCRIPT_NAME:-rltrain.sh} --train-mode sft --hf-checkpoint /model --prompt-data data.parquet
    Polar: ${SCRIPT_NAME:-rltrain.sh} --train-mode polar -n exp -m /model -d data.jsonl
EOF
}

# ============================================================================
# 主函数
# ============================================================================

main() {
	# 1. Polar 早期初始化（在 apply_defaults 之前填 Polar 特有默认值）
	[ "${TRAIN_MODE}" = "polar" ] && polar_init

	# 2. 应用默认值
	apply_config_defaults
	[ "${TRAIN_MODE}" != "sft" ] && apply_rl_defaults
	[ "${TRAIN_MODE}" = "sft" ] && apply_sft_defaults

	# 3. Megatron 覆盖配置
	apply_megatron_overrides

	# 3. 加载模型架构参数
	load_model_args_from_file

	# 4. 校验
	[ "${TRAIN_MODE}" != "sft" ] && { run_validation || exit 1; }
	[ "${TRAIN_MODE}" = "sft" ]   && { validate_sft || { usage; exit 1; }; }
	[ "${TRAIN_MODE}" = "polar" ]  && { validate_prorlar_env || { usage; exit 1; }; }

	# 5. 环境设置
	setup_signal_handlers
	setup_env
	detect_hardware

	# 6. Polar 模式启动服务
	if [ "${TRAIN_MODE}" = "polar" ]; then
		render_prorlar_templates
		polar_start_services || { log_error "Polar 启动失败"; exit 1; }
	fi

	# 7. Ray 集群
	ray_start || { log_error "Ray 启动失败"; exit 1; }

	# ── Worker 早返 (spec §5.3) ────────────────────────────────
	# head 端通过 ray stop 关闭 cluster 时, worker 的 ray start --block
	# 会退出 -> wait 返回 -> EXIT trap -> cleanup. cleanup 会再调一次
	# ray stop --force, 因 worker 上已无 ray 进程故无害 (no-op).
	# 信号: SIGINT exit 130, SIGTERM exit 143 (Ray 2.5+ graceful).
	if [ "${RANK:-0}" != "0" ]; then
		log_info "Worker node (RANK=${RANK}); blocking until head stops Ray."
		wait
		exit 0
	fi
	# ── worker 早返结束 ─────────────────────────────────

	build_runtime_env_json

	# 8. 切换工作目录
	local workdir="${PROJECT_PATH}"
	if [ "${TRAIN_MODE}" = "sft" ] || [ "${TRAIN_MODE}" = "polar" ]; then
		workdir="${MILES_DIR}"
	fi
	cd "${workdir}" || { log_error "cd 失败: $workdir"; exit 1; }
	export PYTHONPATH="${workdir}:${MEGATRON_DIR:-}:${PYTHONPATH:-}"

	# 9. 打印配置
	print_config
	[ "${TRAIN_MODE}" = "polar" ] && print_polar_config
	[ "${TRAIN_MODE}" = "sft" ]   && print_sft_config

	# 10. 启动训练
	launch_training
}

# ============================================================================
# 启动训练
# ============================================================================

launch_training() {
	mkdir -p "${SAVE_DIR}" "${WANDB_DIR}"
	local train_cmd
	case "${TRAIN_MODE}" in
	rl|polar) train_cmd="$(build_rl_train_cmd)" ;;
	sft)      train_cmd="$(build_sft_train_cmd)" ;;
	esac

	if [ "${DRY_RUN}" = "true" ]; then
		log_info "Dry run: 命令不会提交到 Ray"
		echo "ray job submit --address=\"http://${RAY_HEAD_IP}:8265\" \\"
		echo "    --runtime-env-json='${RUNTIME_ENV_JSON}' \\"
		echo "    -- ${train_cmd}"
		return 0
	fi

	ray_submit_job "$train_cmd"
}

# ============================================================================
# 配置打印
# ============================================================================

print_config() {
	local mode="Miles RL Training"
	[ "${TRAIN_MODE}" = "sft" ]   && mode="ReTool SFT Training"
	[ "${TRAIN_MODE}" = "polar" ]  && mode="ProRL / Polar + Slime Training"

	echo ""
	echo -e "${COLOR_BLUE}==============================================${COLOR_NC}"
	echo -e "${COLOR_BLUE}        ${mode}        ${COLOR_NC}"
	echo -e "${COLOR_BLUE}==============================================${COLOR_NC}"
	printf " %-25s %s\n" "Experiment:" "${NAME}"
	printf " %-25s %s\n" "Model:" "${MODEL_NAME_OR_PATH:-${HF_CHECKPOINT}}"
	[ -n "${MODEL_ARG_PATH:-}" ] && printf " %-25s %s\n" "Model args file:" "${MODEL_ARG_PATH:-}"
	printf " %-25s %s\n" "Checkpoint:" "${HF_CHECKPOINT}"
	printf " %-25s %s\n" "Dataset:" "${DATASET:-${PROMPT_DATA}}"
	printf " %-25s %s\n" "Output:" "${SAVE_DIR}"
	echo -e "${COLOR_BLUE}----------------------------------------------${COLOR_NC}"
	printf " %-25s %s\n" "Learning rate:" "${LR}"
	printf " %-25s %s\n" "Batch size:" "${BATCH_SIZE:-${GLOBAL_BATCH_SIZE}}"
	printf " %-25s %s\n" "Epochs:" "${NUM_EPOCH}"
	[ -n "${NUM_ROLLOUT:-}" ] && printf " %-25s %s\n" "Total rollouts:" "${NUM_ROLLOUT:-}"
	echo -e "${COLOR_BLUE}----------------------------------------------${COLOR_NC}"
	printf " %-25s %s\n" "TP:" "${TENSOR_MODEL_PARALLEL_SIZE}"
	printf " %-25s %s\n" "GPUs:" "${ACTOR_NUM_GPUS_PER_NODE}"
	printf " %-25s %s\n" "Dynamic batch:" "${USE_DYNAMIC_BATCH_SIZE}"
	echo -e "${COLOR_BLUE}----------------------------------------------${COLOR_NC}"
	printf " %-25s %s\n" "Wandb:" "${USE_WANDB}"
	[ "${USE_WANDB}" = "true" ] && printf " %-25s %s\n" "Wandb project:" "${WANDB_PROJECT}"
	echo -e "${COLOR_BLUE}==============================================${COLOR_NC}"
	echo ""
}

print_sft_config() {
	echo -e "${COLOR_BLUE}-------------- SFT Config -------------------${COLOR_NC}"
	printf " %-25s %s\n" "Rollout function:" "${ROLLOUT_FUNCTION_PATH}"
	printf " %-25s %s\n" "Loss type:" "${LOSS_TYPE}"
	printf " %-25s %s\n" "Per-token loss:" "${CALCULATE_PER_TOKEN_LOSS}"
	printf " %-25s %s\n" "Global batch size:" "${GLOBAL_BATCH_SIZE}"
	[ -n "${ROTARY_BASE:-}" ] && printf " %-25s %s\n" "Rotary base:" "${ROTARY_BASE:-}"
	echo -e "${COLOR_BLUE}----------------------------------------------${COLOR_NC}"
}

# ============================================================================
# 入口
# ============================================================================

main "$@"
