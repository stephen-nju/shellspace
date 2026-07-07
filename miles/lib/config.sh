#!/bin/bash
#
# 通用默认值（所有模式共用，与 train-mode 无关）
#

apply_config_defaults() {
	# ── 基础路径 ───────────────────────────────────────────────
	NAME="${NAME:-$(date +%Y%m%d_%H%M%S)}"
	PROJECT_PATH="${PROJECT_PATH:-/opt/nas/p/mmu/zb/code/shellspace/miles}"
	HF_HOME="${HF_HOME:-/opt/local/data/}"
	SAVE_DIR_BASE="${SAVE_DIR_BASE:-/opt/nas/n/mmu/zhubin/saved_checkpoint/}"

	# ── Ray ─────────────────────────────────────────────────
	# RAY_NODE_IP: 本机网卡 IP，ray head/worker 进程绑定在此地址
	# 优先读取 ethb_ip 环境变量（Kubernetes Pod 场景），否则自动探测
	# RAY_HEAD_IP: worker 连接 head 节点的地址（多机时需为可路由 IP，单机可用 127.0.0.1）

	RAY_HEAD_IP="${RAY_HEAD_IP:-127.0.0.1}"
	RAY_PORT="${RAY_PORT:-6379}"
	RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
	RAY_DASHBOARD_HOST="${RAY_DASHBOARD_HOST:-0.0.0.0}"
	MASTER_ADDR="${MASTER_ADDR:-${RAY_HEAD_IP}}"

	# ── 多机语义 (spec §4.2, §5.2) ─────────────────────────────
	# F1 兜底: 在覆盖默认值之前先检测 RANK 是否显式给过
	if [ -z "${RANK+x}" ] && [ "${NNODES:-1}" -gt 1 ]; then
		log_warn "NNODES>1 but RANK unset; defaulting to RANK=0 (head). If this is a worker, set RANK=1..N-1 explicitly."
	fi
	RANK="${RANK:-0}"
	NNODES="${NNODES:-1}"

	# 多机一致性断言
	if [ "${NNODES}" -gt 1 ]; then
		if [ "${MASTER_ADDR}" != "${RAY_HEAD_IP}" ]; then
			log_error "MASTER_ADDR (${MASTER_ADDR}) != RAY_HEAD_IP (${RAY_HEAD_IP}); multi-node requires they match"
			return 1
		fi
	fi

	# ── Wandb ───────────────────────────────────────────────
	WANDB_PROJECT="${WANDB_PROJECT:-RLtrain}"
	USE_WANDB="${USE_WANDB:-false}"
	WANDB_MODE="${WANDB_MODE:-offline}"

	# ── 日志 ────────────────────────────────────────────────
	LOG_LEVEL="${LOG_LEVEL:-INFO}"
	LOG_FILE="${LOG_FILE:-/dev/null}"
	DRY_RUN="${DRY_RUN:-false}"

	# ── MODEL_ARGS（由 load_model_args_from_file 填充）──────
	MODEL_ARGS=""

	# ── 路径推断 ───────────────────────────────────────────
	HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_NAME_OR_PATH:-}}"
	PROMPT_DATA="${PROMPT_DATA:-${DATASET}}"
	SAVE_DIR="${SAVE_DIR:-${SAVE_DIR_BASE}${NAME}/checkpoints}"
	WANDB_DIR="${WANDB_DIR:-${SAVE_DIR}/logs}"

	# ── LOAD_DIR 推断 ──────────────────────────────────────
	if [ -z "${LOAD_DIR:-}" ]; then
		if [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
			LOAD_DIR="${SAVE_DIR}"
		elif [ -n "${REF_LOAD:-}" ] && [ -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]; then
			LOAD_DIR="${REF_LOAD}"
		fi
	fi
}
