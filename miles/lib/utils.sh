#!/bin/bash
#
# 工具函数 - Slime RL Training
#

# ============================================================================
# 颜色定义
# ============================================================================

COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_BLUE='\033[0;34m'
COLOR_NC='\033[0m'

# ============================================================================
# 日志函数
# ============================================================================

log() {
	local level="$1"
	local message="$2"
	local color=""
	local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

	case "$level" in
	DEBUG) color="$COLOR_BLUE" ;;
	INFO) color="$COLOR_GREEN" ;;
	WARN) color="$COLOR_YELLOW" ;;
	ERROR) color="$COLOR_RED" ;;
	*) color="$COLOR_NC" ;;
	esac

	echo -e "${color}[${timestamp}] [${level}] ${message}${COLOR_NC}"

	if [ -n "$LOG_FILE" ]; then
		echo "[${timestamp}] [${level}] ${message}" >>"$LOG_FILE"
	fi
}

log_debug() { [ "$LOG_LEVEL" = "DEBUG" ] && log "DEBUG" "$*" || true; }
log_info() { log "INFO" "$*"; }
log_warn() { log "WARN" "$*"; }
log_error() { log "ERROR" "$*"; }

# ============================================================================
# 信号处理 / 进程清理
# ============================================================================

CLEANUP_IN_PROGRESS=0
ALL_PIDS=()   # 所有需要清理的进程 PID
RAY_JOB_ID="" # 当前 Ray job ID

# 注册进程到统一清理列表
register_pid() {
	ALL_PIDS+=("$1")
}

# 注册 Ray job ID，用于清理
register_ray_job() {
	RAY_JOB_ID="$1"
}

cleanup() {
	if [ $CLEANUP_IN_PROGRESS -eq 1 ]; then
		log_warn "Cleanup already in progress, waiting..."
		sleep 5
		return
	fi
	CLEANUP_IN_PROGRESS=1

	log_warn "=========================================="
	log_warn "Cleaning up background services..."
	log_warn "=========================================="

	# 1. 停止 Ray job (如果存在)
	if [ -n "$RAY_JOB_ID" ]; then
		log_info "Stopping Ray job: $RAY_JOB_ID"
		ray job stop "$RAY_JOB_ID" --address="http://${RAY_HEAD_IP}:8265" 2>/dev/null || true
	fi

	# 2. Polar 服务 (polar.sh 提供)
	if declare -F polar_cleanup >/dev/null 2>&1; then
		polar_cleanup
	fi

	# 3. 统一 PIDs 列表 (graceful first)
	for pid in "${ALL_PIDS[@]}"; do
		if kill -0 "$pid" 2>/dev/null; then
			kill "$pid" 2>/dev/null || true
		fi
	done
	sleep 2
	# Force kill 残留
	for pid in "${ALL_PIDS[@]}"; do
		if kill -0 "$pid" 2>/dev/null; then
			kill -9 "$pid" 2>/dev/null || true
		fi
	done

	# 4. Ray cluster
	log_info "Stopping Ray cluster..."
	ray stop --force 2>/dev/null || true

	# 5. 残留进程 (兜底清理，使用更精确的模式避免误杀)
	log_info "Cleaning up residual processes..."
	pkill -9 -f "[t]rain_async.py" 2>/dev/null || true
	pkill -9 -f "[s]glang" 2>/dev/null || true
	pkill -9 -f "[r]ay::" 2>/dev/null || true
	pkill -9 -f "[p]ython.*miles.*train" 2>/dev/null || true

	log_info "Cleanup completed."
}

setup_signal_handlers() {
	trap cleanup SIGINT SIGTERM SIGHUP EXIT
}

# ============================================================================
# 必需参数验证
# ============================================================================

validate_required() {
	local errors=0

	if [ -z "$DATASET" ]; then
		log_error "Missing required: --dataset"
		errors=$((errors + 1))
	fi

	return $errors
}

# ============================================================================
# 路径验证
# ============================================================================

validate_paths() {
	local errors=0

	# 模型路径 - 允许HuggingFace模型名称（可选）
	if [ -n "${MODEL_NAME_OR_PATH:-}" ]; then
		if [ ! -d "$MODEL_NAME_OR_PATH" ] && [ ! -f "$MODEL_NAME_OR_PATH" ]; then
			if [[ "$MODEL_NAME_OR_PATH" != *"/"* ]] && [[ "$MODEL_NAME_OR_PATH" != *":"* ]]; then
				log_warn "Model path '$MODEL_NAME_OR_PATH' not found locally (assuming HF model)"
			fi
		fi
	fi

	# 数据集必须存在
	if [ ! -f "$DATASET" ]; then
		log_error "Dataset not found: $DATASET"
		errors=$((errors + 1))
	fi

	return $errors
}

# ============================================================================
# GPU配置验证
# ============================================================================

validate_gpu_config() {
	local errors=0

	if ! command -v nvidia-smi &>/dev/null; then
		log_warn "nvidia-smi not found, skipping GPU validation"
		return 0
	fi

	GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)

	if [ $GPU_COUNT -eq 0 ]; then
		log_error "No GPUs detected"
		return 1
	fi

	# Tensor Parallel验证
	TP_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
	if [ $TP_SIZE -gt $GPU_COUNT ]; then
		log_error "tensor_model_parallel_size ($TP_SIZE) > GPU count ($GPU_COUNT)"
		errors=$((errors + 1))
	fi

	if [ $((GPU_COUNT % TP_SIZE)) -ne 0 ]; then
		log_warn "tensor_model_parallel_size ($TP_SIZE) does not evenly divide GPU count ($GPU_COUNT)"
	fi

	# Rollout GPU验证
	ROLLOUT_GPUS=${ROLLOUT_NUM_GPUS:-8}
	if [ $ROLLOUT_GPUS -gt $GPU_COUNT ]; then
		log_error "rollout_num_gpus ($ROLLOUT_GPUS) > available GPUs ($GPU_COUNT)"
		errors=$((errors + 1))
	fi

	# Actor GPU验证
	ACTOR_GPUS=${ACTOR_NUM_GPUS_PER_NODE:-8}
	if [ $ACTOR_GPUS -gt $GPU_COUNT ]; then
		log_error "actor_num_gpus_per_node ($ACTOR_GPUS) > available GPUs ($GPU_COUNT)"
		errors=$((errors + 1))
	fi

	# Actor + Rollout 总数验证 (非 Polar 模式)
	if [ "${USE_POLAR:-false}" != "true" ]; then
		local total_gpus=$(($ACTOR_GPUS + ${ROLLOUT_NUM_GPUS:-8}))
		if [ -n "${N_GPUS_PER_NODE}" ] && [ "$total_gpus" -gt "${N_GPUS_PER_NODE}" ]; then
			log_error "actor ($ACTOR_GPUS) + rollout (${ROLLOUT_NUM_GPUS:-8}) = $total_gpus > n_gpus_per_node (${N_GPUS_PER_NODE})"
			errors=$((errors + 1))
		fi
	fi

	return $errors
}

# ============================================================================
# 参数范围验证
# ============================================================================

validate_param_ranges() {
	local errors=0

	check_range() {
		local name=$1 val=$2 min=$3 max=$4
		if [ -n "$val" ]; then
			local result
			result=$(echo "$val < $min || $val > $max" | bc -l 2>/dev/null) || result=""
			if [ -z "$result" ]; then
				log_warn "${name}=${val} could not be validated (bc not installed)"
			elif [ "$result" = "1" ]; then
				log_error "${name}=${val} out of range [${min}, ${max}]"
				errors=$((errors + 1))
			fi
		fi
	}

	# 检查预定义范围的参数
	check_range "lr" "$LR" 1e-8 1e-2
	check_range "eps_clip" "$EPS_CLIP" 0 1
	check_range "eps_clip_high" "$EPS_CLIP_HIGH" 0 1
	check_range "kl_loss_coef" "$KL_LOSS_COEF" 0 1
	check_range "entropy_coef" "$ENTROPY_COEF" 0 1
	check_range "n_samples_per_prompt" "$N_SAMPLES_PER_PROMPT" 1 128
	check_range "rollout_batch_size" "$ROLLOUT_BATCH_SIZE" 1 1024
	check_range "tensor_model_parallel_size" "$TENSOR_MODEL_PARALLEL_SIZE" 1 32
	check_range "max_tokens_per_gpu" "$MAX_TOKENS_PER_GPU" 512 65536

	return $errors
}

# ============================================================================
# 运行所有验证
# ============================================================================

run_validation() {
	local errors=0

	log_info "Running validation..."

	validate_required || errors=$((errors + 1))
	validate_paths || errors=$((errors + 1))
	validate_gpu_config || errors=$((errors + 1))
	validate_param_ranges || errors=$((errors + 1))

	if [ $errors -gt 0 ]; then
		log_error "Validation failed with $errors error(s)"
		return 1
	fi

	log_info "Validation passed"
	return 0
}

# ============================================================================
# 硬件检测
# ============================================================================

detect_hardware() {
	NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l) || true
	if [ $NVLINK_COUNT -gt 0 ]; then
		HAS_NVLINK=1
		log_info "Hardware: NVLink detected (${NVLINK_COUNT} references)"
	else
		HAS_NVLINK=0
		log_info "Hardware: No NVLink"
	fi

	GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
	log_info "Hardware: ${GPU_COUNT} GPUs available"

	export HAS_NVLINK GPU_COUNT
}

# ============================================================================
# 环境配置
# ============================================================================

setup_env() {
	# NCCL配置
	export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}
	export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-}
	export NCCL_IB_TIMEOUT=22
	export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-}
	export NCCL_IB_TC=160
	export NCCL_NET_GDR_LEVEL=2
	export NCCL_IB_HCA=${NCCL_IB_HCA:-}
	export NCCL_ALGO=Ring

	# Python/框架
	export MKL_THREADING_LAYER=GNU
	export HYDRA_FULL_ERROR=1
	export RAY_IGNORE_VERSION_MISMATCH=True
	export PYTHONUNBUFFERED=1

	# CUDA 库路径 (用于编译时链接 -lcuda)
	export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
	export LDFLAGS="-L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu ${LDFLAGS:-}"
	export LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"

	# cuDNN lib path
	if [ -z "${CUDNN_LIB:-}" ] && command -v python3 >/dev/null 2>&1; then
		export CUDNN_LIB="$(python3 -c 'import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], "lib"))' 2>/dev/null || true)"
		[ -n "${CUDNN_LIB}" ] && [ -d "${CUDNN_LIB}" ] && {
			export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
			export LIBRARY_PATH="${CUDNN_LIB}:${LIBRARY_PATH:-}"
		}
	fi

	# 路径
	export HF_HOME
	export WANDB_PROJECT
	export SCRIPT_DIR
}
