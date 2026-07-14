#!/bin/bash
#
# RL 参数校验
#

source "${BASH_SOURCE[0]%/*}/../base/log.sh"

validate_rl() {
	local errors=0

	# 数据集
	if [ -z "${DATASET:-}" ]; then
		log_error "Missing required: --dataset"
		errors=$((errors + 1))
	elif [ ! -f "${DATASET}" ]; then
		log_error "Dataset not found: ${DATASET}"
		errors=$((errors + 1))
	fi

	# 模型路径
	if [ -z "${MODEL_NAME_OR_PATH:-}" ] && [ -z "${HF_CHECKPOINT:-}" ]; then
		log_error "Missing required: --model_name_or_path or --hf_checkpoint"
		errors=$((errors + 1))
	fi

	# 参数范围
	check_range() {
		local name=$1 val=$2 min=$3 max=$4
		[ -z "$val" ] && return 0
		local result
		result=$(echo "$val < $min || $val > $max" | bc -l 2>/dev/null) || return 0
		[ "$result" = "1" ] && { log_error "${name}=${val} out of range [${min}, ${max}]"; errors=$((errors + 1)); }
	}

	check_range "eps_clip" "${EPS_CLIP}" 0 1
	check_range "kl_loss_coef" "${KL_LOSS_COEF}" 0 1
	check_range "entropy_coef" "${ENTROPY_COEF}" 0 1

	return $errors
}
