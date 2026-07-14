#!/bin/bash
#
# SFT 参数校验
#

validate_sft() {
	local errors=0

	if [ -z "${HF_CHECKPOINT:-}" ]; then
		log_error "SFT: --hf_checkpoint is required"
		errors=$((errors + 1))
	elif [ ! -d "${HF_CHECKPOINT}" ]; then
		log_error "SFT: hf_checkpoint not found: ${HF_CHECKPOINT}"
		errors=$((errors + 1))
	fi

	if [ -z "${PROMPT_DATA:-}" ]; then
		log_error "SFT: --prompt_data (parquet file) is required"
		errors=$((errors + 1))
	elif [ ! -f "${PROMPT_DATA}" ]; then
		log_error "SFT: prompt_data not found: ${PROMPT_DATA}"
		errors=$((errors + 1))
	fi

	if [ -z "${SAVE_DIR:-}" ]; then
		log_error "SFT: --save_dir is required"
		errors=$((errors + 1))
	fi

	if [ -n "${ROTARY_BASE:-}" ] && ! [[ "$ROTARY_BASE" =~ ^[0-9]+$ ]]; then
		log_error "SFT: rotary_base must be a positive integer"
		errors=$((errors + 1))
	fi

	return $errors
}
