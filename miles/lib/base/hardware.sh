#!/bin/bash
#
# 硬件检测
#

source "${BASH_SOURCE[0]%/*}/log.sh"

detect_hardware() {
	NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l) || true
	if [ $NVLINK_COUNT -gt 0 ]; then
		HAS_NVLINK=1; log_info "Hardware: NVLink detected (${NVLINK_COUNT} references)"
	else
		HAS_NVLINK=0; log_info "Hardware: No NVLink"
	fi
	GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
	log_info "Hardware: ${GPU_COUNT} GPUs available"
	export HAS_NVLINK GPU_COUNT
}
