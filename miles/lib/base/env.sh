#!/bin/bash
#
# 环境变量配置
#

setup_env() {
	export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}

	# F4 (spec §5.4): 多机 + 既未设 NCCL_SOCKET_IFNAME 也未设 NCCL_IB_HCA 时,
	# 不要默认 eth0 (IB / RoCE / 云上会错). 改为 warn + 不注入, 让 NCCL 自己选.
	if [ "${NNODES:-1}" -gt 1 ] && [ -z "${NCCL_SOCKET_IFNAME:-}" ] && [ -z "${NCCL_IB_HCA:-}" ]; then
		log_warn "NNODES>1 but neither NCCL_SOCKET_IFNAME nor NCCL_IB_HCA set."
		log_warn "NCCL will auto-select interface; if it picks the wrong one, set explicitly (e.g. ib0 / eth0 / ens5)."
	fi
	export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-}
	export NCCL_IB_TIMEOUT=22
	export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-}
	export NCCL_IB_TC=160
	export NCCL_NET_GDR_LEVEL=2
	export NCCL_IB_HCA=${NCCL_IB_HCA:-}
	export NCCL_ALGO=Ring
	export MKL_THREADING_LAYER=GNU
	export HYDRA_FULL_ERROR=1
	export RAY_IGNORE_VERSION_MISMATCH=True
	export PYTHONUNBUFFERED=1
	export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
	export LDFLAGS="-L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu ${LDFLAGS:-}"
	export LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"

	if [ -z "${CUDNN_LIB:-}" ] && command -v python3 >/dev/null 2>&1; then
		export CUDNN_LIB="$(python3 -c 'import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], "lib"))' 2>/dev/null || true)"
		[ -n "${CUDNN_LIB}" ] && [ -d "${CUDNN_LIB}" ] && {
			export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
			export LIBRARY_PATH="${CUDNN_LIB}:${LIBRARY_PATH:-}"
		}
	fi
	export HF_HOME WANDB_PROJECT SCRIPT_DIR
}
