#!/bin/bash
#
# Ray管理 - Slime RL Training
#

# ============================================================================
# 检查Ray状态
# ============================================================================

is_ray_running() {
	pgrep -f "ray::" >/dev/null 2>&1
}

# ============================================================================
# 启动Ray集群
# ============================================================================

ray_start() {
	# ── 启动前清理 (参考 retool 脚本: 防止旧进程干扰) ─────────────────────
	log_info "Cleaning up residual processes before launch..."
	pkill -9 sglang 2>/dev/null || true
	ray stop --force 2>/dev/null || true
	sleep 2
	pkill -9 -f "ray::" 2>/dev/null || true
	pkill -9 -f "train_async" 2>/dev/null || true
	sleep 2
	# 二次确认
	pkill -9 -f "ray::" 2>/dev/null || true
	sleep 1

	log_info "Starting Ray cluster..."
	log_info "  RANK: ${RANK:-0}"
	log_info "  RAY_NODE_IP: ${RAY_NODE_IP}"
	log_info "  RAY_HEAD_IP: ${RAY_HEAD_IP}"

	# GPU 数: RAY_NUM_GPUS 显式设置优先；默认 = actor + rollout (RL/SFT 通用)
	# 注 (spec F5): 简化假设每节点都跑 actor+rollout; 若 actor 只在 head,
	# 用户需在 worker 端 export RAY_NUM_GPUS=$ROLLOUT_NUM_GPUS
	local ray_gpus="${RAY_NUM_GPUS:-}"
	if [ -z "$ray_gpus" ]; then
		ray_gpus="$((${ACTOR_NUM_GPUS_PER_NODE:-0} + ${ROLLOUT_NUM_GPUS:-0}))"
	fi

	local node_ip="${RAY_NODE_IP:-127.0.0.1}"
	local head_ip="${RAY_HEAD_IP:-127.0.0.1}"
	local port="${RAY_PORT:-6379}"

	if [ "${RANK:-0}" = "0" ]; then
		# head 节点 (spec §5.1)
		ray start --head \
			--node-ip-address "${node_ip}" \
			--port "${port}" \
			--num-gpus "${ray_gpus}" \
			--disable-usage-stats \
			--dashboard-host="${RAY_DASHBOARD_HOST}" \
			--dashboard-port="${RAY_DASHBOARD_PORT}"
	else
		# worker 节点 (spec §5.1): --block 让前台阻塞直到 head 关闭
		ray start \
			--node-ip-address "${node_ip}" \
			--address "${head_ip}:${port}" \
			--num-gpus "${ray_gpus}" \
			--block
	fi

	# 等待启动
	local retries=30
	while [ $retries -gt 0 ]; do
		if is_ray_running; then
			log_info "Ray cluster started successfully"

			# F9: worker 模式额外验证 GCS 可达 (pgrep 只看 ray:: 进程存在,
			# 进程能起但 GCS 连不通时仍会被认为成功 -> 后续无限 wait)
			if [ "${RANK:-0}" != "0" ]; then
				if ! ray status --address "${head_ip}:${port}" >/dev/null 2>&1; then
					log_error "Worker cannot reach GCS at ${head_ip}:${port}"
					return 1
				fi
				log_info "Worker GCS reachable at ${head_ip}:${port}"
			fi
			return 0
		fi
		sleep 1
		retries=$((retries - 1))
	done

	log_error "Failed to start Ray cluster"
	return 1
}

# ============================================================================
# 构建运行时环境JSON
# ============================================================================

build_runtime_env_json() {
	# ProRL 模式: PYTHONPATH 需包含 slime + megatron, LD_LIBRARY_PATH 探测 cuDNN
	local py_parts="${PROJECT_PATH}:${SCRIPT_DIR}"
	[ -n "${MILES_DIR:-}" ] && py_parts="${MILES_DIR}:${py_parts}"
	[ -n "${MEGATRON_DIR:-}" ] && py_parts="${MEGATRON_DIR}:${py_parts}"
	[ -n "${POLAR_DIR:-}" ] && py_parts="${POLAR_DIR}/src:${py_parts}"

	# LD_LIBRARY_PATH: 继承当前值，追加 CUDA、cuDNN 和系统 cuda 库
	local ld_parts="${LD_LIBRARY_PATH:-}"
	# 追加 CUDA 库路径 (SGLang/FlashInfer 编译需要 -lcuda)
	ld_parts="/usr/local/cuda/lib64:${ld_parts}"
	# 追加系统 cuda 库路径 (libcuda.so 等)
	ld_parts="/usr/lib/x86_64-linux-gnu:${ld_parts}"
	# 追加 cuDNN (active venv 优先)
	if [ -n "${CUDNN_LIB:-}" ] && [ -d "${CUDNN_LIB}" ]; then
		ld_parts="${CUDNN_LIB}:${ld_parts}"
	elif command -v "${PYTHON_BIN:-python3}" >/dev/null 2>&1; then
		local detected_cudnn
		detected_cudnn="$("${PYTHON_BIN:-python3}" -c 'import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], "lib"))' 2>/dev/null || true)"
		[ -n "$detected_cudnn" ] && [ -d "$detected_cudnn" ] && ld_parts="${detected_cudnn}:${ld_parts}"
	fi
	# 去重
	ld_parts=$(echo "$ld_parts" | tr ':' '\n' | awk '!seen[$0]++' | tr '\n' ':' | sed 's/:$//')

	# CUDA_HOME 自动探测
	local cuda_home="${CUDA_HOME:-}"
	if [ -z "$cuda_home" ]; then
		local cuda_paths=(
			/usr/local/cuda
			/usr/local/cuda-12.8
			/usr/local/cuda-12.6
			/usr/local/cuda-12.4
			/usr/local/cuda-12.3
			/usr/local/cuda-12.2
			/usr/local/cuda-12.1
			/usr/local/cuda-12.0
			/opt/cuda
		)
		for p in "${cuda_paths[@]}"; do
			if [ -d "$p" ]; then
				cuda_home="$p"
				break
			fi
		done
	fi

	# cuDNN lib path
	local cuddn_lib="${CUDNN_LIB:-}"
	if [ -z "$cuddn_lib" ] && command -v "${PYTHON_BIN:-python3}" >/dev/null 2>&1; then
		cuddn_lib="$("${PYTHON_BIN:-python3}" -c 'import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], "lib"))' 2>/dev/null || true)"
	fi

	# LDFLAGS/LIBRARY_PATH for compilation (used by distutils/cmake to find cuda libraries)
	local ld_flags="-L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu"
	local lib_path="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"
	[ -n "$cuddn_lib" ] && [ -d "$cuddn_lib" ] && {
		ld_flags="${ld_flags} -L${cuddn_lib}"
		lib_path="${cuddn_lib}:${lib_path}"
	}

	# Build no_proxy: always include 127.0.0.1 and RAY_HEAD_IP (NCCL master),
	# then append whatever the current shell already has (deduped).
	# This prevents HTTP connections from being routed through a proxy, which
	# would cause RolloutManager.init_tracking and set_rollout_manager to hang.
	local _extra_proxy="${no_proxy:-${NO_PROXY:-}}"
	local _np="127.0.0.1,${RAY_HEAD_IP}"
	if [ -n "$_extra_proxy" ]; then
		_np="${_np},$(echo "$_extra_proxy" | tr ',' '\n' | awk '!seen[$0]++' | tr '\n' ',' | sed 's/,$//')"
	fi

	RUNTIME_ENV_JSON=$(
		cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${py_parts}",
    "LD_LIBRARY_PATH": "${ld_parts}",
    "LDFLAGS": "${ld_flags}",
    "LIBRARY_PATH": "${lib_path}",
    "CUDA_HOME": "${cuda_home}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "PYTORCH_ALLOC_CONF": "max_split_size_mb:2048,expandable_segments:True",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:2048,expandable_segments:True",
    "NVTE_DEBUG": "1",
    "NVTE_DEBUG_LEVEL": "2",
    "WANDB_DIR": "${WANDB_DIR:-${PROJECT_PATH}/logs}",
    "VIRTUAL_ENV": "${VIRTUAL_ENV:-}",
    "POLAR_APPTAINER_BIN": "${POLAR_APPTAINER_BIN:-}",
    "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}",
    "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
    "NO_PROXY": "${_np}",
    "no_proxy": "${_np}"
  }
}
EOF
	)
	export RUNTIME_ENV_JSON
}

# ============================================================================
# 提交Ray作业
# ============================================================================

ray_submit_job() {
	local train_cmd="$1"
	local output
	local exit_code

	log_info "Submitting Ray job..."

	# 打印 Ray 运行时环境变量（调试用）
	log_info "=== Ray Runtime Env JSON ==="
	echo "${RUNTIME_ENV_JSON}" | python3 -m json.tool 2>/dev/null || echo "${RUNTIME_ENV_JSON}"
	log_info "=== Train Command ==="
	echo "${train_cmd}" | sed 's/ --/\n  --/g'
	log_info "================================"

	output=$(ray job submit \
		--no-wait \
		--address="http://${RAY_HEAD_IP}:${RAY_DASHBOARD_PORT}" \
		--runtime-env-json="${RUNTIME_ENV_JSON}" \
		-- ${train_cmd} 2>&1)

	exit_code=$?

	if [ $exit_code -ne 0 ]; then
		log_error "Job submission failed: $output"
		# 异常退出: 触发全局清理
		cleanup
		return 1
	fi

	# 解析 job_id (格式: Job 'raysubmit_xxx' submitted successfully)
	local job_id
	job_id=$(echo "$output" | grep -oP "Job '\K[^']+" || echo "unknown")

	# 注册 job ID 到清理列表
	register_ray_job "$job_id"

	log_info "Job submitted: ${job_id}"
	log_info "Dashboard: http://${RAY_HEAD_IP}:${RAY_DASHBOARD_PORT}"
	log_info "Streaming training logs (Ctrl+C to detach)..."
	echo ""

	# 流式输出训练日志 (--follow 会阻塞直到 job 结束; Ctrl+C 触发 trap cleanup)
	ray job logs "${job_id}" --follow --address="http://${RAY_HEAD_IP}:${RAY_DASHBOARD_PORT}" 2>&1 || true

	return 0
}

# ============================================================================
# Ray作业管理
# ============================================================================

ray_job_status() {
	local job_id="$1"
	ray job status "$job_id" --address="http://${RAY_HEAD_IP}:${RAY_DASHBOARD_PORT}" 2>/dev/null || true
}

ray_job_logs() {
	local job_id="$1"
	ray job logs "$job_id" --address="http://${RAY_HEAD_IP}:${RAY_DASHBOARD_PORT}" 2>/dev/null || true
}
