#!/bin/bash
#
# 模型配置模块
# 从指定路径导入模型配置
#

# ── 从文件加载模型架构参数 ─────────────────────────────────────────────────────
# 用法: load_model_args_from_file
# 直接 source MODEL_ARG_PATH 文件，利用 bash 数组展开
# MODEL_ARGS 必须在文件中定义为 bash 数组
load_model_args_from_file() {
	local model_arg_path="${MODEL_ARG_PATH:-}"

	if [ -z "$model_arg_path" ]; then
		return 0
	fi

	if [ ! -f "$model_arg_path" ]; then
		log_error "Model arg file not found: $model_arg_path"
		return 1
	fi

	log_info "Loading model args from: $model_arg_path"

	# 直接 source 文件，利用 bash 展开数组
	# 先清空，避免残留
	unset MODEL_ARGS
	local model_arg_file="$model_arg_path"
	source "$model_arg_file" 2>/dev/null || {
		log_error "Failed to source model arg file: $model_arg_file"
		return 1
	}

	# 如果是 bash 数组，展开为空格分隔的字符串
	if declare -p MODEL_ARGS 2>/dev/null | grep -q 'declare -a'; then
		log_info "Detected bash array format, expanding MODEL_ARGS"
		# 展开数组元素为空格分隔的字符串
		local expanded=""
		for arg in "${MODEL_ARGS[@]}"; do
			# 去除参数值两端的引号
			arg="${arg%\"}"
			arg="${arg#\"}"
			arg="${arg%\'}"
			arg="${arg#\'}"
			expanded="${expanded}${arg} "
		done
		MODEL_ARGS="${expanded%" "}"
	else
		# 如果不是数组，尝试作为普通变量使用（兼容纯文本格式）
		# 去除可能残留的引号
		MODEL_ARGS="${MODEL_ARGS%\"}"
		MODEL_ARGS="${MODEL_ARGS#\"}"
		MODEL_ARGS="${MODEL_ARGS%\'}"
		MODEL_ARGS="${MODEL_ARGS#\'}"
	fi

	log_info "MODEL_ARGS loaded: $(echo "${MODEL_ARGS}" | cut -c1-200)..."
}

# ── 兼容旧接口 ────────────────────────────────────────────────────────────────
# load_model_config 已被 load_model_args_from_file 替代，保留以兼容
load_model_config() {
	load_model_args_from_file
}
