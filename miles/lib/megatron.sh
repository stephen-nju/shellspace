#!/bin/bash
#
# Megatron 专用配置
#

apply_megatron_overrides() {
	# ── OptimizerParamScheduler Override ─────────────────────────────
	# 设为 true 时，会忽略 checkpoint 中保存的 LR scheduler 状态，
	# 使用当前配置的 LR 值（避免 checkpoint LR 与当前 LR 不匹配导致的 assertion 错误）
	OVERRIDE_OPT_PARAM_SCHEDULER="${OVERRIDE_OPT_PARAM_SCHEDULER:-true}"
}
