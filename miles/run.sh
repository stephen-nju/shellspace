#!/bin/bash
#
# Slime RL Training Launcher
# 带详细日志的启动脚本
#

set -euo pipefail

# 时间戳
export DATE=$(date "+%m%d_%H%M%S")
echo "==============================================="
echo "  Training scripts date: ${DATE}"
echo "==============================================="

# HF cache
export HF_HOME=/opt/local/data/

# Python path for Megatron and Slime imports

# 日志目录和文件 - 使用绝对路径避免cd后失效
SCRIPT_DIR_RUN="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LOG_DIR="${SCRIPT_DIR_RUN}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_${DATE}.log"
export LOG_FILE

# 多机配置
hoststr=${hoststr:-""}
if [ -n "$hoststr" ]; then
	echo "hoststr=${hoststr}"
	echo "$hoststr" | sed 's/,/\n/g' >/opt/nas/p/mmu/zb/code/shellspace/slime/cache/hostfile
fi

# 启用DEBUG日志
export LOG_LEVEL=${LOG_LEVEL:-DEBUG}

# 同时输出到终端和日志文件
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== Script started, log file: ${LOG_FILE} ====="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Working dir: $(pwd)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Script dir: ${SCRIPT_DIR_RUN}"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching rltrain.sh..."
echo ""
# 启动训练

# export MODEL_ARG_PATH="/opt/nas/p/mmu/zb/code/shellspace/slime/repo/ProRL/slime/scripts/models/qwen3.5-4B.sh"
# export ACTOR_NUM_GPUS_PER_NODE=2
# export ROLLOUT_NUM_GPUS=6
# export ROLLOUT_BATCH_SIZE=4
# export N_SAMPLES_PER_PROMPT=4
export USE_WANDB=true
# ./rltrain.sh \
# 	--use_polar \
# 	--name="${DATE}_Qwen3.5-4B-slime_test" \
# 	--dataset /opt/nas/p/mmu/zb/code/shellspace/slime/repo/ProRL/ProRL-Agent-Server/examples/swegym_slime_grpo/swegym_train_imagetest.jsonl \
# 	--hf_checkpoint /opt/nas/n/mmu/models/Qwen3.5-4B \
# 	--ref_load /opt/nas/n/mmu/zhubin/DATA/huggingface/Qwen3.5-4B_torch_discheckpoints/\
# 	--log_file "${LOG_FILE}" \
# 	--log_level DEBUG

export MODEL_ARG_PATH="/opt/nas/p/mmu/zb/code/shellspace/miles/repo/miles/scripts/models/qwen3-4B-Instruct-2507.sh"
export MILES_DIR="/opt/nas/p/mmu/zb/code/shellspace/miles/repo/miles/"
export MEGATRON_DIR="/opt/nas/p/mmu/zb/code/shellspace/miles/repo/Megatron-LM/"
export SGLANG_HOST_IP=127.0.0.1
export RAY_NODE_IP=127.0.0.1
export MASTER_ADDR=127.0.0.1

# ── 多机部署 (optional) ───────────────────────────────────────
# Head 节点:
#   RANK=0 RAY_NODE_IP=<head_ip> RAY_HEAD_IP=<head_ip> NNODES=N ./run.sh ...
# Worker 节点 (RANK=1..N-1):
#   RANK=<i> RAY_NODE_IP=<worker_ip> RAY_HEAD_IP=<head_ip> NNODES=N ./run.sh ...
# 必传:
#   RAY_NODE_IP   本机真实网卡 IP (不要用 127.0.0.1)
#   RAY_HEAD_IP   head 节点 IP (所有节点必须一致)
#   NCCL_SOCKET_IFNAME  本机通信网卡名 (多机时建议显式设, 否则 NCCL 自动选)
# 可选:
#   NCCL_IB_HCA   IB HCA 名 (如 mlx5_0)
#   NCCL_DEBUG=INFO  打开 NCCL 调试日志
# ──────────────────────────────────────────────────────────────
./rltrain.sh \
	--train-mode rl \
	--name="${DATE}_Qwen3-4B-Instruction-miles_rl_test" \
	--dataset /opt/nas/n/mmu/zhubin/DATA/huggingface/dapo-math-17k-processed/dapo_math_17k_cleaned.jsonl \
	--hf_checkpoint /opt/nas/n/mmu/zhubin/saved_checkpoint/0627_160016_Qwen3-4B-Instruction-miles_test_hf \
	--ref_load /opt/nas/n/mmu/zhubin/saved_checkpoint/0627_160016_Qwen3-4B-Instruction-miles_test_torch_distr \
	--eval_prompt_data /opt/nas/n/mmu/zhubin/DATA/huggingface/aime-2024/aime-2024.jsonl \
	--log_file "${LOG_FILE}" \
	--num-rollout 3000 \
	--override_opt_param_scheduler true \
	--log_level DEBUG

# --ref_load /opt/nas/n/mmu/zhubin/DATA/huggingface/Qwen3-4B-Instruct-2507_torch_dist/ \
# --hf_checkpoint /opt/nas/n/mmu/models/Qwen/Qwen3-4B-Instruct-2507 \
