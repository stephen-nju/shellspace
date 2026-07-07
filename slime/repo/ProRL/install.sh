# 说明
# 安装 slime megatron-lm ProRL-Agent-Server
# 模型参数转换
source scripts/models/qwen3.5-4B.sh
PYTHONPATH="/opt/nas/p/mmu/zb/code/shellspace/slime/repo/ProRL/Megatron-LM:/opt/nas/p/mmu/zb/code/shellspace/slime/repo/ProRL/slime" torchrun --nproc_per_node 1 \
	tools/convert_hf_to_torch_dist.py \
	${MODEL_ARGS[@]} \
	--hf-checkpoint /opt/nas/n/mmu/models/Qwen3.5-4B \
	--save /opt/nas/n/mmu/zhubin/DATA/huggingface/Qwen3.5-4B_torch_dist \
	--tensor-model-parallel-size 1 \
	--pipeline-model-parallel-size 1 \
	--context-parallel-size 1 \
	--expert-model-parallel-size 1 \
	--expert-tensor-parallel-size 1 \
	--no-gradient-accumulation-fusion


# 用于构建apptainer镜像
python prepare_apptainer_images.py \
        --agent-cli-dir "./tmp/swegym_agent_cli/opt_node" \
        --image-dir "./tmp/swegym_apptainer_images" \
        --cache-dir "./tmp/apptainer_cache" \
        --tmp-dir "./tmp/apptainer_tmp" \
        --skip-cli \
        --jobs 1
# megatron-lm的patch
git apply /opt/nas/p/mmu/zb/code/shellspace/slime/repo/ProRL/slime/docker/patch/latest/megatron.patch --3way
# 本地安装
# 修改apptainer的相关配置
