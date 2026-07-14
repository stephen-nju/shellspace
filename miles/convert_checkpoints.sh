cd repo/miles

# PYTHONPATH=/opt/nas/p/mmu/zb/code/shellspace/miles/repo/Megatron-LM python tools/convert_torch_dist_to_hf.py \
# 	--input-dir /opt/nas/n/mmu/zhubin/saved_checkpoint/0627_141447_Qwen3-4B-Instruction-miles_test/checkpoints/iter_0000749/ \
# 	--output-dir /opt/nas/n/mmu/zhubin/saved_checkpoint/0627_160016_Qwen3-4B-Instruction-miles_test_hf/ \
# 	--origin-hf-dir /opt/nas/n/mmu/models/Qwen/Qwen3-4B-Instruct-2507


source scripts/models/qwen3-4B-Instruct-2507.sh
PYTHONPATH="/opt/nas/p/mmu/zb/code/shellspace/miles/repo/Megatron-LM/:/opt/nas/p/mmu/zb/code/shellspace/miles/repo/miles/" torchrun --nproc_per_node 1 \
tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /opt/nas/n/mmu/zhubin/saved_checkpoint/0627_160016_Qwen3-4B-Instruction-miles_test_hf/ \
    --save  /opt/nas/n/mmu/zhubin/saved_checkpoint/0627_160016_Qwen3-4B-Instruction-miles_test_torch_distr \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
