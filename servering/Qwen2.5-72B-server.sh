source ~/.bashrc
conda activate zbvllm
export LD_LIBRARY_PATH=/opt/nas/p/conda/envs/deepseek_r1_qwen_32b/lib/python3.11/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server --model /opt/tools/resource/easy_resource/model/DeepSeek-R1 --served-model-name serving-0401-s4 --port 9001 --trust-remote-code --max_model_len 8192 --quantization fp8 --tensor-parallel-size 1 --gpu_memory_utilization 0.95 >/opt/nas/p/log/output1.log 2>&1 &
