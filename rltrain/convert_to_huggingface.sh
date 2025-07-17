python -m verl.model_merger merge \
    --backend fsdp\
    --tie-word-embedding \
    --local_dir \
    --target_dir /path/to/merged_hf_model

