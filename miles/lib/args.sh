#!/bin/bash
#
# CLI 参数解析 (纯模块: 不调用业务逻辑, 仅把 args 解析为 shell 全局变量)
#
# 约定:
#   - VALUE_FLAGS / BOOL_FLAGS: --long-name -> ENV_VAR 映射表
#   - parse_args "$@" 返回: 0=成功, 1=getopt 错误, 2=未知选项, 10=-h/--help
#   - 新增选项只需在两表里各加一行, parse_args 自动 dispatch
#

declare -A VALUE_FLAGS=(
    # ── Train mode ────────────────────────────────────────────────────────────
    [train_mode]="TRAIN_MODE"

    # ── Base ─────────────────────────────────────────────────────────────────
    [name]="NAME"
    [model_name_or_path]="MODEL_NAME_OR_PATH"
    [hf_checkpoint]="HF_CHECKPOINT"
    [dataset]="DATASET"
    [eval_dataset]="EVAL_DATASET"
    [n_samples_per_prompt]="N_SAMPLES_PER_PROMPT"
    [rollout_batch_size]="ROLLOUT_BATCH_SIZE"
    [rollout_max_response_len]="ROLLOUT_MAX_RESPONSE_LEN"
    [rollout_max_prompt_len]="ROLLOUT_MAX_PROMPT_LEN"
    [rollout_temperature]="ROLLOUT_TEMPERATURE"
    [rollout_top_p]="ROLLOUT_TOP_P"
    [rollout_top_k]="ROLLOUT_TOP_K"
    [lr]="LR"
    [batch_size]="BATCH_SIZE"
    [num_epoch]="NUM_EPOCH"
    [num_rollout]="NUM_ROLLOUT"
    [advantage_estimator]="ADVANTAGE_ESTIMATOR"
    [eps_clip]="EPS_CLIP"
    [eps_clip_high]="EPS_CLIP_HIGH"
    [kl_loss_coef]="KL_LOSS_COEF"
    [kl_loss_type]="KL_LOSS_TYPE"
    [entropy_coef]="ENTROPY_COEF"
    [gamma]="GAMMA"
    [lambd]="LAMBD"
    [nnodes]="NNODES"
    [n_gpus_per_node]="N_GPUS_PER_NODE"
    [actor_num_nodes]="ACTOR_NUM_NODES"
    [actor_num_gpus_per_node]="ACTOR_NUM_GPUS_PER_NODE"
    [rollout_num_gpus]="ROLLOUT_NUM_GPUS"
    [rollout_num_gpus_per_engine]="ROLLOUT_NUM_GPUS_PER_ENGINE"
    [ref_load]="REF_LOAD"
    [load]="LOAD_DIR"
    [load_dir]="LOAD_DIR"
    [save]="SAVE_DIR"
    [save_dir]="SAVE_DIR"
    [save_interval]="SAVE_INTERVAL"
    [save_hf]="SAVE_HF"
    [optimizer]="OPTIMIZER_TYPE"
    [optimizer_type]="OPTIMIZER_TYPE"
    [lr_decay_style]="LR_DECAY_STYLE"
    [weight_decay]="WEIGHT_DECAY"
    [adam_beta1]="ADAM_BETA1"
    [adam_beta2]="ADAM_BETA2"
    [clip_grad]="CLIP_GRAD"
    [override_opt_param_scheduler]="OVERRIDE_OPT_PARAM_SCHEDULER"
    [tensor_model_parallel_size]="TENSOR_MODEL_PARALLEL_SIZE"
    [pipeline_model_parallel_size]="PIPELINE_MODEL_PARALLEL_SIZE"
    [context_parallel_size]="CONTEXT_PARALLEL_SIZE"
    [expert_model_parallel_size]="EXPERT_MODEL_PARALLEL_SIZE"
    [expert_tensor_parallel_size]="EXPERT_TENSOR_PARALLEL_SIZE"
    [recompute_granularity]="RECOMPUTE_GRANULARITY"
    [recompute_method]="RECOMPUTE_METHOD"
    [recompute_num_layers]="RECOMPUTE_NUM_LAYERS"
    [max_tokens_per_gpu]="MAX_TOKENS_PER_GPU"
    [log_probs_chunk_size]="LOG_PROBS_CHUNK_SIZE"
    [attention_dropout]="ATTENTION_DROPOUT"
    [hidden_dropout]="HIDDEN_DROPOUT"
    [attention_backend]="ATTENTION_BACKEND"
    [tis_clip]="TIS_CLIP"
    [sglang_mem_fraction_static]="SGLANG_MEM_FRACTION_STATIC"
    [sglang_context_length]="SGLANG_CONTEXT_LENGTH"
    [sglang_tool_call_parser]="SGLANG_TOOL_CALL_PARSER"
    [sglang_router_policy]="SGLANG_ROUTER_POLICY"
    [sglang_router_port]="SGLANG_ROUTER_PORT"
    [sglang_router_ip]="SGLANG_ROUTER_IP"
    [input_key]="INPUT_KEY"
    [label_key]="LABEL_KEY"
    [metadata_key]="METADATA_KEY"
    [prompt_data]="PROMPT_DATA"
    [eval_prompt_data]="EVAL_PROMPT_DATA"
    [eval_dataset]="EVAL_DATASET"
    [hf_checkpoint]="HF_CHECKPOINT"
    [data_source_path]="DATA_SOURCE_PATH"
    [rollout_function_path]="ROLLOUT_FUNCTION_PATH"
    [custom_rm_path]="CUSTOM_RM_PATH"
    [custom_reward_post_process_path]="CUSTOM_REWARD_POST_PROCESS_PATH"
    [custom_config_path]="CUSTOM_CONFIG_PATH"
    [topology]="TOPOLOGY_PATH"
    [wandb_project]="WANDB_PROJECT"
    [wandb_group]="WANDB_GROUP"
    [wandb_dir]="WANDB_DIR"
    [wandb_mode]="WANDB_MODE"
    [project_path]="PROJECT_PATH"
    [hf_home]="HF_HOME"
    [log_file]="LOG_FILE"
    [log_level]="LOG_LEVEL"
    [model_arg_path]="MODEL_ARG_PATH"
    [distributed_timeout_minutes]="DISTRIBUTED_TIMEOUT_MINUTES"

    # ── SFT 专用 ────────────────────────────────────────────────────────────────
    [global_batch_size]="GLOBAL_BATCH_SIZE"
    [rotary_base]="ROTARY_BASE"
    [min_lr]="MIN_LR"
    [lr_warmup_fraction]="LR_WARMUP_FRACTION"
    [loss_type]="LOSS_TYPE"
)

declare -A BOOL_FLAGS=(
    [dry_run]="DRY_RUN"
    [use_polar]="USE_POLAR"
    [use_wandb]="USE_WANDB"
    [disable_wandb_random_suffix]="DISABLE_WANDB_RANDOM_SUFFIX"
    [use_kl_loss]="USE_KL_LOSS"
    [use_tis]="USE_TIS"
    [sequence_parallel]="SEQUENCE_PARALLEL"
    [use_dynamic_batch_size]="USE_DYNAMIC_BATCH_SIZE"
    [normalize_advantages]="NORMALIZE_ADVANTAGES"
    [attention_softmax_in_fp32]="ATTENTION_SOFTMAX_IN_FP32"
    [no_gradient_accumulation_fusion]="NO_GRADIENT_ACCUMULATION_FUSION"
    [dynamic_history]="DYNAMIC_HISTORY"
    [accumulate_allreduce_grads_in_fp32]="ACCUMULATE_ALLREDUCE_GRADS_IN_FP32"

    # ── SFT 专用 ────────────────────────────────────────────────────────────────
    [calculate_per_token_loss]="CALCULATE_PER_TOKEN_LOSS"
    [disable_compute_advantages_and_returns]="DISABLE_COMPUTE_ADVANTAGES_AND_RETURNS"
    [debug_train_only]="DEBUG_TRAIN_ONLY"
    [apply_chat_template]="APPLY_CHAT_TEMPLATE"
)

parse_args() {
    # 不清空变量，保留环境变量设置的值
    # 只设置命令行明确传入的参数

    local long_opts="help"
    for k in "${!VALUE_FLAGS[@]}"; do
        long_opts+=",${k}:"
        # 同时注册 hyphen 版本，方便 CLI 使用 --train-mode 等格式
        local hyphen="${k//_/-}"
        [ "$hyphen" != "$k" ] && long_opts+=",${hyphen}:"
    done
    for k in "${!BOOL_FLAGS[@]}"; do
        long_opts+=",${k}"
        local hyphen="${k//_/-}"
        [ "$hyphen" != "$k" ] && long_opts+=",${hyphen}"
    done

    local options
    options=$(getopt -l "${long_opts}" -o "h:n:m:d:" -a -- "$@") 2>/dev/null || return 1
    eval set -- "$options"

    while true; do
        case "$1" in
        -h|--help)         return 10 ;;
        -n)                shift; NAME="$1" ;;
        -m)                shift; MODEL_NAME_OR_PATH="$1" ;;
        -d)                shift; DATASET="$1" ;;
        --)                shift; break ;;
        --*)
            local opt="${1#--}"
            local opt_key="${opt//-/_}"
            if [ -n "${VALUE_FLAGS[$opt_key]+x}" ]; then
                shift
                eval "${VALUE_FLAGS[$opt_key]}=\$1"
            elif [ -n "${BOOL_FLAGS[$opt_key]+x}" ]; then
                eval "${BOOL_FLAGS[$opt_key]}=true"
            elif [ "$opt_key" = "use_polar" ]; then
                # 向后兼容: --use_polar 同时设置 USE_POLAR 和 TRAIN_MODE
                USE_POLAR="true"
                TRAIN_MODE="polar"
            else
                log_error "未知选项: $1"
                return 2
            fi
            ;;
        *)                 log_error "非法参数: $1"; return 2 ;;
        esac
        shift
    done
    return 0
}
