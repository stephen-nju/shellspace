exec >/opt/nas/p/zhubin/easyjobLog/mstrain.log

export dataset_dir=/opt/nas/p/zhubin/DATA/train_data/openai
# 构建dataset的名称
# export union_conversations_v5_norm_markdown=$dataset_dir/union_conversations_v5/union_conversations_train_norm_v5_openai_markdownv1.jsonl
# export union_conversations_v4_ll_markdown=$dataset_dir/union_conversations_v5/union_conversations_train_long_v4_less4096_openai_markdownv1.jsonl
# export diting_v1_markdown=$dataset_dir/diting_iio_llf_v1_train_openai_markdownv1.jsonl
# export diting_v1_dev=$dataset_dir/diting_iio_llf_v1_dev_openai_markdownv1.jsonl
# export dialogsum_chinese_markdown=$dataset_dir/dialogsum_conversation_train_iio_llf_v1_markdownv1.jsonl
# export samsum_chinese_markdown=$dataset_dir/samsum_conversation_train_iio_llf_v1_markdownv1.jsonl
# export liantong_conversations_v1_markdown=$dataset_dir/liantong_conversation_train_norm_v1_openai_markdownv1.jsonl
# export union_conversations_v5_dev_markdown=$dataset_dir/union_conversations_v5/union_conversation_v5.4_iio_llf_dev_openai_markdownv1.jsonl
export alpaca_gpt4_zh_retain=$dataset_dir/alpaca_gpt4_zh_retain.jsonl

# ./mstrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen2_5 --lora_rank 32 --lora_target all-linear --loraplus_lr_ratio 16 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-1.5B/ \
# 	--name 0206_ms_Qwen2.5-1.5B_lud_lorap16_ep3_lr2e4_bs4 \
# 	--dataset $union_conversations_v5_norm_markdown,$union_conversations_v4_ll_markdown,$diting_v1_markdown,$liantong_conversations_v1_markdown \
# 	--gradient_accumulation_steps=4 --max_length=4096 --epochs=3 --batch_size 4 --lr=2e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset $union_conversations_v5_dev_markdown --eval_strategy steps --eval_steps 500 \

# wait
# ./mstrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen2_5 --lora_rank 32 --lora_target all-linear --loraplus_lr_ratio 16 \
# 	--model_name_or_path /opt/nas/n/zhubin/DATA/models/Qwen/Qwen2.5-1.5B-Instruct/ \
# 	--name 0206_ms_Qwen2.5-1.5B_Instruct_lud_lorap16_ep3_lr2e4_bs4 \
# 	--dataset $union_conversations_v5_norm_markdown,$union_conversations_v4_ll_markdown,$diting_v1_markdown,$liantong_conversations_v1_markdown \
# 	--gradient_accumulation_steps=4 --max_length=4096 --epochs=3 --batch_size 4 --lr=2e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset $union_conversations_v5_dev_markdown --eval_strategy steps --eval_steps 500 \

# wait
# ./mstrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen2_5 --lora_rank 32 --lora_target all-linear --loraplus_lr_ratio 16 \
# 	--model_name_or_path /opt/nas/p/zhubin/DATA/models/Qwen/Qwen2.5-3B \
# 	--name 0206_ms_Qwen2.5-3B_lud_lorap16_ep3_lr2e4_bs4 \
# 	--dataset $union_conversations_v5_norm_markdown,$union_conversations_v4_ll_markdown,$diting_v1_markdown,$liantong_conversations_v1_markdown \
# 	--gradient_accumulation_steps=4 --max_length=4096 --epochs=3 --batch_size 4 --lr=2e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset $union_conversations_v5_dev_markdown --eval_strategy steps --eval_steps 500 \
# wait
# ./mstrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen2_5 --lora_rank 32 --lora_target all-linear --loraplus_lr_ratio 16 \
# 	--model_name_or_path /opt/nas/p/zhubin/DATA/models/Qwen/Qwen2.5-3B-Instruct \
# 	--name 0206_ms_Qwen2.5-3B-Instruct_lud_lorap16_ep3_lr2e4_bs4 \
# 	--dataset $union_conversations_v5_norm_markdown,$union_conversations_v4_ll_markdown,$diting_v1_markdown,$liantong_conversations_v1_markdown \
# 	--gradient_accumulation_steps=4 --max_length=4096 --epochs=3 --batch_size 4 --lr=2e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset $union_conversations_v5_dev_markdown --eval_strategy steps --eval_steps 500 \

# ./mstrain.sh --do_train --do_eval --stage sft --finetuning_type full --template qwen2_5 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-1.5B/ \
# 	--name 0207_ms_Qwen2.5-1.5B_lud_full_ep3_lr2e5_bs4 \
# 	--dataset $union_conversations_v5_norm_markdown,$union_conversations_v4_ll_markdown,$diting_v1_markdown,$liantong_conversations_v1_markdown \
# 	--gradient_accumulation_steps=4 --max_length=4096 --epochs=3 --batch_size 4 --lr=2e-5 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset $union_conversations_v5_dev_markdown --eval_strategy steps --eval_steps 500 \

# ./mstrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template default --lora_rank 32 --lora_target all-linear --loraplus_lr_ratio 16 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-1.5B/ \
# 	--name 0207_ms_Qwen2.5-1.5B_lud_template_def_lorap16_ep3_lr2e4_bs4 \
# 	--dataset $union_conversations_v5_norm_markdown,$union_conversations_v4_ll_markdown,$diting_v1_markdown,$liantong_conversations_v1_markdown \
# 	--gradient_accumulation_steps=4 --max_length=4096 --epochs=3 --batch_size 4 --lr=2e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset $union_conversations_v5_dev_markdown --eval_strategy steps --eval_steps 500 \

# ./mstrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen2_5 --lora_rank 32 --lora_target all-linear,all-embedding,lm_head --loraplus_lr_ratio 16 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-1.5B/ \
# 	--name 0207_ms_Qwen2.5-1.5B_lud_template_qwen_lm_head_lorap16_ep3_lr2e4_bs4 \
# 	--dataset $union_conversations_v5_norm_markdown,$union_conversations_v4_ll_markdown,$diting_v1_markdown,$liantong_conversations_v1_markdown \
# 	--gradient_accumulation_steps=4 --max_length=4096 --epochs=3 --batch_size 4 --lr=2e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset $union_conversations_v5_dev_markdown --eval_strategy steps --eval_steps 500 \

#  ./mstrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen2_5 --lora_rank 32 --lora_target all-linear --loraplus_lr_ratio 16 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-1.5B/ \
# 	--name 0207_ms_Qwen2.5-1.5B_alpaca_lorap16_ep3_lr2e4_bs4 \
# 	--dataset $alpaca_gpt4_zh_retain \
# 	--gradient_accumulation_steps=4 --max_length=4096 --epochs=3 --batch_size 4 --lr=2e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset $union_conversations_v5_dev_markdown --eval_strategy steps --eval_steps 500 \

# ./mstrain.sh --do_train --do_eval --stage sft --finetuning_type full --template qwen2_5 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-1.5B/ \
# 	--name TEST \
# 	--dataset $union_conversations_v5_norm_markdown,$union_conversations_v4_ll_markdown,$diting_v1_markdown,$liantong_conversations_v1_markdown \
# 	--gradient_accumulation_steps=4 --max_length=4096 --epochs=30 --batch_size 4 --lr=2e-5 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset $union_conversations_v5_dev_markdown --eval_strategy steps --eval_steps 500 \

#lora 微调
# ./mstrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen2_5 --lora_rank 32 --lora_target all-linear --loraplus_lr_ratio 16 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-1.5B/ \
# 	--name 0208_ms_Qwen2.5-1.5B_lud_lorap16_ep5_lr5e4_bs4 \
# 	--dataset $union_conversations_v5_norm_markdown,$union_conversations_v4_ll_markdown,$diting_v1_markdown,$liantong_conversations_v1_markdown \
# 	--gradient_accumulation_steps=4 --max_length=4096 --epochs=5 --batch_size 4 --lr=5e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset $union_conversations_v5_dev_markdown --eval_strategy steps --eval_steps 500 \

# ./mstrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen2_5 --lora_rank 32 --lora_target all-linear --loraplus_lr_ratio 16 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-1.5B/ \
# 	--name 0208_ms_Qwen2.5-1.5B_lud_lorap16_ep5_lr1e5_bs4 \
# 	--dataset $union_conversations_v5_norm_markdown,$union_conversations_v4_ll_markdown,$diting_v1_markdown,$liantong_conversations_v1_markdown \
# 	--gradient_accumulation_steps=4 --max_length=4096 --epochs=5 --batch_size 4 --lr=1e-5 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset $union_conversations_v5_dev_markdown --eval_strategy steps --eval_steps 500 \

# ./mstrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen2_5 --lora_rank 32 --lora_target all-linear,all-embedding,lm_head --loraplus_lr_ratio 16 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-1.5B/ \
# 	--name 0208_ms_Qwen2.5-1.5B_lud_template_qwen_lm_head_lorap16_ep5_lr5e4_bs4 \
# 	--dataset $union_conversations_v5_norm_markdown,$union_conversations_v4_ll_markdown,$diting_v1_markdown,$liantong_conversations_v1_markdown \
# 	--gradient_accumulation_steps=4 --max_length=4096 --epochs=5 --batch_size 4 --lr=5e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset $union_conversations_v5_dev_markdown --eval_strategy steps --eval_steps 500 \

# ./mstrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen2_5 --lora_rank 32 --lora_target all-linear,all-embedding,lm_head --loraplus_lr_ratio 16 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-1.5B/ \
# 	--name 0208_ms_Qwen2.5-1.5B_lud_template_qwen_lm_head_lorap16_ep5_lr1e3_bs4 \
# 	--dataset $union_conversations_v5_norm_markdown,$union_conversations_v4_ll_markdown,$diting_v1_markdown,$liantong_conversations_v1_markdown \
# 	--gradient_accumulation_steps=4 --max_length=4096 --epochs=5 --batch_size 4 --lr=1e-3 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset $union_conversations_v5_dev_markdown --eval_strategy steps --eval_steps 500 \

# ./mstrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen2_5 --lora_rank 32 --lora_target all-linear,all-embedding,lm_head --loraplus_lr_ratio 16 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-1.5B/ \
# 	--name 0208_ms_Qwen2.5-1.5B_lud_template_qwen_lm_head_lorap16_ep5_lr5e4_bs4_gas1 \
# 	--dataset $union_conversations_v5_norm_markdown,$union_conversations_v4_ll_markdown,$diting_v1_markdown,$liantong_conversations_v1_markdown \
# 	--gradient_accumulation_steps=1 --max_length=4096 --epochs=5 --batch_size 4 --lr=5e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset $union_conversations_v5_dev_markdown --eval_strategy steps --eval_steps 500 \

# ./mstrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template default --lora_rank 32 --lora_target all-linear --loraplus_lr_ratio 16 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-1.5B/ \
# 	--name 0208_ms_Qwen2.5-1.5B_lud_template_default_lorap16_ep5_lr5e4_bs4 \
# 	--dataset $union_conversations_v5_norm_markdown,$union_conversations_v4_ll_markdown,$diting_v1_markdown,$liantong_conversations_v1_markdown \
# 	--gradient_accumulation_steps=4 --max_length=4096 --epochs=5 --batch_size 4 --lr=5e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset $union_conversations_v5_dev_markdown --eval_strategy steps --eval_steps 500 \

# ./mstrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen2_5 --lora_rank 32 --lora_target all-linear --loraplus_lr_ratio 16 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2-1.5B \
# 	--name 0210_ms_Qwen2-1.5B_lud_template_default_lorap16_ep5_lr5e4_bs4 \
# 	--dataset $union_conversations_v5_norm_markdown,$union_conversations_v4_ll_markdown,$diting_v1_markdown,$liantong_conversations_v1_markdown \
# 	--gradient_accumulation_steps=4 --max_length=4096 --epochs=5 --batch_size 4 --lr=5e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset $union_conversations_v5_dev_markdown --eval_strategy steps --eval_steps 500 \

export DATE=$(date "+%m%d")
echo "training scripts date ${DATE}"

export dataset_dir=/opt/nas/p/zhubin/DATA/train_data/no_think
# 构建dataset的名称
export alpaca_gpt4_zh_retain=/opt/nas/p/zhubin/DATA/train_data/no_think/alpaca_gpt4_zh_retain_no_think.json
export COIG_PC_core_summary_part=/opt/nas/p/zhubin/DATA/train_data/no_think/COIG_PC_core_summary_train_no_think.json
export callsum_v6_train_norm_markdown=/opt/nas/p/zhubin/DATA/train_data/no_think/conversations_full_iio_test_v6_markdown_no_think.json
export diting_v2_markdown=/opt/nas/p/zhubin/DATA/train_data/no_think/diting_full_iio_v2_markdown_no_think.json
export beta_nosise_v1_markdown=/opt/nas/p/zhubin/DATA/train_data/no_think/beta_noise_full_iio_v1_markdown_no_think.json
export callsum_v6_test_markdown=/opt/nas/p/zhubin/DATA/train_data/no_think/conversations_full_iio_test_v6_markdown_no_think.json


./mstrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen3 --lora_rank 32 --lora_alpha 1 --lora_target all-linear --loraplus_lr_ratio 16 \
	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen3-1.7B/ \
	--name "ms_${DATE}_Qwen3-1.7B_accdb_neft5_r32_lorap16_ep3_lr1e4_bs4" \
	--dataset $alpaca_gpt4_zh_retain,$COIG_PC_core_summary_part,$diting_v2_markdown,$callsum_v6_train_norm_markdown,$beta_nosise_v1_markdown \
	--gradient_accumulation_steps=4 --max_length=4096 --epochs=3 --batch_size 4 --lr=1e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
	--neftune_noise_alpha 5 \
	--eval_dataset $callsum_v6_test_markdown --eval_strategy steps --eval_steps 500

./mstrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen3 --lora_rank 32 --lora_alpha 1 --lora_target all-linear --loraplus_lr_ratio 16 \
	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen3-4B/ \
	--name "ms_${DATE}_Qwen3-4B_accdb_neft5_r32_lorap16_ep3_lr1e4_bs4" \
	--dataset $alpaca_gpt4_zh_retain,$COIG_PC_core_summary_part,$diting_v2_markdown,$callsum_v6_train_norm_markdown,$beta_nosise_v1_markdown \
	--gradient_accumulation_steps=4 --max_length=4096 --epochs=3 --batch_size 4 --lr=1e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
	--neftune_noise_alpha 5 \
	--eval_dataset $callsum_v6_test_markdown --eval_strategy steps --eval_steps 500
