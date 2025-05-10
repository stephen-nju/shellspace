# export dataset_dir=/opt/nas/p/zhubin/DATA/train_data/openai
# export union_conversations_v5_dev_markdown=$dataset_dir/union_conversations_v5/union_conversation_v5.4_iio_llf_dev_openai_markdownv1.jsonl
# export alpaca_gpt4_zh_retain_eval=$dataset_dir/alpaca_gpt4_zh_retain_eval.jsonl

# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/swifttest/v3-20250206-103024/checkpoint-1586

# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0206_ms_Qwen2.5-1.5B_lud_lorap16_ep3_lr2e4_bs4/v2-20250207-093336/checkpoint-199
# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0206_ms_Qwen2.5-1.5B_lud_lorap16_ep3_lr2e4_bs4/v2-20250207-093336/checkpoint-398
# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0206_ms_Qwen2.5-1.5B_lud_lorap16_ep3_lr2e4_bs4/v2-20250207-093336/checkpoint-594

# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0206_ms_Qwen2.5-1.5B_Instruct_lud_lorap16_ep3_lr2e4_bs4/v0-20250207-101310/checkpoint-199
# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0206_ms_Qwen2.5-1.5B_Instruct_lud_lorap16_ep3_lr2e4_bs4/v0-20250207-101310/checkpoint-398
# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0206_ms_Qwen2.5-1.5B_Instruct_lud_lorap16_ep3_lr2e4_bs4/v0-20250207-101310/checkpoint-594

# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0206_ms_Qwen2.5-3B_lud_lorap16_ep3_lr2e4_bs4/v0-20250207-105306/checkpoint-199
# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0206_ms_Qwen2.5-3B_lud_lorap16_ep3_lr2e4_bs4/v0-20250207-105306/checkpoint-398
# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0206_ms_Qwen2.5-3B_lud_lorap16_ep3_lr2e4_bs4/v0-20250207-105306/checkpoint-594

# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0206_ms_Qwen2.5-3B-Instruct_lud_lorap16_ep3_lr2e4_bs4/v0-20250207-115554/checkpoint-199
# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0206_ms_Qwen2.5-3B-Instruct_lud_lorap16_ep3_lr2e4_bs4/v0-20250207-115554/checkpoint-398
# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0206_ms_Qwen2.5-3B-Instruct_lud_lorap16_ep3_lr2e4_bs4/v0-20250207-115554/checkpoint-594

# export union_conversations_v5_norm_markdown=$dataset_dir/union_conversations_v5/union_conversations_train_norm_v5_openai_markdownv1.jsonl
# ./mseval.sh --eval_dataset $union_conversations_v5_norm_markdown#128 \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0206_ms_Qwen2.5-1.5B_lud_lorap16_ep3_lr2e4_bs4/v2-20250207-093336/checkpoint-199

# ./mseval.sh --eval_dataset $union_conversations_v5_norm_markdown#128 \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0206_ms_Qwen2.5-1.5B_lud_lorap16_ep3_lr2e4_bs4/v2-20250207-093336/checkpoint-398

# ./mseval.sh --eval_dataset $union_conversations_v5_norm_markdown#128 \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0206_ms_Qwen2.5-1.5B_lud_lorap16_ep3_lr2e4_bs4/v2-20250207-093336/checkpoint-594

# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0207_ms_Qwen2.5-1.5B_lud_full_ep3_lr2e5_bs4/v0-20250207-172106/checkpoint-199
# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0207_ms_Qwen2.5-1.5B_lud_full_ep3_lr2e5_bs4/v0-20250207-172106/checkpoint-398
# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0207_ms_Qwen2.5-1.5B_lud_full_ep3_lr2e5_bs4/v0-20250207-172106/checkpoint-594

# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0207_ms_Qwen2.5-1.5B_lud_template_def_lorap16_ep3_lr2e4_bs4/v0-20250207-175242/checkpoint-199
# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0207_ms_Qwen2.5-1.5B_lud_template_def_lorap16_ep3_lr2e4_bs4/v0-20250207-175242/checkpoint-398
# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0207_ms_Qwen2.5-1.5B_lud_template_def_lorap16_ep3_lr2e4_bs4/v0-20250207-175242/checkpoint-594

# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0207_ms_Qwen2.5-1.5B_lud_template_qwen_lm_head_lorap16_ep3_lr2e4_bs4/v0-20250207-183215/checkpoint-199
# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0207_ms_Qwen2.5-1.5B_lud_template_qwen_lm_head_lorap16_ep3_lr2e4_bs4/v0-20250207-183215/checkpoint-398
# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0207_ms_Qwen2.5-1.5B_lud_template_qwen_lm_head_lorap16_ep3_lr2e4_bs4/v0-20250207-183215/checkpoint-594

##alpaca训练集

# ./mseval.sh --finetuning_type lora --eval_dataset $alpaca_gpt4_zh_retain_eval \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0207_ms_Qwen2.5-1.5B_alpaca_lorap16_ep3_lr2e4_bs4/v0-20250208-094315/checkpoint-333

# ./mseval.sh --finetuning_type lora --eval_dataset $alpaca_gpt4_zh_retain_eval \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0207_ms_Qwen2.5-1.5B_alpaca_lorap16_ep3_lr2e4_bs4/v0-20250208-094315/checkpoint-666

# ./mseval.sh  --finetuning_type lora --eval_dataset $alpaca_gpt4_zh_retain_eval \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0207_ms_Qwen2.5-1.5B_alpaca_lorap16_ep3_lr2e4_bs4/v0-20250208-094315/checkpoint-999

##alpaca通话摘要数据
# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0207_ms_Qwen2.5-1.5B_alpaca_lorap16_ep3_lr2e4_bs4/v0-20250208-094315/checkpoint-333
# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0207_ms_Qwen2.5-1.5B_alpaca_lorap16_ep3_lr2e4_bs4/v0-20250208-094315/checkpoint-666
# ./mseval.sh --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0207_ms_Qwen2.5-1.5B_alpaca_lorap16_ep3_lr2e4_bs4/v0-20250208-094315/checkpoint-999

##全量微调
# ./mseval.sh  --finetuning_type full --eval_dataset $union_conversations_v5_dev_markdown \
# --model=/opt/nas/p/zhubin/saved_checkpoint/0207_ms_Qwen2.5-1.5B_lud_full_ep3_lr2e5_bs4/v0-20250207-172106/checkpoint-199
# ./mseval.sh  --finetuning_type full --eval_dataset $union_conversations_v5_dev_markdown \
# --model=/opt/nas/p/zhubin/saved_checkpoint/0207_ms_Qwen2.5-1.5B_lud_full_ep3_lr2e5_bs4/v0-20250207-172106/checkpoint-398
# ./mseval.sh  --finetuning_type full --eval_dataset $union_conversations_v5_dev_markdown \
# --model=/opt/nas/p/zhubin/saved_checkpoint/0207_ms_Qwen2.5-1.5B_lud_full_ep3_lr2e5_bs4/v0-20250207-172106/checkpoint-594

# ./mseval.sh --finetuning_type lora --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0208_ms_Qwen2.5-1.5B_lud_lorap16_ep5_lr5e4_bs4/v0-20250208-114758/checkpoint-199
# ./mseval.sh --finetuning_type lora --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0208_ms_Qwen2.5-1.5B_lud_lorap16_ep5_lr5e4_bs4/v0-20250208-114758/checkpoint-398
# ./mseval.sh --finetuning_type lora --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0208_ms_Qwen2.5-1.5B_lud_lorap16_ep5_lr5e4_bs4/v0-20250208-114758/checkpoint-597
# ./mseval.sh --finetuning_type lora --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0208_ms_Qwen2.5-1.5B_lud_lorap16_ep5_lr5e4_bs4/v0-20250208-114758/checkpoint-796
# ./mseval.sh --finetuning_type lora --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0208_ms_Qwen2.5-1.5B_lud_lorap16_ep5_lr5e4_bs4/v0-20250208-114758/checkpoint-990

# ./mseval.sh --finetuning_type lora --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0208_ms_Qwen2.5-1.5B_lud_lorap16_ep5_lr1e5_bs4/v0-20250208-125030/checkpoint-199
# ./mseval.sh --finetuning_type lora --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0208_ms_Qwen2.5-1.5B_lud_lorap16_ep5_lr1e5_bs4/v0-20250208-125030/checkpoint-398
# ./mseval.sh --finetuning_type lora --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0208_ms_Qwen2.5-1.5B_lud_lorap16_ep5_lr1e5_bs4/v0-20250208-125030/checkpoint-597
# ./mseval.sh --finetuning_type lora --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0208_ms_Qwen2.5-1.5B_lud_lorap16_ep5_lr1e5_bs4/v0-20250208-125030/checkpoint-796
# ./mseval.sh --finetuning_type lora --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0208_ms_Qwen2.5-1.5B_lud_lorap16_ep5_lr1e5_bs4/v0-20250208-125030/checkpoint-990

# ./mseval.sh --finetuning_type lora --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0208_ms_Qwen2.5-1.5B_lud_template_qwen_lm_head_lorap16_ep5_lr5e4_bs4/v1-20250208-145520/checkpoint-597
#  ./mseval.sh --finetuning_type lora --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0208_ms_Qwen2.5-1.5B_lud_template_qwen_lm_head_lorap16_ep5_lr5e4_bs4/v1-20250208-145520/checkpoint-398

# ./mseval.sh --finetuning_type lora --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0208_ms_Qwen2.5-1.5B_lud_template_default_lorap16_ep5_lr5e4_bs4/v0-20250208-162453/checkpoint-199

# ./mseval.sh --finetuning_type lora --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0208_ms_Qwen2.5-1.5B_lud_template_default_lorap16_ep5_lr5e4_bs4/v0-20250208-162453/checkpoint-398

# ./mseval.sh --finetuning_type lora --eval_dataset $union_conversations_v5_dev_markdown \
# --adapters=/opt/nas/p/zhubin/saved_checkpoint/0210_ms_Qwen2-1.5B_lud_template_default_lorap16_ep5_lr5e4_bs4/v0-20250210-094605/checkpoint-199

export callsum_v6_test_markdown=/opt/nas/p/zhubin/DATA/train_data/no_think/conversations_full_iio_test_v6_markdown_no_think.json

./mseval.sh --finetuning_type lora --eval_dataset $callsum_v6_test_markdown \
--adapters=/opt/nas/p/zhubin/saved_checkpoint/ms_0430_Qwen3-1.7B_accdb_neft5_r32_lorap16_ep3_lr1e4_bs4/v0-20250430-094000/checkpoint-420
