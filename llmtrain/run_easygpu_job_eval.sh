exec >/opt/nas/p/zhubin/easyjobLog/pytorch_eval.log
echo "hoststr==${hoststr}"
echo "pwd==$(pwd)"
echo $hoststr | sed 's/,/\n/g' >/opt/nas/p/zhubin/code/Llmtrain/cache/hostfile
export HF_HOME=/opt/local/data/
# vc -proxy open
# 配置wandb
# export WANDB_API_KEY=04c30ea6f4b2e78a13aa48f65ddeff512213be6c
# export WANDB_MODE=offline
# wandb offline
# ./magiclmnano_sft.sh --do_train --do_eval --stage sft --hostfile /opt/nas/p/zhubin/code/Llmtrain/cache/hostfile \
# 	--finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target wqkv,w1,w2,w3 --loraplus_lr_ratio 16 \
# 	--max_samples 100 --batch_size 2 \
# 	--name=TEST --dataset baichuan_multiturn_demo \
# 	--gradient_accumulation_steps 1 --cutoff_len=4096 --epochs=3 --lr=1e-4 --save_strategy=epoch --warmup_ratio 0.03 --save_total_limit=3 --eval_dataset \
# 	union_conversations_v5_dev --eval_strategy=steps --eval_steps=300

# ./llmtrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template glm-edge --lora_rank 32 --lora_target q_proj,k_proj,v_proj --loraplus_lr_ratio 16 \
# 	--name=20150121_glm4edge_acfvculdu_loraplus16_ep3_lr5e4_bs4 \
# 	--model_name_or_path /opt/nas/n/zhubin/DATA/models/THUDM/glm-edge-4b-chat/ \
# 	--dataset alpace_gpt4_zh_retain,COIG_PC_core_summary_part,firefly_summary_part,vcsum_headlines,csds_dialogue,union_conversations_v5_norm_markdown,liantong_conversations_v1_markdown,diting_v1_markdown,union_conversations_v4_ll_markdown \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=3 --lr=5e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset union_conversations_v5_dev --eval_strategy=steps --eval_steps=300 \
# 	--hostfile /opt/nas/p/zhubin/code/Llmtrain/cache/hostfile

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-3B/ \
# 	--adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/20250122_qwen2.5_3B_acfvculdu_loraplus16_ep3_lr2e4_bs4/checkpoint-1054/ \
# 	--eval_dataset union_conversations_v5_dev_markdown --finetuning_type lora

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-3B/ \
# 	--adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/20250122_qwen2.5_3B_acfvculdu_loraplus16_ep3_lr2e4_bs4/checkpoint-2109/ \
# 	--eval_dataset union_conversations_v5_dev_markdown --finetuning_type lora

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-3B/ \
# 	--adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/20250122_qwen2.5_3B_acfvculdu_loraplus16_ep3_lr2e4_bs4/checkpoint-3162/ \
# 	--eval_dataset union_conversations_v5_dev_markdown --finetuning_type lora

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2-1.5B/ \
# 	--adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0123_qwen2.5_1.5B_auldu_rvt_loraplus16_ep3_lr2e4_bs4/checkpoint-198/ \
# 	--eval_dataset union_conversations_v5_dev_markdown --finetuning_type lora

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2-1.5B/ \
# 	--adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0123_qwen2.5_1.5B_auldu_rvt_loraplus16_ep3_lr2e4_bs4/checkpoint-396/ \
# 	--eval_dataset union_conversations_v5_dev_markdown --finetuning_type lora

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2-1.5B/ \
# 	--adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0123_qwen2.5_1.5B_auldu_rvt_loraplus16_ep3_lr2e4_bs4/checkpoint-594/ \
# 	--eval_dataset union_conversations_v5_dev_markdown --finetuning_type lora

# 3B模型测评
# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-3B/ \
# 	--adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0123_qwen2.5_3B_auldu_loraplus16_ep3_lr2e4_bs4/checkpoint-198/ \
# 	--eval_dataset union_conversations_v5_dev_markdown --finetuning_type lora

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-3B/ \
# 	--adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0123_qwen2.5_3B_auldu_loraplus16_ep3_lr2e4_bs4/checkpoint-396/ \
# 	--eval_dataset union_conversations_v5_dev_markdown --finetuning_type lora

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-3B/ \
# 	--adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0123_qwen2.5_3B_auldu_loraplus16_ep3_lr2e4_bs4/checkpoint-594 \
# 	--eval_dataset union_conversations_v5_dev_markdown --finetuning_type lora

# ./magiclmnano_eval.sh --hoststr "$hoststr" \
# 	--adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/20250117_magiclmnano_acfvculdu_loraplus16_ep3_lr1e4_bs4/checkpoint-1054/ \
# 	--eval_dataset union_conversations_v5_dev_markdown --finetuning_type lora

# ./magiclmnano_eval.sh --hoststr "$hoststr" \
# 	--adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/20250117_magiclmnano_acfvculdu_loraplus16_ep3_lr1e4_bs4/checkpoint-2109/ \
# 	--eval_dataset union_conversations_v5_dev_markdown --finetuning_type lora

# ./magiclmnano_eval.sh --hoststr "$hoststr" \
# 	--adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/20250117_magiclmnano_acfvculdu_loraplus16_ep3_lr1e4_bs4/checkpoint-3162/ \
# 	--eval_dataset union_conversations_v5_dev_markdown --finetuning_type lora

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0210_Qwen2.5-14B-Instruct_neft_acfvculddu_markdown_ep3_lr4e5_bs1/checkpoint-1146 \
# 	--eval_dataset union_conversations_v5_dev_markdown --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0210_Qwen2.5-14B-Instruct_neft_acfvculddu_markdown_ep3_lr4e5_bs1/checkpoint-2292 \
# 	--eval_dataset union_conversations_v5_dev_markdown --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0210_Qwen2.5-14B-Instruct_neft_acfvculddu_markdown_ep3_lr4e5_bs1/checkpoint-3438 \
# 	--eval_dataset union_conversations_v5_dev_markdown --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0222_Qwen2.5-14B-Instruct_neft_accdb_markdown_ep3_lr1e5_bs1/checkpoint-714 \
# 	--eval_dataset callsum_v6_test_markdown --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0222_Qwen2.5-14B-Instruct_neft_accdb_markdown_ep3_lr1e5_bs1/checkpoint-1429 \
# 	--eval_dataset callsum_v6_test_markdown --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0222_Qwen2.5-14B-Instruct_neft_accdb_markdown_ep3_lr1e5_bs1/checkpoint-2142 \
# 	--eval_dataset callsum_v6_test_markdown --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0222_Qwen2.5-14B-Instruct_neft_accdb_markdown_ep3_lr2e5_bs1/checkpoint-714 \
# 	--eval_dataset callsum_v6_test_markdown --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0222_Qwen2.5-14B-Instruct_neft_accdb_markdown_ep3_lr2e5_bs1/checkpoint-1429 \
# 	--eval_dataset callsum_v6_test_markdown --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0222_Qwen2.5-14B-Instruct_neft_accdb_markdown_ep3_lr2e5_bs1/checkpoint-2142 \
# 	--eval_dataset callsum_v6_test_markdown --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0327_Qwen2.5-14B-Instruct_neft_accdb_markdown_wd_ep3_lr7e6_bs1/checkpoint-711 \
# 	--eval_dataset diting_v7.2_markdown --output_name "business_test_v2" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0327_Qwen2.5-14B-Instruct_neft_accdb_markdown_wd_ep3_lr7e6_bs1/checkpoint-711 \
# 	--eval_dataset callsum_v7_test_markdown --output_name "callsum_v7_test_markdown" --finetuning_type full
# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0327_Qwen2.5-14B-Instruct_neft_accdb_markdown_wd_ep3_lr7e6_bs1/checkpoint-1422 \
# 	--eval_dataset diting_v7.2_markdown --output_name "business_test_v2" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0327_Qwen2.5-14B-Instruct_neft_accdb_markdown_wd_ep3_lr7e6_bs1/checkpoint-1422 \
# 	--eval_dataset callsum_v7_test_markdown --output_name "callsum_v7_test_markdown" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0327_Qwen2.5-14B-Instruct_neft_accdb_markdown_wd_ep3_lr7e6_bs1/checkpoint-2130 \
# 	--eval_dataset diting_v7.2_markdown --output_name "business_test_v2" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0327_Qwen2.5-14B-Instruct_neft_accdb_markdown_wd_ep3_lr7e6_bs1/checkpoint-2130 \
# 	--eval_dataset callsum_v7_test_markdown --output_name "callsum_v7_test_markdown" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0328_Qwen2.5-14B-Instruct_neft_accdb_markdown_wd_ep3_lr7e6_bs1/checkpoint-1426 \
# 	--eval_dataset callsum_v7_test_markdown --output_name "callsum_v7_test_markdown" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0328_Qwen2.5-14B-Instruct_neft_accdb_markdown_wd_ep3_lr7e6_bs1/checkpoint-1426 \
# 	--eval_dataset diting_v7.2_markdown --output_name "business_test_v2" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0329_Qwen2.5-14B-Instruct_neft_icdb_markdown_wd_ep2_lr7e6_bs1/checkpoint-1079 \
# 	--eval_dataset callsum_v7_test_markdown --output_name "callsum_v7_test_markdown" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0329_Qwen2.5-14B-Instruct_neft_icdb_markdown_wd_ep2_lr7e6_bs1/checkpoint-1079 \
# 	--eval_dataset diting_v7.2_markdown --output_name "business_test_v2" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0402_Qwen2.5-14B-Instruct_neft5_accdb_markdown_wd_ep3_lr7e6_bs1/checkpoint-719 \
# 	--eval_dataset callsum_v7_test_markdown --output_name "callsum_v7_test_markdown" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0402_Qwen2.5-14B-Instruct_neft5_accdb_markdown_wd_ep3_lr7e6_bs1/checkpoint-719 \
# 	--eval_dataset diting_v7.2_markdown --output_name "business_test_v2" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0402_Qwen2.5-14B-Instruct_neft5_accdb_markdown_wd_ep3_lr7e6_bs1/checkpoint-1438 \
# 	--eval_dataset callsum_v7_test_markdown --output_name "callsum_v7_test_markdown" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0402_Qwen2.5-14B-Instruct_neft5_accdb_markdown_wd_ep3_lr7e6_bs1/checkpoint-1438 \
# 	--eval_dataset diting_v7.2_markdown --output_name "business_test_v2" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0402_Qwen2.5-14B-Instruct_neft5_accdb_markdown_wd_ep3_lr7e6_bs1/checkpoint-2154 \
# 	--eval_dataset callsum_v7_test_markdown --output_name "callsum_v7_test_markdown" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0402_Qwen2.5-14B-Instruct_neft5_accdb_markdown_wd_ep3_lr7e6_bs1/checkpoint-2154 \
# 	--eval_dataset diting_v7.2_markdown --output_name "business_test_v2" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0423_Qwen2.5-14B-Instruct_neft5_afcdbzd_v8_v7_markdown_wd_ep2_lr7e6_bs1/checkpoint-1355/ \
# 	--eval_dataset callsum_v8_test_markdown --output_name "callsum_v8_test_markdown" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0423_Qwen2.5-14B-Instruct_neft5_afcdbzd_v8_v7_markdown_wd_ep2_lr7e6_bs1/checkpoint-1355/ \
# 	--eval_dataset diting_v8.2_markdown --output_name "business_test_v2" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0423_Qwen2.5-14B-Instruct_neft5_afcdbzd_v8_v7_markdown_wd_ep2_lr7e6_bs1/checkpoint-2708/ \
# 	--eval_dataset callsum_v8_test_markdown --output_name "callsum_v8_test_markdown" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0423_Qwen2.5-14B-Instruct_neft5_afcdbzd_v8_v7_markdown_wd_ep2_lr7e6_bs1/checkpoint-2708/ \
# 	--eval_dataset diting_v8.2_markdown --output_name "business_test_v2" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0423_Qwen2.5-14B-Instruct_neft5_afcdbzd_v8_v7_markdown_wd_ep2_lr1e5_bs1/checkpoint-2708/ \
# 	--eval_dataset callsum_v8_test_markdown --output_name "callsum_v8_test_markdown" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0423_Qwen2.5-14B-Instruct_neft5_afcdbzd_v8_v7_markdown_wd_ep2_lr1e5_bs1/checkpoint-2708/ \
# 	--eval_dataset diting_v8.2_markdown --output_name "business_test_v2" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --template qwen3 --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen3-4B/ \
# 	--finetuning_type lora --adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0429_Qwen3-4B-Instruct_neft5_accdb_markdown_lora_ep2_lr2e4_bs4/checkpoint-1430/ \
# 	--eval_dataset callsum_v6_test_markdown --output_name "callsum_v6_test_markdown"

# ./Qwen_eval.sh --hoststr "$hoststr" --template qwen3 --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen3-4B/ \
# 	--finetuning_type lora --adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0429_Qwen3-4B-Instruct_neft5_accdb_markdown_lora_ep2_lr2e4_bs4/checkpoint-715/ \
# 	--eval_dataset callsum_v6_test_markdown --output_name "callsum_v6_test_markdown"

# ./Qwen_eval.sh --hoststr "$hoststr" --template qwen3 --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen3-4B/ \
# 	--finetuning_type lora --adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0429_Qwen3-4B-Instruct_neft5_accdb_markdown_lora_ep2_lr2e4_bs4/checkpoint-2142/ \
# 	--eval_dataset callsum_v6_test_markdown --output_name "callsum_v6_test_markdown"

# ./Qwen_eval.sh --hoststr "$hoststr" --template qwen3 --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen3-1.7B/ \
# 	--finetuning_type lora --adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0429_Qwen3-1.7B-Instruct_neft5_accdb_markdown_lora_ep2_lr2e4_bs4/checkpoint-715/ \
# 	--eval_dataset callsum_v6_test_markdown --output_name "callsum_v6_test_markdown"

# ./Qwen_eval.sh --hoststr "$hoststr" --template qwen3 --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen3-1.7B/ \
# 	--finetuning_type lora --adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0429_Qwen3-1.7B-Instruct_neft5_accdb_markdown_lora_ep2_lr2e4_bs4/checkpoint-1430/ \
# 	--eval_dataset callsum_v6_test_markdown --output_name "callsum_v6_test_markdown"

# ./Qwen_eval.sh --hoststr "$hoststr" --template qwen3 --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen3-1.7B/ \
# 	--finetuning_type lora --adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0429_Qwen3-1.7B-Instruct_neft5_accdb_markdown_lora_ep2_lr2e4_bs4/checkpoint-2142 \
# 	--eval_dataset callsum_v6_test_markdown --output_name "callsum_v6_test_markdown"

# ./Qwen_eval.sh --hoststr "$hoststr" --template qwen3 --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen3-1.7B/ \
# 	--finetuning_type lora --adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0430_Qwen3-1.7B-Instruct_neft5_cdb_markdown_lora32_alpha1_ep2_lr2e4_bs4/checkpoint-299 \
# 	--eval_dataset callsum_v6_test_markdown --output_name "callsum_v6_test_markdown"

# ./Qwen_eval.sh --hoststr "$hoststr" --template qwen3 --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen3-1.7B/ \
# 	--finetuning_type lora --adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0430_Qwen3-1.7B-Instruct_neft5_cdb_markdown_lora32_alpha1_ep2_lr2e4_bs4/checkpoint-596 \
# 	--eval_dataset callsum_v6_test_markdown --output_name "callsum_v6_test_markdown"

# ./Qwen_eval.sh --hoststr "$hoststr" --template qwen3 --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen3-4B/ \
# 	--finetuning_type lora --adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0430_Qwen3-4B-Instruct_neft5_cdb_markdown_lora32_alpha1_ep2_lr2e4_bs4/checkpoint-299 \
# 	--eval_dataset callsum_v6_test_markdown --output_name "callsum_v6_test_markdown"

# ./Qwen_eval.sh --hoststr "$hoststr" --template qwen3 --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen3-4B/ \
# 	--finetuning_type lora --adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0430_Qwen3-4B-Instruct_neft5_cdb_markdown_lora32_alpha1_ep2_lr2e4_bs4/checkpoint-299 \
# 	--eval_dataset callsum_v6_test_markdown --output_name "callsum_v6_test_markdown"


# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0508_Qwen2.5-14B-Instruct_neft5_afcdbzd_v9_v7_markdown_wd_ep2_lr7e6_bs1/checkpoint-1355 \
# 	--eval_dataset callsum_v9_test_markdown --output_name "callsum_v9_test_markdown" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0508_Qwen2.5-14B-Instruct_neft5_afcdbzd_v9_v7_markdown_wd_ep2_lr7e6_bs1/checkpoint-1355 \
# 	--eval_dataset diting_v9.2_markdown --output_name "business_test_v2" --finetuning_type full


# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0508_Qwen2.5-14B-Instruct_neft5_afcdbzd_v9_v7_markdown_wd_ep2_lr7e6_bs1/checkpoint-2708 \
# 	--eval_dataset callsum_v9_test_markdown --output_name "callsum_v9_test_markdown" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0508_Qwen2.5-14B-Instruct_neft5_afcdbzd_v9_v7_markdown_wd_ep2_lr7e6_bs1/checkpoint-2708 \
# 	--eval_dataset diting_v9.2_markdown --output_name "business_test_v2" --finetuning_type full


# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0509_Qwen2.5-14B-Instruct_neft5_afcdbzd_v9_v7_markdown_wd_ep2_lr1e5_bs1/checkpoint-1355 \
# 	--eval_dataset callsum_v9_test_markdown --output_name "callsum_v9_test_markdown" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0509_Qwen2.5-14B-Instruct_neft5_afcdbzd_v9_v7_markdown_wd_ep2_lr1e5_bs1/checkpoint-1355 \
# 	--eval_dataset diting_v9.2_markdown --output_name "business_test_v2" --finetuning_type full


# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0509_Qwen2.5-14B-Instruct_neft5_afcdbzd_v9_v7_markdown_wd_ep2_lr1e5_bs1/checkpoint-2708 \
# 	--eval_dataset callsum_v9_test_markdown --output_name "callsum_v9_test_markdown" --finetuning_type full

# ./Qwen_eval.sh --hoststr "$hoststr" --model_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0509_Qwen2.5-14B-Instruct_neft5_afcdbzd_v9_v7_markdown_wd_ep2_lr1e5_bs1/checkpoint-2708 \
# 	--eval_dataset diting_v9.2_markdown --output_name "business_test_v2" --finetuning_type full
