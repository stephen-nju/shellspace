exec >/opt/nas/p/zhubin/easyjobLog/pytorch.log
echo "hoststr==${hoststr}"
echo $hoststr | sed 's/,/\n/g' >/opt/nas/p/zhubin/code/Llmtrain/cache/hostfile
export HF_HOME=/opt/local/data/
vc -proxy open
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

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/cache/hostfile \
#     --stage sft --finetuning_type full --name=0228_Qwen2.5-14B-Instruct_neft_accdb_markdown_ep3_lr1e5_bs1 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset alpace_gpt4_zh_retain,COIG_PC_core_summary_part,distill_r1_4w,callsum_v6_train_markdown,diting_v2_markdown,beta_noise_v1_markdown \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 2 --lr 1e-5 --save_strategy epoch --save_total_limit 100 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v6_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.1

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/cache/hostfile \
#     --stage sft --finetuning_type full --name=0301_Qwen2.5-14B-Instruct_neft_cdb_markdown_ep3_lr1e5_bs1 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset callsum_v6_train_markdown,diting_v2_markdown,beta_noise_v1_markdown \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 3 --lr 1e-5 --save_strategy epoch --save_total_limit 100 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v6_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/cache/hostfile \
#     --stage sft --finetuning_type full --name=0301_Qwen2.5-14B-Instruct_neft_cdb_markdown_wd_ep3_lr7e6_bs1 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset callsum_v6_train_markdown,diting_v2_markdown,beta_noise_v1_markdown \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 3 --lr 7e-6 --weight_decay 0.1 \
# 	--save_strategy epoch --save_total_limit 100 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v6_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/cache/hostfile \
# 	--stage sft --finetuning_type full --name=TEST \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset alpace_gpt4_zh_retain,COIG_PC_core_summary_part,callsum_v7.1_train_markdown,diting_v7.2_markdown,beta_noise_v7.1_markdown \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 3 --lr 7e-6 --weight_decay 0.1 \
# 	--save_strategy epoch --save_total_limit 100 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v7_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/cache/hostfile \
# 	--stage sft --finetuning_type full --name=0328_Qwen2.5-14B-Instruct_neft_accdb_markdown_wd_ep3_lr7e6_bs1 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset alpace_gpt4_zh_retain,COIG_PC_core_summary_part,callsum_v7.1_train_markdown,diting_v7.2_markdown,beta_noise_v7.1_markdown \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 3 --lr 7e-6 --weight_decay 0.1 \
# 	--save_strategy epoch --save_total_limit 100 --seed 42 \
# 	--neftune_noise_alpha 15 --eval_dataset callsum_v7_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.05


# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/cache/hostfile \
# 	--stage sft --finetuning_type full --name=0402_Qwen2.5-14B-Instruct_neft_accdb_markdown_wd_ep3_lr7e6_bs1 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset Inf_gen_100k,callsum_v7.1_train_markdown,diting_v7.2_markdown,beta_noise_v7.1_markdown \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 3 --lr 7e-6 --weight_decay 0.1 \
# 	--save_strategy epoch --save_total_limit 100 --seed 42 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v7_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03


./Qwen_eval.sh --hoststr "$hoststr" --template qwen3 --model_name_or_path /opt/nas/p/models/Qwen_models/Qwen3-4B/ \
	--finetuning_type lora --adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/0430_Qwen3-4B-Instruct_neft5_cdb_markdown_lora32_alpha1_ep2_lr2e4_bs4/checkpoint-300 \
	--eval_dataset callsum_v6_test_markdown --output_name "callsum_v6_test_markdown"
