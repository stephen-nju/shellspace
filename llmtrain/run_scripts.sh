#!/usr/bin/bash

export DATE=$(date "+%m%d")
echo "training scripts date ${DATE}"
#magiclmnano --do_train --do_eval --stage sft --finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target wqkv,w1,w2,w3 \
# 	--name=1202_magiclm_nano_neft_cudsls_lora_ep3_lr2e4_bs4 --dataset COIG_CQIA_train,union_conversations_v4_norm,dialogsum_chinese,samsum_chinese,liantong_conversations_v1,swindle_data_v1 --neftune_noise_alpha 5 \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=3 --lr=2e-4 --save_strategy=epoch --warmup_ratio 0.03 --save_total_limit=3 --eval_dataset \
# 	union_conversations_v4_dev --eval_strategy=steps --eval_steps=300 --include node12

# wait

# magiclmnano --do_train --do_eval --stage sft --finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target wqkv,w1,w2,w3 \
# 	--name=1202_magiclm_nano_neft_cudsls_lora_ep3_lr5e4_bs4 --dataset COIG_CQIA_train,union_conversations_v4_norm,dialogsum_chinese,samsum_chinese,liantong_conversations_v1,swindle_data_v1 --neftune_noise_alpha 5 \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=3 --lr=5e-4 --save_strategy=epoch --warmup_ratio 0.03 --save_total_limit=3 --eval_dataset \
# 	union_conversations_v4_dev --eval_strategy=steps --eval_steps=300 --include node12

# magiclmnano --do_train --do_eval --stage sft --finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target wqkv,w1,w2,w3 \
# 	--name=1202_magiclm_nano_neft_cudsls_lora_ep3_lr6e3_bs4 --dataset COIG_CQIA_train,union_conversations_v4_norm,dialogsum_chinese,samsum_chinese,liantong_conversations_v1,swindle_data_v1 --neftune_noise_alpha 5 \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=3 --lr=6e-3 --save_strategy=epoch --warmup_ratio 0.03 --save_total_limit=3 --eval_dataset \
# 	union_conversations_v4_dev --eval_strategy=steps --eval_steps=300 --include node12

# wait
# magiclmnano --do_train --do_eval --stage sft --finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target wqkv,w1,w2,w3 \
# 	--name=1202_magiclm_nano_neft_cudsls_lora_ep3_lr7e4_bs4 --dataset COIG_CQIA_train,union_conversations_v4_norm,dialogsum_chinese,samsum_chinese,liantong_conversations_v1,swindle_data_v1 --neftune_noise_alpha 5 \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=3 --lr=7e-4 --save_strategy=epoch --warmup_ratio 0.03 --save_total_limit=3 --eval_dataset \
# 	union_conversations_v4_dev --eval_strategy=steps --eval_steps=300 --include node12

# wait

# magiclmnano --do_train --do_eval --stage sft --finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target wqkv,w1,w2,w3 --loraplus_lr_ratio 16 \
# 	--name=1202_magiclm_nano_neft_cudsls_loraplus16_ep3_lr1e4_bs4 --dataset COIG_CQIA_train,union_conversations_v4_norm,dialogsum_chinese,samsum_chinese,liantong_conversations_v1,swindle_data_v1 --neftune_noise_alpha 5 \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=3 --lr=1e-4 --save_strategy=epoch --warmup_ratio 0.03 --save_total_limit=3 --eval_dataset \
# 	union_conversations_v4_dev --eval_strategy=steps --eval_steps=300 --include node12
# wait

# magiclmnano --do_train --do_eval --stage sft --finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target wqkv,w1,w2,w3,wo --loraplus_lr_ratio 16 \
# 	--name=1219_magiclm_nano_neft_cusl_loraplus16_ep3_lr1e4_bs4 --dataset COIG_CQIA_train,union_conversations_v4_norm,liantong_conversations_v1,swindle_data_v1 --neftune_noise_alpha 5 \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=3 --lr=1e-4 --save_strategy=epoch --warmup_ratio 0.03 --save_total_limit=3 --eval_dataset \
# 	union_conversations_v4_dev --eval_strategy=steps --eval_steps=300 --include node12

# magiclmnano --do_train --do_eval --stage sft --finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target wqkv,w1,w2,w3 --loraplus_lr_ratio 16 \
# 	--name=1202_magiclm_nano_cudsls_loraplus16_ep3_lr1e4_bs4 --dataset COIG_CQIA_train,union_conversations_v4_norm,dialogsum_chinese,samsum_chinese,liantong_conversations_v1,swindle_data_v1 \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=3 --lr=1e-4 --save_strategy=epoch --warmup_ratio 0.03 --save_total_limit=3 --eval_dataset \
# 	union_conversations_v4_dev --eval_strategy=steps --eval_steps=300 --include node12

# wait
# magiclmnano --do_train --do_eval --stage sft --finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target wqkv,w1,w2,w3 --loraplus_lr_ratio 8 \
# 	--name=1202_magiclm_nano_neft_cudsls_loraplus8_ep3_lr1e4_bs4 --dataset COIG_CQIA_train,union_conversations_v4_norm,dialogsum_chinese,samsum_chinese,liantong_conversations_v1,swindle_data_v1 --neftune_noise_alpha 5 \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=3 --lr=1e-4 --save_strategy=epoch --warmup_ratio 0.03 --save_total_limit=3 --eval_dataset \
# 	union_conversations_v4_dev --eval_strategy=steps --eval_steps=300 --include node12

# ./magiclmnano_predict.sh --finetuning_type lora --adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/1202_magiclm_nano_neft_cudsls_loraplus16_ep3_lr1e4_bs4 \
# 	--eval_dataset union_conversations_v4_norm \
# 	--output_dir=/home/jovyan/zhubin/code/DataGeneration/DPO/v4_norm \
# 	--include node12:0,1,2,3
# wait

# 温度采样
# ./magiclmnano_predict.sh --finetuning_type lora --temperature 1.5 --adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/1211_magiclm_nano_cusl_loraplus16_ep3_lr1e4_bs4/ \
# 	--eval_dataset dialogsum_chinese \
# 	--output_dir=/home/jovyan/zhubin/code/DataGeneration/DPO/dialogsum_chinese_temperature_1.5 \
# 	--include node12:0,1,2,3,4,5,6,7

# ./magiclmnano_predict.sh --finetuning_type lora --temperature 4 --adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/1230_magiclm_nano_neft_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--eval_dataset dialogsum_chinese_markdown \
# 	--output_dir=/opt/nas/p/zhubin/code/DataGeneration/DPO/dialogsum_chinese_temperature_4 \

# ./magiclmnano_predict.sh --finetuning_type lora --temperature 4 --adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/1230_magiclm_nano_neft_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--eval_dataset samsum_chinese_markdown \
# 	--output_dir=/opt/nas/p/zhubin/code/DataGeneration/DPO/dialogsum_chinese_temperature_4 \

# ./magiclmnano_predict.sh --finetuning_type lora --temperature 4 --adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/1230_magiclm_nano_neft_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--eval_dataset realconv_instruct_markdown \
# 	--output_dir=/opt/nas/p/zhubin/code/DataGeneration/DPO/realconv_instruct_markdown_temperature_4 \

# ./magiclmnano_predict.sh --finetuning_type lora --temperature 4 --adapter_name_or_path /opt/nas/p/zhubin/saved_checkpoint/1230_magiclm_nano_neft_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--eval_dataset naturalconv9k_instruct_markdown \
# 	--output_dir=/opt/nas/p/zhubin/code/DataGeneration/DPO/naturalconv9k_instruct_markdown_temperature_4 \

# wait
# ./magiclmnano_predict.sh --finetuning_type lora --temperature 1.5 --adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/1211_magiclm_nano_cusl_loraplus16_ep3_lr1e4_bs4/ \
# 	--eval_dataset samsum_chinese \
# 	--output_dir=/home/jovyan/zhubin/code/DataGeneration/DPO/samsum_chinese_temperature_1.5 \
# 	--include node12:0,1,2,3,4,5,6,7

# wait
# ./magiclmnano_predict.sh --finetuning_type lora --temperature 2 --adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/1211_magiclm_nano_cusl_loraplus16_ep3_lr1e4_bs4/ \
# 	--eval_dataset dialogsum_chinese \
# 	--output_dir=/home/jovyan/zhubin/code/DataGeneration/DPO/dialogsum_chinese_temperature_2 \
# 	--include node12:0,1,2,3,4,5,6,7

# wait
# ./magiclmnano_predict.sh --finetuning_type lora --temperature 2 --adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/1211_magiclm_nano_cusl_loraplus16_ep3_lr1e4_bs4/ \
# 	--eval_dataset samsum_chinese \
# 	--output_dir=/home/jovyan/zhubin/code/DataGeneration/DPO/samsum_chinese_temperature_2 \
# 	--include node12:0,1,2,3,4,5,6,7

# ./magiclmnano_predict.sh --finetuning_type lora --adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/1211_magiclm_nano_cusl_loraplus16_ep3_lr1e4_bs4/ \
# 	--eval_dataset COIG_CQIA_train \
# 	--output_dir=/home/jovyan/zhubin/code/DataGeneration/DPO/COIG_CQIA_train \
# 	--include node12:0,1,2,3,4,5,6,7

# ./magiclmnano_dpo.sh --do_train --stage dpo --finetuning_type lora --adapter_name_or_path=/home/jovyan/zhubin/saved_checkpoint/1202_magiclm_nano_neft_cudsls_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset conversation_dpo_v2 --epochs=3 --name=1210_magiclmnano_dpo_ep3_lr1e6_bs4 --save_total_limit=10 --warmup_ratio=0.03 --include=node12:0,1,2,3,4,5,6,7

# LORA训练
# magiclmnano --do_train --do_eval --stage sft --finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target wqkv,w1,w2,w3 --loraplus_lr_ratio 16 \
# 	--name=1211_magiclm_nano_cusl_loraplus16_ep3_lr1e4_bs4 --dataset COIG_CQIA_train,union_conversations_v4_norm,liantong_conversations_v1,swindle_data_v1 \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=3 --lr=1e-4 --save_strategy=epoch --warmup_ratio 0.03 --save_total_limit=3 --eval_dataset \
# 	union_conversations_v5_dev --eval_strategy=steps --eval_steps=300 --include node12

#DPO训练
# wait
# ./magiclmnano_dpo.sh --do_train --stage dpo --finetuning_type lora --adapter_name_or_path=/home/jovyan/zhubin/saved_checkpoint/1211_magiclm_nano_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset conversation_dpo_v3 --epochs=3 --name=1211_magiclm_nano_dpo_v3_ep3_lr1e6_bs4 --save_total_limit=10 --warmup_ratio=0.03 --include=node12:0,1,2,3,4,5,6,7
# wait
# ./magiclmnano_dpo.sh --do_train --stage dpo --finetuning_type lora --adapter_name_or_path=/home/jovyan/zhubin/saved_checkpoint/1202_magiclm_nano_neft_cudsls_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset conversation_dpo_v3 --epochs=3 --name=1211_magiclm_nano_dpo_v3_ep3_lr5e6_bs4 --lr=5e-6 --save_total_limit=10 --warmup_ratio=0.03 --include=node12:0,1,2,3,4,5,6,7

# ./magiclmnano_dpo.sh --do_train --stage dpo --finetuning_type lora --adapter_name_or_path=/home/jovyan/zhubin/saved_checkpoint/1219_magiclm_nano_neft_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset conversation_dpo_v3 --epochs=3 --name=1219_magiclm_nano_dpo_v3_ep3_lr1e6_bs4 --save_total_limit=10 --warmup_ratio=0.03 --include=node12:0,1,2,3,4,5,6,7
# wait
# ./magiclmnano_dpo.sh --do_train --stage dpo --finetuning_type lora --adapter_name_or_path=/home/jovyan/zhubin/saved_checkpoint/1219_magiclm_nano_neft_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset conversation_dpo_v3 --epochs=3 --name=1219_magiclm_nano_dpo_v3_ep3_lr5e6_bs4 --lr=5e-6 --save_total_limit=10 --warmup_ratio=0.03 --include=node12:0,1,2,3,4,5,6,7

# #simpo训练

# ./magiclmnano_dpo.sh --do_train --stage dpo --finetuning_type lora --adapter_name_or_path=/home/jovyan/zhubin/saved_checkpoint/1219_magiclm_nano_neft_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset conversation_dpo_v3 --epochs=3 --name=1219_magiclm_nano_simpo_v3_ep3_lr1e6_bs4_beat2_gamma_1.6 --pref_loss=simpo --pref_beta=2 --simpo_gamma=1.6 --lr=1e-6 --save_total_limit=10 --warmup_ratio=0.03 --include=node12:0,1,2,3,4,5,6,7

# wait
# ./magiclmnano_dpo.sh --do_train --stage dpo --finetuning_type lora --adapter_name_or_path=/home/jovyan/zhubin/saved_checkpoint/1219_magiclm_nano_neft_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset conversation_dpo_v3 --epochs=3 --name=1219_magiclm_nano_simpo_v3_ep3_lr5e7_bs4_beat2_gamma_1.6 --pref_loss=simpo --pref_beta=2 --simpo_gamma=1.6 --lr=5e-7 --save_total_limit=10 --warmup_ratio=0.03 --include=node12:0,1,2,3,4,5,6,7

# ./magiclmnano_dpo.sh --do_train --stage dpo --finetuning_type lora --adapter_name_or_path=/home/jovyan/zhubin/saved_checkpoint/1211_magiclm_nano_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset conversation_dpo_v3 --epochs=3 --name=1211_magiclm_nano_simpo_v3_ep3_lr1e6_bs4_beat2_gamma_1.6 --pref_loss=simpo --pref_beta=2 --simpo_gamma=1.6 --lr=1e-6 --save_total_limit=10 --warmup_ratio=0.03 --include=node12:0,1,2,3,4,5,6,7

# wait
# ./magiclmnano_dpo.sh --do_train --stage dpo --finetuning_type lora --adapter_name_or_path=/home/jovyan/zhubin/saved_checkpoint/1211_magiclm_nano_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset conversation_dpo_v3 --epochs=3 --name=1211_magiclm_nano_simpo_v3_ep3_lr5e7_bs4_beat2_gamma_1.6 --pref_loss=simpo --pref_beta=2 --simpo_gamma=1.6 --lr=5e-7 --save_total_limit=10 --warmup_ratio=0.03 --include=node12:0,1,2,3,4,5,6,7

# #lora_rank=16模型 SFT
# magiclmnano --do_train --do_eval --stage sft --finetuning_type lora --lora_rank 16 --lora_alpha 1 --lora_target wqkv,w1,w2,w3 --loraplus_lr_ratio 16 \
# 	--name=1211_magiclm_nano_cusl_loraplus16_rk16_ep3_lr1e4_bs4 --dataset COIG_CQIA_train,union_conversations_v4_norm,liantong_conversations_v1,swindle_data_v1 \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=3 --lr=1e-4 --save_strategy=epoch --warmup_ratio 0.03 --save_total_limit=3 --eval_dataset \
# 	union_conversations_v5_dev --eval_strategy=steps --eval_steps=300 --include node12

# #lora_rank=16模型 DPO
# ./magiclmnano_dpo.sh --do_train --stage dpo --finetuning_type lora --adapter_name_or_path=/home/jovyan/zhubin/saved_checkpoint/1211_magiclm_nano_cusl_loraplus16_rk16_ep3_lr1e4_bs4 \
# 	--dataset conversation_dpo_v3 --epochs=3 --name=1211_magiclm_nano_dpo_v3_rk16_ep3_lr5e6_bs4 --lr=5e-6 --save_total_limit=10 --warmup_ratio=0.03 --include=node12:0,1,2,3,4,5,6,7

# 第四版本DPO训练
# ./magiclmnano_dpo.sh --do_train --stage dpo --finetuning_type lora --adapter_name_or_path=/home/jovyan/zhubin/saved_checkpoint/1211_magiclm_nano_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset conversation_dpo_v3,real_conversation_dpo_v2,natrual_conv9k_v2 --epochs=3 --name=1214_magiclm_nano_dpo_crn_v4_ep3_lr1e6_bs4 \
# 	--save_total_limit=10 --warmup_ratio=0.03 --include=node12:0,1,2,3,4,5,6,7

# wait
# ./magiclmnano_dpo.sh --do_train --stage dpo --finetuning_type lora --adapter_name_or_path=/home/jovyan/zhubin/saved_checkpoint/1211_magiclm_nano_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset conversation_dpo_v3,real_conversation_dpo_v2,natrual_conv9k_v2 --epochs=3 --name=1214_magiclm_nano_dpo_crn_v4_ep3_lr5e6_bs4 \
# 	--lr=5e-6 --save_total_limit=10 --warmup_ratio=0.03 --include=node12:0,1,2,3,4,5,6,7

# wait
# # simpo训练
# ./magiclmnano_dpo.sh --do_train --stage dpo --finetuning_type lora --adapter_name_or_path=/home/jovyan/zhubin/saved_checkpoint/1211_magiclm_nano_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset conversation_dpo_v3,real_conversation_dpo_v2,natrual_conv9k_v2 --epochs=3 --name=1214_magiclm_nano_simpo_crn_v4_ep3_lr1e6_bs4_beat2_gamma_1.6 \
# 	--pref_loss=simpo --pref_beta=2 --simpo_gamma=1.6 --lr=1e-6 --save_total_limit=10 --warmup_ratio=0.03 --include=node12:0,1,2,3,4,5,6,7

# wait
# ./magiclmnano_dpo.sh --do_train --stage dpo --finetuning_type lora --adapter_name_or_path=/home/jovyan/zhubin/saved_checkpoint/1211_magiclm_nano_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset conversation_dpo_v3,real_conversation_dpo_v2,natrual_conv9k_v2 --epochs=3 --name=1214_magiclm_nano_simpo_crn_v4_ep3_lr5e7_bs4_beat2_gamma_1.6 \
# 	--pref_loss=simpo --pref_beta=2 --simpo_gamma=1.6 --lr=5e-7 --save_total_limit=10 --warmup_ratio=0.03 --include=node12:0,1,2,3,4,5,6,7

#添加Noise数据的偏好数据
# ./magiclmnano_dpo.sh --do_train --stage dpo --finetuning_type lora --adapter_name_or_path=/home/jovyan/zhubin/saved_checkpoint/1211_magiclm_nano_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset conversation_dpo_v3,real_conversation_noise_dpo_v2,natrual_conv9k_noise_v2 --epochs=3 --name=1216_magiclm_nano_dpo_crn_noise_v4_ep3_lr1e6_bs4 \
# 	--save_total_limit=10 --warmup_ratio=0.03 --include=node11:0,1,2,3,4,5,6,7

# wait
# ./magiclmnano_dpo.sh --do_train --stage dpo --finetuning_type lora --adapter_name_or_path=/home/jovyan/zhubin/saved_checkpoint/1211_magiclm_nano_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset conversation_dpo_v3,real_conversation_noise_dpo_v2,natrual_conv9k_noise_v2 --epochs=3 --name=1216_magiclm_nano_dpo_crn_noise_v4_ep3_lr5e6_bs4 \
# 	--lr=5e-6 --save_total_limit=10 --warmup_ratio=0.03 --include=node11:0,1,2,3,4,5,6,7

# wait
# # simpo训练
# ./magiclmnano_dpo.sh --do_train --stage dpo --finetuning_type lora --adapter_name_or_path=/home/jovyan/zhubin/saved_checkpoint/1211_magiclm_nano_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset conversation_dpo_v3,real_conversation_noise_dpo_v2,natrual_conv9k_noise_v2 --epochs=3 --name=1216_magiclm_nano_simpo_crn_noise_v4_ep3_lr1e6_bs4_beat2_gamma_1.6 \
# 	--pref_loss=simpo --pref_beta=2 --simpo_gamma=1.6 --lr=1e-6 --save_total_limit=10 --warmup_ratio=0.03 --include=node11:0,1,2,3,4,5,6,7

# wait
# ./magiclmnano_dpo.sh --do_train --stage dpo --finetuning_type lora --adapter_name_or_path=/home/jovyan/zhubin/saved_checkpoint/1211_magiclm_nano_cusl_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset conversation_dpo_v3,real_conversation_noise_dpo_v2,natrual_conv9k_noise_v2 --epochs=3 --name=1216_magiclm_nano_simpo_crn_noise_v4_ep3_lr5e7_bs4_beat2_gamma_1.6 \
# 	--pref_loss=simpo --pref_beta=2 --simpo_gamma=1.6 --lr=5e-7 --save_total_limit=10 --warmup_ratio=0.03 --include=node11:0,1,2,3,4,5,6,7

# ./magiclmnano_rm.sh --do_train --stage rm --finetuning_type full \
# 	--dataset conversation_dpo_v3,real_conversation_noise_dpo_v2,natrual_conv9k_noise_v2 --epochs=3 --name=TEST \
# 	--lr=5e-7 --save_total_limit=10 --warmup_ratio=0.03 --include=node12

# # ppo
# ./magiclmnano_ppo.sh --do_train --stage ppo --finetuning_type lora \
# 	--adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/1219_magiclm_nano_neft_cusl_loraplus16_ep3_lr1e4_bs4/checkpoint-665/ \
# 	--dataset call_summarization_ppo_train --epochs=1 --name=1221_magiclm_nano_s665_ppo_ep1_lr5e6_bs2 \
# 	--lr=5e-6 --save_total_limit=100 --warmup_ratio=0.1 --include=node12:1,2,3,4,5
# wait
# ./magiclmnano_ppo.sh --do_train --stage ppo --finetuning_type lora \
# 	--adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/1219_magiclm_nano_neft_cusl_loraplus16_ep3_lr1e4_bs4/checkpoint-1995/ \
# 	--dataset call_summarization_ppo_train --epochs=1 --name=1221_magiclm_nano_s1995_ppo_ep1_lr5e6_bs2 \
# 	--lr=5e-6 --save_total_limit=100 --warmup_ratio=0.1 --include=node12:1,2,3,4,5

# wait
# ./magiclmnano_ppo.sh --do_train --stage ppo --finetuning_type lora \
# 	--adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/1219_magiclm_nano_neft_cusl_loraplus16_ep3_lr1e4_bs4/checkpoint-665/ \
# 	--dataset call_summarization_ppo_train --epochs=1 --name=1221_magiclm_nano_s665_ppo_ep1_lr3e6_bs2 \
# 	--lr=3e-6 --save_total_limit=100 --warmup_ratio=0.1 --include=node12:1,2,3,4,5

# wait
# ./magiclmnano_ppo.sh --do_train --stage ppo --finetuning_type lora \
# 	--adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/1219_magiclm_nano_neft_cusl_loraplus16_ep3_lr1e4_bs4/checkpoint-665/ \
# 	--dataset call_summarization_ppo_train --epochs=1 --name=1221_magiclm_nano_s665_ppo_ep1_lr2e6_bs2 \
# 	--lr=2e-6 --batch_size=2 --save_total_limit=100 --warmup_ratio=0.1 --include=node12:1,2,3,4,5

# wait
# ./magiclmnano_ppo.sh --do_train --stage ppo --finetuning_type lora \
# 	--adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/1219_magiclm_nano_neft_cusl_loraplus16_ep3_lr1e4_bs4/checkpoint-665/ \
# 	--dataset call_summarization_ppo_train --epochs=1 --name=1221_magiclm_nano_s665_ppo_ep1_lr5e6_bs4 \
# 	--lr=5e-6 --batch_size=4 --save_total_limit=100 --warmup_ratio=0.1 --include=node12:1,2,3,4,5

# wait
# ./magiclmnano_ppo.sh --do_train --stage ppo --finetuning_type lora \
# 	--adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/1219_magiclm_nano_neft_cusl_loraplus16_ep3_lr1e4_bs4/checkpoint-1995/ \
# 	--dataset call_summarization_ppo_train --epochs=1 --name=1221_magiclm_nano_s1995_ppo_ep1_lr5e6_bs4 \
# 	--lr=5e-6 --batch_size=4 --save_total_limit=100 --warmup_ratio=0.1 --include=node12:1,2,3,4,5

# 先验证学习率
# 在验证checkpoint
# 在验证batchsize

# ./llmtrain.sh --do_train --stage sft --name=1223_Qwen2.5_14B_neft_cfvcaulsd_ep2_lr2e5_bs4 --model_name_or_path /home/jovyan/zhubin/DATA/models/Qwen/Qwen2.5-14B --template qwen \
# 	--dataset COIG_CQIA_train,firefly_summary_part,vcsum_headlines,csds_dialogue,alimeeting,union_conversations_v4_norm,liantong_conversations_v1,samsum_chinese,dialogsum_chinese \
# 	--finetuning_type full --batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 2 --lr=2e-5 --save_strategy=steps --save_steps=500 --save_total_limit=100 \
# 	--eval_dataset=union_conversations_v5_dev --neftune_noise_alpha=5 --eval_strategy=steps --eval_steps=500 --warmup_ratio=0.01 --include=node14

# ./llmtrain.sh --do_train --stage sft --name=0103_Qwen2.5_14B_neft_acfvcaulsd_markdown_ep2_lr5e5_bs2 --model_name_or_path /opt/nas/p/zhubin/DATA/models/Qwen/Qwen2.5-14B --template qwen \
# 	--dataset alpace_gpt4_zh_retain,COIG_PC_core_summary_part,firefly_summary_part,vcsum_headlines,csds_dialogue,union_conversations_v4_norm_markdown,liantong_conversations_v1_markdown,samsum_chinese_markdown,dialogsum_chinese_markdown \
# 	--finetuning_type full --batch_size 2 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 2 --lr=5e-5 --save_strategy=steps --save_steps=500 --save_total_limit=100 \
# 	--eval_dataset=union_conversations_v5_dev_markdown --neftune_noise_alpha=5 --eval_strategy=steps --eval_steps=500 --warmup_ratio=0.02

# ./magiclmnano_ppo.sh --do_train --stage ppo --finetuning_type lora \
# 	--adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/1219_magiclm_nano_neft_cusl_loraplus16_ep3_lr1e4_bs4/ \
# 	金 --dataset call_summarization_ppo_train --epochs=1 --name=1226_magiclm_nano_ep3_p2pp_ppo_ep1_lr3e6_bs2 \
# 	--ppo_target=4 --ppo_score_norm=true --ppo_whiten_rewards=true --ppo_init_kl_coef=0.2 --ppo_cliprange=0.1 --ppo_cliprange_value=0.1 --ppo_adap_kl_ctrl=false \
# 	--lr=5e-6 --batch_size=2 --gradient_accumulation_steps=4 --save_total_limit=100 --warmup_ratio=0.1 --include=node12:1,2,3,4,5

# magiclmnano --do_train --do_eval --stage sft --finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target wqkv,wo,w1,w2,w3 --loraplus_lr_ratio 16 \
# 	--name=1230_magiclm_nano_neft_cusl_loraplus16_ep3_lr1e4_bs4 --dataset COIG_CQIA_train,firefly_summary_part,csds_dialogue,union_conversations_v4_norm_markdown,liantong_conversations_v1_markdown --neftune_noise_alpha 5 \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=3 --lr=1e-4 --save_strategy=epoch --warmup_ratio 0.03 --save_total_limit=3 --eval_dataset \
# 	union_conversations_v5_dev_markdown --eval_strategy=steps --eval_steps=300 --include nlp-nlp-sum-0

# wait
# magiclmnano --do_train --do_eval --stage sft --finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target wqkv,w1,w2,w3 \
# 	--name=TEST --dataset COIG_CQIA_train,union_conversations_v4_norm,dialogsum_chinese,samsum_chinese,liantong_conversations_v1,swindle_data_v1 --neftune_noise_alpha 5 \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=1000 --lr=1e-4 --batch_size=4 --save_strategy=epoch --warmup_ratio 0.03 --save_total_limit=10 --eval_dataset \
# 	union_conversations_v4_dev --eval_strategy=steps --eval_steps=300 --include nlp-nlp-sum-1

# magiclmnano --do_train --do_eval --stage sft --finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target wqkv,wo,w1,w2,w3 --loraplus_lr_ratio 16 \
# 	--name=1231_magiclm_nano_neft_cafvcaulsd_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset COIG_CQIA_train,alpace_gpt4_zh_retain,firefly_summary_part,vcsum_headlines,csds_dialogue,union_conversations_v4_norm_markdown,liantong_conversations_v1_markdown,samsum_chinese_markdown,dialogsum_chinese_markdown \
# 	--neftune_noise_alpha 5 \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=2 --lr=1e-4 --save_strategy=epoch --warmup_ratio 0.02 --save_total_limit=3 --eval_dataset \
# 	union_conversations_v5_dev_markdown --eval_strategy=steps --eval_steps=300

# ./llmtrain.sh --do_train --stage sft --name=0111_Qwen2.5_14B_neft_acfvculddu_markdown_ep3_lr4e5_bs1 --model_name_or_path /opt/nas/p/zhubin/DATA/models/Qwen/Qwen2.5-14B --template qwen \
# 	--dataset alpace_gpt4_zh_retain,COIG_PC_core_summary_part,firefly_summary_part,vcsum_headlines,csds_dialogue,union_conversations_v4_norm_markdown,liantong_conversations_v1_markdown,dialogsum_chinese_markdown,diting_v1_markdown,union_conversations_v4_ll_markdown \
# 	--finetuning_type full --batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 3 --lr=4e-5 --save_strategy=epoch --save_total_limit=100 \
# 	--eval_dataset=union_conversations_v5_dev_markdown --neftune_noise_alpha=5 --eval_strategy=steps --eval_steps=500 --warmup_ratio=0.03

# wait
# ./llmtrain.sh --do_train --stage sft --name=0111_Qwen2.5_14B_neft_acfvculddu_markdown_ep3_lr2e5_bs1 --model_name_or_path /opt/nas/p/zhubin/DATA/models/Qwen/Qwen2.5-14B --template qwen \
# 	--dataset alpace_gpt4_zh_retain,COIG_PC_core_summary_part,firefly_summary_part,vcsum_headlines,csds_dialogue,union_conversations_v4_norm_markdown,liantong_conversations_v1_markdown,dialogsum_chinese_markdown,diting_v1_markdown,union_conversations_v4_ll_markdown \
# 	--finetuning_type full --batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 3 --lr=2e-5 --save_strategy=epoch --save_total_limit=100 \
# 	--eval_dataset=union_conversations_v5_dev_markdown --neftune_noise_alpha=5 --eval_strategy=steps --eval_steps=500 --warmup_ratio=0.03

# ./magiclmnano_sft.sh --do_train --do_eval --stage sft --finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target wqkv,w1,w2,w3 --loraplus_lr_ratio 16 \
# 	--name=0123_magiclmnano_auldu_loraplus16_ep3_lr1e4_bs4 \
# 	--dataset alpace_gpt4_zh_retain,union_conversations_v5_norm_markdown,liantong_conversations_v1_markdown,diting_v1_markdown,union_conversations_v4_ll_markdown \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=3 --lr=1e-4 --save_strategy=epoch --warmup_ratio 0.03 --save_total_limit=3 \
# 	--eval_dataset union_conversations_v5_dev --eval_strategy=steps --eval_steps=300
# wait

# ./llmtrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen --lora_rank 32 --lora_target all --loraplus_lr_ratio 16 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-1.5B \
# 	--name 0123_qwen2.5_1.5B_auldu_loraplus16_ep3_lr2e4_bs4 \
# 	--dataset union_conversations_v5_norm_markdown,liantong_conversations_v1_markdown,diting_v1_markdown,union_conversations_v4_ll_markdown \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=3 --lr=1e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset union_conversations_v5_dev --eval_strategy=steps --eval_steps=500

# ./llmtrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen --lora_rank 32 --lora_target all --loraplus_lr_ratio 16 \
# 	--model_name_or_path /opt/nas/p/models/qwen_models/qwen2.5-1.5b \
# 	--name 0123_qwen2.5_1.5b_auldu_rvt_loraplus16_ep3_lr2e4_bs4 \
# 	--dataset union_conversations_v5_norm_markdown,liantong_conversations_v1_markdown,diting_v1_markdown,union_conversations_v4_ll_markdown \
# 	--gradient_accumulation_steps=4 --resize_vocab true --cutoff_len=4096 --epochs=3 --lr=1e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset union_conversations_v5_dev --eval_strategy=steps --eval_steps=500

###instruct 模型lora训练
# ./llmtrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen --lora_rank 32 --lora_target all --loraplus_lr_ratio 16 \
# 	--model_name_or_path /opt/nas/n/zhubin/DATA/models/Qwen/Qwen2.5-1.5B-Instruct/ \
# 	--name 0123_qwen2.5_1.5b_instr_auldu__loraplus16_ep3_lr2e4_bs4 \
# 	--dataset union_conversations_v5_norm_markdown,liantong_conversations_v1_markdown,diting_v1_markdown,union_conversations_v4_ll_markdown \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=3 --lr=1e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset union_conversations_v5_dev --eval_strategy=steps --eval_steps=500

# wait
# ./llmtrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen --lora_rank 32 --lora_target all --loraplus_lr_ratio 16 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-3B/ \
# 	--name 0123_qwen2.5_3B_auldu_loraplus16_ep3_lr2e4_bs4 \
# 	--dataset union_conversations_v5_norm_markdown,liantong_conversations_v1_markdown,diting_v1_markdown,union_conversations_v4_ll_markdown \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=3 --lr=1e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset union_conversations_v5_dev_markdown --eval_strategy=steps --eval_steps=500

###3B instruct 微调
# ./llmtrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template qwen --lora_rank 32 --lora_target all --loraplus_lr_ratio 16 \
# 	--model_name_or_path \
# 	--name 0123_qwen2.5_3B_auldu_loraplus16_ep3_lr2e4_bs4 \
# 	--dataset union_conversations_v5_norm_markdown,liantong_conversations_v1_markdown,diting_v1_markdown,union_conversations_v4_ll_markdown \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=3 --lr=1e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset union_conversations_v5_dev_markdown --eval_strategy=steps --eval_steps=500

# wait
# ./llmtrain.sh --do_train --do_eval --stage sft --finetuning_type lora --template glm-edge --lora_rank 32 --lora_target all --loraplus_lr_ratio 16 \
# 	--name=0123_glm4edge_auldu_loraplus16_ep3_lr5e4_bs4 \
# 	--model_name_or_path /opt/nas/n/zhubin/DATA/models/THUDM/glm-edge-4b-chat/ \
# 	--dataset union_conversations_v5_norm_markdown,liantong_conversations_v1_markdown,diting_v1_markdown,union_conversations_v4_ll_markdown \
# 	--gradient_accumulation_steps=4 --cutoff_len=4096 --epochs=3 --lr=5e-4 --save_strategy=epoch --warmup_ratio 0.1 --save_total_limit=10 \
# 	--eval_dataset union_conversations_v5_dev_markdown --eval_strategy=steps --eval_steps=300

# ./llmtrain.sh --do_train --do_eval --stage sft --finetuning_type full --name=0222_Qwen2.5-14B-Instruct_neft_accdb_markdown_ep3_lr1e5_bs1 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset alpace_gpt4_zh_retain,COIG_PC_core_summary_part,callsum_v6_train_markdown,diting_v2_markdown,beta_noise_v1_markdown \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 3 --lr 1e-5 --save_strategy epoch --save_total_limit 100 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v6_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03

# # wait
# ./llmtrain.sh --do_train --do_eval --stage sft --finetuning_type full --name=0222_Qwen2.5-14B-Instruct_neft_accdb_markdown_ep3_lr2e5_bs1 \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset alpace_gpt4_zh_retain,COIG_PC_core_summary_part,callsum_v6_train_markdown,diting_v2_markdown,beta_noise_v1_markdown \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 3 --lr 2e-5 --save_strategy epoch --save_total_limit 100 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v6_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03

# ./llmtrain.sh --do_train --do_eval --stage sft --finetuning_type full --name=TEST \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset alpace_gpt4_zh_retain \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 3 --lr 2e-5 --save_strategy epoch --save_total_limit 100 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v6_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/config/hostfile \
# 	--stage sft --finetuning_type full --name="${DATE}_Qwen2.5-14B-Instruct_neft5_accdb_markdown_wd_ep3_lr7e6_bs1" \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset alpace_gpt4_zh_retain,COIG_PC_core_summary_part,callsum_v8.1_train_markdown,diting_v8.2_markdown,beta_noise_v8.1_markdown \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 3 --lr 7e-6 --weight_decay 0.1 \
# 	--save_strategy epoch --save_total_limit 100 --seed 42 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v8_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/config/hostfile \
# 	--stage sft --finetuning_type full --name="${DATE}_Qwen2.5-14B-Instruct_neft15_iccdb_markdown_wd_ep3_lr7e6_bs1" \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset Inf_gens,COIG_PC_core_summary_part,callsum_v8.1_train_markdown,diting_v8.2_markdown,beta_noise_v8.1_markdown \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 3 --lr 7e-6 --weight_decay 0.1 \
# 	--save_strategy epoch --save_total_limit 100 --seed 42 \
# 	--neftune_noise_alpha 15 --eval_dataset callsum_v8_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/config/hostfile \
# 	--stage sft --finetuning_type full --name="${DATE}_Qwen2.5-14B-Instruct_neft5_acdb_markdown_wd_ep3_lr7e6_bs1" \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset alpace_gpt4_zh_retain,callsum_v8.1_train_markdown,diting_v8.2_markdown,beta_noise_v8.1_markdown \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 3 --lr 7e-6 --weight_decay 0.1 \
# 	--save_strategy epoch --save_total_limit 100 --seed 42 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v8_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/config/hostfile \
# 	--stage sft --finetuning_type full --name="${DATE}_Qwen2.5-14B-Instruct_neft5_accdb_markdown_wd_ep2_lr7e6_bs1" \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset alpace_gpt4_zh_retain,COIG_PC_core_summary_part,callsum_v8.1_train_markdown,diting_v8.2_markdown,beta_noise_v8.1_markdown \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 3 --lr 7e-6 --weight_decay 0.1 \
# 	--save_strategy epoch --save_total_limit 100 --seed 42 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v8_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/config/hostfile \
# 	--stage sft --finetuning_type full --name="${DATE}_Qwen2.5-14B-Instruct_neft5_afcdbzd_v8_v7_markdown_wd_ep2_lr7e6_bs1" \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset alpace_gpt4_zh_retain,firefly_summary_part,callsum_v8.1_train_markdown,diting_v8.2_markdown,beta_noise_v8.1_markdown,zdjt_v8_markdown,diting_fraud_v8_markdown,callsum_v7.1_train_markdown,diting_v7.2_markdown \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 2 --lr 7e-6 --weight_decay 0.1 \
# 	--save_strategy epoch --save_total_limit 100 --seed 42 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v8_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/config/hostfile \
# 	--stage sft --finetuning_type full --name="${DATE}_Qwen2.5-14B-Instruct_neft5_afcdbzd_v8_v7_markdown_wd_ep2_lr1e5_bs1" \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset alpace_gpt4_zh_retain,firefly_summary_part,callsum_v8.1_train_markdown,diting_v8.2_markdown,beta_noise_v8.1_markdown,zdjt_v8_markdown,diting_fraud_v8_markdown,callsum_v7.1_train_markdown,diting_v7.2_markdown \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 2 --lr 1e-5 --weight_decay 0.1 \
# 	--save_strategy epoch --save_total_limit 100 --seed 42 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v8_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03

#Qwen3 1.7B 训练

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/config/hostfile \
# 	--stage sft --finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target all --loraplus_lr_ratio 16 \
# 	--name="${DATE}_Qwen3-1.7B-Instruct_neft5_accdb_markdown_lora_ep2_lr2e4_bs4" \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen3-1.7B/ --template qwen3 \
# 	--dataset alpace_gpt4_zh_retain,COIG_PC_core_summary_part,callsum_v6_train_norm_markdown,diting_v2_markdown,beta_noise_v1_markdown \
# 	--batch_size 4 --gradient_accumulation_steps 4 --cutoff_len 4096 --epochs 3 --lr 2e-4 \
# 	--save_strategy epoch --save_total_limit 100 --seed 42 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v6_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03

# #Qwen3 4B 训练

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/config/hostfile \
# 	--stage sft --finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target all --loraplus_lr_ratio 16 \
# 	--name="${DATE}_Qwen3-4B-Instruct_neft5_accdb_markdown_lora_ep2_lr2e4_bs4" \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen3-4B/ --template qwen3 \
# 	--dataset alpace_gpt4_zh_retain,COIG_PC_core_summary_part,callsum_v6_train_norm_markdown,diting_v2_markdown,beta_noise_v1_markdown \
# 	--batch_size 4 --gradient_accumulation_steps 4 --cutoff_len 4096 --epochs 3 --lr 2e-4 \
# 	--save_strategy epoch --save_total_limit 100 --seed 42 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v6_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/config/hostfile \
# 	--stage sft --finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target all --loraplus_lr_ratio 16 \
# 	--name="${DATE}_Qwen3-1.7B-Instruct_neft5_cdb_markdown_lora32_alpha1_ep2_lr2e4_bs4" \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen3-1.7B/ --template qwen3 \
# 	--dataset callsum_v6_train_norm_markdown,diting_v2_markdown,beta_noise_v1_markdown \
# 	--batch_size 4 --gradient_accumulation_steps 4 --cutoff_len 4096 --epochs 2 --lr 2e-4 \
# 	--save_strategy epoch --save_total_limit 100 --seed 42 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v6_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03

# #Qwen3 4B 训练

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/config/hostfile \
# 	--stage sft --finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target all --loraplus_lr_ratio 16 \
# 	--name="${DATE}_Qwen3-4B-Instruct_neft5_cdb_markdown_lora32_alpha1_ep2_lr2e4_bs4" \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen3-4B/ --template qwen3 \
# 	--dataset callsum_v6_train_norm_markdown,diting_v2_markdown,beta_noise_v1_markdown \
# 	--batch_size 4 --gradient_accumulation_steps 4 --cutoff_len 4096 --epochs 2 --lr 2e-4 \
# 	--save_strategy epoch --save_total_limit 100 --seed 42 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v6_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/config/hostfile \
# 	--stage sft --finetuning_type full --name="${DATE}_Qwen2.5-14B-Instruct_neft5_afcdbzd_v9_v7_markdown_wd_ep2_lr7e6_bs1" \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset alpace_gpt4_zh_retain,firefly_summary_part,callsum_v9.1_train_markdown,diting_v9.2_markdown,beta_noise_v9.1_markdown,zdjt_v9_markdown,diting_fraud_v9_markdown,callsum_v7.1_train_markdown,diting_v7.2_markdown \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 2 --lr 7e-6 --weight_decay 0.1 \
# 	--save_strategy epoch --save_total_limit 100 --seed 42 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v9_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03 \
# 	--enable_liger_kernel true --flash_attn fa2 \

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/config/hostfile \
# 	--stage sft --finetuning_type full --name="${DATE}_Qwen2.5-14B-Instruct_neft5_afcdbzd_v9_v7_markdown_wd_ep2_lr1e5_bs1" \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset alpace_gpt4_zh_retain,firefly_summary_part,callsum_v9.1_train_markdown,diting_v9.2_markdown,beta_noise_v9.1_markdown,zdjt_v9_markdown,diting_fraud_v9_markdown,callsum_v7.1_train_markdown,diting_v7.2_markdown \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 2 --lr 1e-5 --weight_decay 0.1 \
# 	--save_strategy epoch --save_total_limit 100 --seed 42 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v9_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03 \
# 	--enable_liger_kernel true --flash_attn fa2 \

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/config/hostfile \
# 	--stage sft --finetuning_type full --name="${DATE}_Qwen2.5-14B-Instruct_neft5_afcdbzd_v9_v8_markdown_wd_ep2_lr7e6_bs1" \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset alpace_gpt4_zh_retain,firefly_summary_part,callsum_v9.1_train_markdown,diting_v9.2_markdown,beta_noise_v9.1_markdown,zdjt_v9_markdown,diting_fraud_v9_markdown,callsum_v8.1_train_markdown,diting_v8.2_markdown \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 2 --lr 7e-6 --weight_decay 0.1 \
# 	--save_strategy epoch --save_total_limit 100 --seed 42 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v9_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03 \
# 	--enable_liger_kernel true --flash_attn fa2

# ./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/config/hostfile \
# 	--stage sft --finetuning_type full --name="${DATE}_Qwen2.5-14B-Instruct_neft5_afcdbzd_v9_v8_markdown_wd_ep2_lr1e5_bs1" \
# 	--model_name_or_path /opt/nas/p/models/Qwen_models/Qwen2.5-14B-Instruct --template qwen \
# 	--dataset alpace_gpt4_zh_retain,firefly_summary_part,callsum_v9.1_train_markdown,diting_v9.2_markdown,beta_noise_v9.1_markdown,zdjt_v9_markdown,diting_fraud_v9_markdown,callsum_v8.1_train_markdown,diting_v8.2_markdown \
# 	--batch_size 1 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 2 --lr 1e-5 --weight_decay 0.1 \
# 	--save_strategy epoch --save_total_limit 100 --seed 42 \
# 	--neftune_noise_alpha 5 --eval_dataset callsum_v9_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03 \
# 	--enable_liger_kernel true --flash_attn fa2

./llmtrain.sh --do_train --do_eval --hostfile /opt/nas/p/zhubin/code/Llmtrain/config/hostfile \
	--stage sft --finetuning_type lora --lora_rank 32 --lora_alpha 1 --lora_target all --loraplus_lr_ratio 16 \
	--name="${DATE}_Qwen3-4B-pruning_afcdbzd_v9_v8_markdown_wd_ep2_lr2e4_bs4" \
	--model_name_or_path /opt/nas/n/xwgeng/Working/LLaMA-Factory/ckpts/depth_pruning_sft --template qwen3 \
	--dataset alpace_gpt4_zh_retain,firefly_summary_part,callsum_v9.1_train_markdown,diting_v9.2_markdown,beta_noise_v9.1_markdown,zdjt_v9_markdown,diting_fraud_v9_markdown,callsum_v8.1_train_markdown,diting_v8.2_markdown \
	--batch_size 4 --gradient_accumulation_steps 16 --cutoff_len 4096 --epochs 2 --lr 2e-4 --weight_decay 0.1 \
	--save_strategy epoch --save_total_limit 100 --seed 42 \
	--eval_dataset callsum_v9_test_markdown --eval_strategy steps --eval_steps 500 --warmup_ratio 0.03 \
	--enable_liger_kernel true --flash_attn fa2

/opt/nas/p/zhubin/run_GPU/run_full_gpu.sh
