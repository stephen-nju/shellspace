# exec >/opt/nas/p/mmu/zb/code/shellspace/easyjobLog/llmtrain.log

vc -node list | awk '{print $0,"slots=8"}' >/opt/nas/p/mmu/zb/code/shellspace/cache/ms_hostfile
vc -proxy open

export DATE=$(date "+%Y%m%d-%H%M%S")
echo "training scripts date ${DATE}"
export HF_HOME=/opt/local/data/
export hostfile=/opt/nas/p/mmu/zb/code/shellspace/cache/ms_hostfile

# export schedule_train_ds=/opt/nas/n/mmu/zhubin/DATA/YOYO_memory/Schedule_train_datasets/20260226_v1/train_datasets_0226.json
# export schedule_dev_ds=/opt/nas/n/mmu/zhubin/DATA/YOYO_memory/Schedule_train_datasets/20260226_v1/dev_datasets_0226.json

# export schedule_train_ds=/opt/nas/n/mmu/zhubin/DATA/YOYO_memory/Schedule_train_datasets/20260303_v2/train_datasets_0303.json
# export schedule_dev_ds=/opt/nas/n/mmu/zhubin/DATA/YOYO_memory/Schedule_train_datasets/20260303_v2/dev_datasets_0303.json

export schedule_train_ds=/opt/nas/n/mmu/zhubin/DATA/YOYO_memory/Schedule_train_datasets/20260317_v3/train_datasets_0317.json
export schedule_dev_ds=/opt/nas/n/mmu/zhubin/DATA/YOYO_memory/Schedule_train_datasets/20260317_v3/dev_datasets_0317.json

export model_name=/opt/tools/resource/easy_model/model/Qwen3-VL-4B-Instruct/

./mstrain.sh --do_train --do_eval --stage sft --finetuning_type lora --lora_rank 16 --lora_alpha 32 --lora_target all-linear --loraplus_lr_ratio 16 \
	--hostfile ${hostfile} \
	--model_name_or_path $model_name \
	--name "${DATE}_Qwen3-VL-4B-Instruct_schedule_r16_ep2_lr1e4_bs4" \
	--dataset $schedule_train_ds \
	--gradient_accumulation_steps=4 --max_length=4096 --epochs=4 --batch_size 1 --lr=1e-4 --save_strategy=epoch --warmup_ratio 0.05 --save_total_limit=10 \
	--eval_dataset $schedule_dev_ds --eval_strategy steps --eval_steps=300
