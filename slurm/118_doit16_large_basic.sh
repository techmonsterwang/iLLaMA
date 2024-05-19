set -x

root_imagenet='/mnt/petrelfs/share/images/'
MODEL=doit16_large
OUTPUT='output/118_doit16_large_basic'

srun -p Gveval2 \
    --job-name=llmeval_118 \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=spot \
    python -m torch.distributed.launch --nproc_per_node=8 main_soft_ceph.py \
    --model $MODEL --epochs 300 --warmup_epochs 30 --mixup 0.95 --cutmix 1.0 --model_ema true --model_ema_eval true \
    --batch_size 64 --lr 1e-4 --weight_decay 0.3 --update_freq 8 \
    --drop_path 0.7 --drop_mode standard \
    --mask_mode soft --mask_schedule constant --cutoff_soft 50 \
    --data_path $root_imagenet \
    --output_dir $OUTPUT