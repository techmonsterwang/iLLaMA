set -x

root_imagenet='/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/'
MODEL=doit22_large
OUTPUT='output/hft13_doit22_large_basic'
FINETUNE='/mnt/petrelfs/wangjiahao/DoiT/output/l9_doit22_large_basic/checkpoint-89.pth'

srun -p eval \
    --job-name=llmeval_hft13 \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=8 main_soft_fthr_ceph.py \
    --model $MODEL --drop_path 0.3 --drop_mode standard --input_size 384 \
    --batch_size 16 --lr 1.1e-4 --update_freq 4 \
    --warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
    --cutmix 0 --mixup 0 \
    --mask_mode soft --mask_schedule constant --cutoff_soft 0 \
    --finetune $FINETUNE \
    --data_path $root_imagenet \
    --output_dir $OUTPUT
