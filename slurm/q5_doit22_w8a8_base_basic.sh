set -x

root_imagenet='/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/'
MODEL=doit22_base_w8a8
OUTPUT='output/q5_doit22_w8a8_base_basic'

srun -p eval \
    --job-name=llmeval_q5 \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=8 main_soft_ceph.py \
    --model $MODEL --epochs 300 --mixup 0.95 --cutmix 1.0 \
    --batch_size 128 --lr 3e-3 --weight_decay 0 --update_freq 4 \
    --drop_path 0.4 --drop_mode standard \
    --mask_mode soft --mask_schedule linear --cutoff_soft 25 \
    --data_path $root_imagenet \
    --output_dir $OUTPUT