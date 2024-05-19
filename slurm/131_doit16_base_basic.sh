set -x

root_imagenet='/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/'
MODEL=doit16_base
OUTPUT='output/131_doit16_base_basic'

srun -p eval \
    --job-name=llmeval_131 \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=8 main_soft_ceph.py \
    --model $MODEL --epochs 300 --warmup_epochs 75 --mixup 0.95 --cutmix 1.0 \
    --batch_size 128 --lr 4e-3 --update_freq 4 \
    --drop_path 0.4 --drop_mode standard \
    --mask_mode soft --mask_schedule constant --cutoff_soft 50 \
    --data_path $root_imagenet \
    --output_dir $OUTPUT