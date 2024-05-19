set -x

root_imagenet='/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/'
MODEL=doit22_small
OUTPUT='output/170_doit16_tiny_basic'

srun -p gvembodied \
    --job-name=llmeval_170 \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=8 main_soft_ceph.py \
    --model $MODEL --epochs 300 --mixup 0.5 --cutmix 0.5 \
    --batch_size 128 --lr 4e-3 --update_freq 4 \
    --drop_path 0.1 --drop_mode standard \
    --mask_mode soft --mask_schedule constant --cutoff_soft 50 \
    --data_path $root_imagenet \
    --output_dir $OUTPUT