set -x

root_imagenet='/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/'
MODEL=doit16_tiny
OUTPUT='output/72_doit16_tiny_basic'

srun -p eval \
    --job-name=llmeval_72 \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=8 main_soft_ceph.py \
    --model $MODEL --epochs 300 --mixup 0.2 --cutmix 0.2 \
    --batch_size 128 --lr 4e-3 --update_freq 4 \
    --dropout 0 --drop_mode standard \
    --mask_mode soft --mask_schedule linear --cutoff_soft 50 \
    --data_path $root_imagenet \
    --output_dir $OUTPUT