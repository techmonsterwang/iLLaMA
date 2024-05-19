set -x

root_imagenet='/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/'
MODEL=doit16_tiny
OUTPUT='output/70_doit16_tiny_basic'

srun -p eval \
    --job-name=llmeval_70 \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=8 main_soft_ceph.py \
    --model $MODEL --epochs 300 \
    --batch_size 128 --lr 4e-3 --update_freq 4 \
    --drop_path 0.5 --drop_mode early --drop_schedule linear --cutoff_epoch 50 \
    --mask_mode soft --mask_schedule linear --cutoff_soft 25 \
    --data_path $root_imagenet \
    --output_dir $OUTPUT