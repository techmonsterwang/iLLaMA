set -x

root_imagenet='/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/'
MODEL=vit_tiny
OUTPUT='output/2_vit_tiny_improved'

srun -p llmeval \
    --job-name=llmeval_2 \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=8 main_ceph.py \
    --model $MODEL --epochs 600 --mixup 0.3 --cutmix 0.3 \
    --batch_size 128 --lr 4e-3 --update_freq 4 \
    --dropout 0.1 --drop_mode early --drop_schedule linear --cutoff_epoch 50 \
    --data_path $root_imagenet \
    --output_dir $OUTPUT