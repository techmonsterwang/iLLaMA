set -x

root_imagenet='/mnt/petrelfs/wangjiahao/datasets/classificaton/imagenet/'
MODEL=vit_tiny
OUTPUT='output/debug'

srun -p llmeval \
    --job-name=llmeval_1 \
    --gres=gpu:2 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=spot \
    python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --model $MODEL --epochs 300 \
    --batch_size 128 --lr 4e-3 --update_freq 4 \
    --dropout 0.1 --drop_mode early --drop_schedule linear --cutoff_epoch 50 \
    --data_path $root_imagenet \
    --output_dir $OUTPUT