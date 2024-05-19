set -x

root_imagenet='/mnt/petrelfs/wangjiahao/datasets/classificaton/imagenet/'
MODEL=illama_tiny
OUTPUT='output/debug'

srun -p gvembodied \
    --job-name=debug \
    --gres=gpu:4 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=spot \
    python -m torch.distributed.launch --nproc_per_node=4 main_soft.py \
    --model $MODEL --epochs 300 --warmup_epochs 5 --mixup 0.1 --cutmix 0.1 \
    --batch_size 128 --lr 4e-3 --update_freq 4 \
    --dropout 0 --drop_mode standard \
    --mask_mode soft --mask_schedule constant --cutoff_soft 50 \
    --data_path $root_imagenet \
    --output_dir $OUTPUT