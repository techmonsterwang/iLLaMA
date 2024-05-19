set -x

root_imagenet='/mnt/petrelfs/share/images/'
MODEL=doit7_base
OUTPUT='output/95_doit7_base_basic'

srun -p Gveval2 \
    --job-name=llmeval_95 \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=spot \
    python -m torch.distributed.launch --nproc_per_node=8 main_ceph.py \
    --model $MODEL --epochs 300 --mixup 0.9 --cutmix 1.0 \
    --batch_size 128 --lr 4e-3 --update_freq 4 \
    --drop_path 0.4 --drop_mode standard \
    --data_path $root_imagenet \
    --output_dir $OUTPUT