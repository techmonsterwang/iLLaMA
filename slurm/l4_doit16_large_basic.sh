set -x

root_imagenet21k='/mnt/petrelfs/wangjiahao/datasets/classification/imagenet21k/'
MODEL=doit16_large
OUTPUT='output/l4_doit16_large_basic'

srun -p Gveval2 \
    --job-name=llmeval_l4 \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=spot \
    python -m torch.distributed.launch --nproc_per_node=8 main_soft.py \
    --model $MODEL --warmup_epochs 5 --epochs 90 \
    --batch_size 32 --lr 1e-3 --weight_decay 0.01 --update_freq 16 \
    --drop_path 0.1 --drop_mode standard \
    --mask_mode soft --mask_schedule constant --cutoff_soft 45 \
    --data_set IMNET21K --nb_classes 10450 --data_path $root_imagenet21k \
    --output_dir $OUTPUT