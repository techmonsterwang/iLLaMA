root_imagenet='/your/path/to/imagenet/'
MODEL=illama_base
OUTPUT='output/path'

python -m torch.distributed.launch --nproc_per_node=8 main_soft.py \
    --model $MODEL --epochs 300 --mixup 0.95 --cutmix 1.0 \
    --batch_size 128 --lr 4e-3 --update_freq 4 \
    --drop_path 0.4 --drop_mode standard \
    --mask_mode soft --mask_schedule linear --cutoff_soft 25 \
    --data_path $root_imagenet \
    --output_dir $OUTPUT