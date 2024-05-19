root_imagenet='/your/path/to/imagenet/'
MODEL=illama_tiny
OUTPUT='output/path'

python -m torch.distributed.launch --nproc_per_node=8 main_soft.py \
    --model $MODEL --epochs 300 --warmup_epochs 5 --mixup 0.1 --cutmix 0.1 \
    --batch_size 128 --lr 4e-3 --update_freq 4 \
    --dropout 0 --drop_mode standard \
    --mask_mode soft --mask_schedule constant --cutoff_soft 50 \
    --data_path $root_imagenet \
    --output_dir $OUTPUT