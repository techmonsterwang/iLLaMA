set -x

root_dataset='/mnt/petrelfs/wangjiahao/datasets/classificaton/flowers/'
PRETRAINED='/mnt/petrelfs/wangjiahao/DoiT/pretrained/doit-tiny-in1k.pth'
MODEL=doit16_tiny
OUTPUT='output/t26_doit_tiny_flowers'

srun -p gvembodied \
    -x SH-IDC1-10-140-0-201 \
    --job-name=doit_t26_transfer \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=spot \
    python -m torch.distributed.launch --nproc_per_node=8 main_soft_transfer.py \
    --model $MODEL --warmup_epochs 100 --epochs 600 \
    --batch_size 64 --lr 2e-3 --update_freq 1 --use_amp true \
    --mask_mode soft --mask_schedule linear --cutoff_soft 25 \
    --finetune $PRETRAINED \
    --data_path $root_dataset \
    --data_set flowers \
    --output_dir $OUTPUT