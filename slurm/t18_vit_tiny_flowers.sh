set -x

root_dataset='/mnt/petrelfs/wangjiahao/datasets/classificaton/flowers/'
PRETRAINED='/mnt/petrelfs/wangjiahao/DoiT/pretrained/vit-tiny-in1k.pth'
MODEL=vit_tiny
OUTPUT='output/t18_vit_tiny_flowers'

srun -p gvembodied \
    --job-name=doit_t18_transfer \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=spot \
    python -m torch.distributed.launch --nproc_per_node=8 main_transfer.py \
    --model $MODEL --warmup_epochs 100 --epochs 600 \
    --batch_size 64 --lr 2e-3 --update_freq 1 --use_amp true \
    --finetune $PRETRAINED \
    --data_path $root_dataset \
    --data_set flowers \
    --output_dir $OUTPUT

