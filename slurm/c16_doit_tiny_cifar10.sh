set -x

root_dataset='/mnt/petrelfs/wangjiahao/datasets/classificaton/cifar10/'
PRETRAINED='/mnt/petrelfs/wangjiahao/DoiT/pretrained/doit-tiny-in1k.pth'
MODEL=doit16_tiny
OUTPUT='output/c16_doit_tiny_cifar10'

srun -p gvembodied \
    -x SH-IDC1-10-140-0-197 \
    --job-name=doit_c16_transfer \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=8 main_soft.py \
    --model $MODEL --warmup_epochs 50 --epochs 300 \
    --batch_size 64 --lr 2e-3 --update_freq 1 --use_amp true \
    --mask_mode soft --mask_schedule linear --cutoff_soft 25 \
    --finetune $PRETRAINED \
    --data_path $root_dataset \
    --data_set CIFAR10 \
    --output_dir $OUTPUT
