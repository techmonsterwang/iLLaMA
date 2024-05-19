set -x

root_dataset='/mnt/petrelfs/wangjiahao/datasets/classificaton/cifar100/'
PRETRAINED='/mnt/petrelfs/wangjiahao/DoiT/pretrained/doit-tiny-in1k.pth'
MODEL=doit16_tiny
OUTPUT='output/r8_doit_tiny_cifar100'

srun -p gvembodied \
    -x SH-IDC1-10-140-1-160 \
    --job-name=doit_r8_transfer \
    --gres=gpu:4 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=spot \
    python -m torch.distributed.launch --nproc_per_node=4 main_soft_transfer.py \
    --model $MODEL --warmup_epochs 50 --epochs 300 \
    --batch_size 256 --lr 2e-3 --update_freq 1 --use_amp true \
    --mask_mode soft --mask_schedule constant --cutoff_soft 50 \
    --finetune $PRETRAINED \
    --data_path $root_dataset \
    --data_set CIFAR100 \
    --output_dir $OUTPUT
