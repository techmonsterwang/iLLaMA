set -x

root_dataset='/mnt/petrelfs/wangjiahao/datasets/classificaton/cifar10/'
PRETRAINED='/mnt/petrelfs/wangjiahao/DoiT/pretrained/doit-tiny-in1k-75.1.pth'
MODEL=doit7_tiny
OUTPUT='output/r19_doit_tiny_cifar10'

srun -p eval \
    --job-name=doit_r19_transfer \
    --gres=gpu:4 \
    --cpus-per-task=16 \
    --preempt \
    --quotatype=spot \
    python -m torch.distributed.launch --nproc_per_node=4 main_transfer.py \
    --model $MODEL --warmup_epochs 50 --epochs 300 \
    --batch_size 256 --lr 2e-3 --update_freq 1 --use_amp true \
    --finetune $PRETRAINED \
    --data_path $root_dataset \
    --data_set CIFAR10 \
    --output_dir $OUTPUT



