set -x

root_dataset='/mnt/petrelfs/wangjiahao/datasets/classificaton/cifar100/'
PRETRAINED='/mnt/petrelfs/wangjiahao/DoiT/pretrained/vit-tiny-in1k.pth'
MODEL=vit_tiny
OUTPUT='output/t1_vit_tiny_cifar100'

srun -p gvembodied \
    --job-name=doit_t1_transfer \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=8 main_transfer.py \
    --model $MODEL --warmup_epochs 50 --epochs 300 \
    --batch_size 64 --lr 2e-3 --update_freq 1 --use_amp true \
    --finetune $PRETRAINED \
    --data_path $root_dataset \
    --data_set CIFAR100 \
    --output_dir $OUTPUT



