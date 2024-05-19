set -x

root_imagenet='/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/'
MODEL=doit22_base
OUTPUT='output/ft15_doit22_base_basic'
FINETUNE='/mnt/petrelfs/wangjiahao/DoiT/output/l6_doit22_base_basic/checkpoint-89.pth'

srun -p gvembodied \
    -x SH-IDC1-10-140-0-201 \
    --job-name=llmeval_ft15 \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=8 main_soft_ceph.py \
    --model $MODEL --drop_path 0.2 --drop_mode standard --input_size 224 \
    --batch_size 32 --lr 1e-4 --update_freq 2 \
    --warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
    --cutmix 0 --mixup 0 \
    --finetune $FINETUNE \
    --data_path $root_imagenet \
    --output_dir $OUTPUT


