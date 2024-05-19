set -x

root_imagenet='/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/'
MODEL=doit22_base
OUTPUT='output/hft11_doit22_base_basic'
FINETUNE='/mnt/petrelfs/wangjiahao/DoiT/pretrained/illama-base-in1k-81.6.pth'

srun -p eval \
    --job-name=llmeval_hft11 \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=8 main_soft_fthr_ceph.py \
    --model $MODEL --drop_path 0.8 --drop_mode standard --input_size 384 \
    --batch_size 32 --lr 8e-5 --update_freq 2 \
    --warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
    --cutmix 0 --mixup 0 \
    --mask_mode soft --mask_schedule constant --cutoff_soft 0 \
    --finetune $FINETUNE \
    --data_path $root_imagenet \
    --output_dir $OUTPUT
