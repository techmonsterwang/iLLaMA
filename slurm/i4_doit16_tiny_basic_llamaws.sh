set -x

root_imagenet='/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/'
MODEL=doit22_tiny
OUTPUT='output/i4_doit16_tiny_basic_llamaws'
FINETUNE='/mnt/petrelfs/wangjiahao/DoiT/llama2/pretrained/illama_ws_tiny.pth'

srun -p eval \
    -x SH-IDC1-10-140-24-20 \
    --job-name=llmeval_i4 \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=8 main_soft_ceph.py \
    --model $MODEL --epochs 300 --warmup_epochs 5 --mixup 0.1 --cutmix 0.1 \
    --batch_size 128 --lr 2e-3 --update_freq 4 \
    --dropout 0 --drop_mode standard \
    --mask_mode soft --mask_schedule constant --cutoff_soft 50 \
    --finetune $FINETUNE \
    --data_path $root_imagenet \
    --output_dir $OUTPUT