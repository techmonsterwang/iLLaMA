set -x

root_imagenet='/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/'
MODEL=doit22_small
OUTPUT='output/i5_doit16_small_basic_llamaws'
FINETUNE='/mnt/petrelfs/wangjiahao/DoiT/llama2/pretrained/illama_ws_small.pth'

srun -p eval \
    -x SH-IDC1-10-140-24-20 \
    --job-name=llmeval_i5 \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=8 main_soft_ceph.py \
    --model $MODEL --epochs 300 --warmup_epochs 5 --mixup 0.5 --cutmix 0.5 \
    --batch_size 128 --lr 2e-3 --update_freq 4 \
    --drop_path 0.1 --drop_mode standard \
    --mask_mode soft --mask_schedule linear --cutoff_soft 50 \
    --finetune $FINETUNE \
    --data_path $root_imagenet \
    --output_dir $OUTPUT