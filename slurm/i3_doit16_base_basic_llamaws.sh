set -x

root_imagenet='/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/'
MODEL=doit22_base
OUTPUT='output/i3_doit16_base_basic_llamaws'
FINETUNE='/mnt/petrelfs/wangjiahao/DoiT/llama2/pretrained/illama_ws_base.pth'

srun -p gvembodied \
    --job-name=llmeval_i3 \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=8 main_soft_ceph.py \
    --model $MODEL --epochs 300 --mixup 0.95 --cutmix 1.0 \
    --batch_size 128 --lr 4e-3 --update_freq 4 \
    --drop_path 0.4 --drop_mode standard \
    --mask_mode soft --mask_schedule linear --cutoff_soft 25 \
    --finetune $FINETUNE \
    --data_path $root_imagenet \
    --output_dir $OUTPUT