set -x

root_imagenet='/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/'
MODEL=doit7_tiny
OUTPUT='output/63_doit7_tiny_basic'
LOG_DIR='log'
JOB_NAME='llmeval_63'

srun -p gvembodied \
    --job-name=${JOB_NAME} \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=spot \
    --async -o $LOG_DIR/${JOB_NAME}.log \
    python -m torch.distributed.launch --nproc_per_node=8 main_ceph.py \
    --model $MODEL --epochs 300 --mixup 0.3 --cutmix 0.3 \
    --batch_size 128 --lr 4e-3 --update_freq 4 \
    --dropout 0 --drop_mode standard \
    --data_path $root_imagenet \
    --output_dir $OUTPUT