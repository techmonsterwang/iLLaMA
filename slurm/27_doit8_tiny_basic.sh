set -x

root_imagenet='/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/'
MODEL=doit8_tiny
OUTPUT='output/27_doit8_tiny_basic'

srun -p eval \
    --job-name=llmeval_27 \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=8 main_ceph.py \
    --model $MODEL --epochs 300 \
    --batch_size 128 --lr 4e-3 --update_freq 4 \
    --dropout 0 --drop_mode standard \
    --data_path $root_imagenet \
    --output_dir $OUTPUT