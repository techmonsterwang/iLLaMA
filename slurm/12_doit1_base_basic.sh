set -x

root_imagenet='/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/'
MODEL=doit1_base
OUTPUT='output/12_doit1_base_basic'

srun -p eval \
    --job-name=llmeval_12 \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=8 main_ceph.py \
    --model $MODEL --epochs 300 \
    --batch_size 128 --lr 4e-3 --update_freq 4 \
    --drop_path 0.4 --drop_mode standard \
    --data_path $root_imagenet \
    --output_dir $OUTPUT
