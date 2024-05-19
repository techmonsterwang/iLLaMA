set -x

root_imagenet='/mnt/petrelfs/wangjiahao/datasets/classificaton/imagenet/'

# illama-tiny: 75.0
MODEL=illama_tiny
RESUME='/mnt/petrelfs/wangjiahao/DoiT/pretrained/illama-tiny-in1k-75.0.pth'

srun -p gvembodied \
    --job-name=evaluation_224 \
    --gres=gpu:2 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --model $MODEL --eval true \
    --data_path $root_imagenet \
    --resume $RESUME


# illama-small: 79.9
MODEL=illama_small
RESUME='/mnt/petrelfs/wangjiahao/DoiT/pretrained/illama-small-in1k-79.9.pth'

srun -p gvembodied \
    --job-name=evaluation_224 \
    --gres=gpu:2 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --model $MODEL --eval true \
    --data_path $root_imagenet \
    --resume $RESUME


# illama-base: 81.6
MODEL=illama_base
RESUME='/mnt/petrelfs/wangjiahao/DoiT/pretrained/illama-base-in1k-81.6.pth'

srun -p gvembodied \
    --job-name=evaluation_224 \
    --gres=gpu:2 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --model $MODEL --eval true \
    --data_path $root_imagenet \
    --resume $RESUME


# illama-base: 83.6
MODEL=illama_base
RESUME='/mnt/petrelfs/wangjiahao/DoiT/pretrained/illama-base-in21kin1k-224-83.6.pth'

srun -p gvembodied \
    --job-name=evaluation_224 \
    --gres=gpu:2 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --model $MODEL --eval true \
    --data_path $root_imagenet \
    --resume $RESUME