set -x

root_imagenet='/mnt/petrelfs/wangjiahao/datasets/classificaton/imagenet/'

# illama-base: 81.6
MODEL=illama_base
RESUME='/mnt/petrelfs/wangjiahao/DoiT/pretrained/illama-base-in1k-384-83.0.pth'

srun -p gvembodied \
    --job-name=evaluation_384 \
    --gres=gpu:2 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=2 main_soft_fthr.py \
    --model $MODEL --input_size 384 --eval true \
    --data_path $root_imagenet \
    --resume $RESUME


# illama-base: 85.0
MODEL=illama_base
RESUME='/mnt/petrelfs/wangjiahao/DoiT/pretrained/illama-base-in21kin1k-384-85.0.pth'

srun -p gvembodied \
    --job-name=evaluation_384 \
    --gres=gpu:2 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=2 main_soft_fthr.py \
    --model $MODEL --input_size 384 --eval true \
    --data_path $root_imagenet \
    --resume $RESUME


# illama-large: 86.0
MODEL=illama_large
RESUME='/mnt/petrelfs/wangjiahao/DoiT/pretrained/illama-large-in21kin1k-384-86.0.pth'

srun -p gvembodied \
    --job-name=evaluation_384 \
    --gres=gpu:2 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=2 main_soft_fthr.py \
    --model $MODEL --input_size 384 --eval true \
    --data_path $root_imagenet \
    --resume $RESUME