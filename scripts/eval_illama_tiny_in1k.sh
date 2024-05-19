root_imagenet='/mnt/petrelfs/wangjiahao/datasets/classificaton/imagenet/'


# illama-tiny: 75.0
MODEL=illama_tiny
RESUME='/mnt/petrelfs/wangjiahao/DoiT/pretrained/illama-tiny-in1k-75.0.pth'

srun -p gvembodied \
    --job-name=eval \
    --gres=gpu:1 \
    --cpus-per-task=16 \
    --preempt \
    --quotatype=spot \
    python main.py --model $MODEL --eval true \
    --resume $RESUME \
    --data_path $root_imagenet


# illama-small: 79.9
MODEL=illama_small
RESUME='/mnt/petrelfs/wangjiahao/DoiT/pretrained/illama-small-in1k-79.9.pth'

srun -p gvembodied \
    --job-name=eval \
    --gres=gpu:1 \
    --cpus-per-task=16 \
    --preempt \
    --quotatype=spot \
    python main.py --model $MODEL --eval true \
    --resume $RESUME \
    --data_path $root_imagenet


# illama-base: 81.6
MODEL=illama_base
RESUME='/mnt/petrelfs/wangjiahao/DoiT/pretrained/illama-base-in1k-81.6.pth'

srun -p gvembodied \
    --job-name=eval \
    --gres=gpu:1 \
    --cpus-per-task=16 \
    --preempt \
    --quotatype=spot \
    python main.py --model $MODEL --eval true \
    --resume $RESUME \
    --data_path $root_imagenet