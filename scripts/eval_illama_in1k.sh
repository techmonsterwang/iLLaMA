set -x

root_imagenet='/mnt/petrelfs/wangjiahao/datasets/classificaton/imagenet/'

# illama-tiny: 75.0
MODEL=illama_tiny
RESUME='/mnt/petrelfs/wangjiahao/DoiT/pretrained/illama-tiny-in1k-75.0.pth'
OUTPUT='output/eval_tiny_750'

srun -p gvembodied \
    --job-name=debug \
    --gres=gpu:2 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=spot \
    python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --model $MODEL --eval true \
    --data_path $root_imagenet \
    --resume $RESUME \
