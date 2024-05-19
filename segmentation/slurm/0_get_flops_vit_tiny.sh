set -x

CONFIG=configs/vit/upernet/upernet_vit_tiny_12_512_slide_160k_ade20k_ms3.py
GPUS=1
PORT=${PORT:-29451}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p eval \
    --job-name=get_flops_vit_tiny \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --quotatype=reserved \
    python tools/get_flops.py $CONFIG --shape 512 512