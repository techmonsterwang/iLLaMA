set -x

CONFIG=configs/vit/upernet/upernet_vit_base_12_512_slide_160k_ade20k_ms1.py
GPUS=1
PORT=${PORT:-29452}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p eval \
    --job-name=get_flops_vit_base \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --quotatype=reserved \
    python tools/get_flops.py $CONFIG --shape 512 512