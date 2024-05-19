set -x

CONFIG=configs/doit/upernet/upernet_doit_base_12_512_slide_160k_ade20k_ms1.py
GPUS=1
PORT=${PORT:-29432}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p eval \
    --job-name=get_flops_doit_base \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --quotatype=reserved \
    python tools/get_flops.py $CONFIG --shape 512 512