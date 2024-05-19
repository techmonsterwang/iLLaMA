set -x

CONFIG=configs/vit/upernet/upernet_vit_base_12_512_slide_160k_ade20k_ms1.py
PRETRAINED=/mnt/petrelfs/wangjiahao/DoiT/pretrained/vit-base-in1k.pth
WORD_DIR=work_dirs/lv6_train_upernet_vit_base_512_160k_ade20k
GPUS=8
PORT=${PORT:-29273}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p gvembodied \
    --job-name=lv6_train_upernet_vit_base_512_160k_ade20k \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        tools/train.py $CONFIG --work-dir $WORD_DIR --seed 0 --deterministic \
        --options model.pretrained=$PRETRAINED --launcher pytorch