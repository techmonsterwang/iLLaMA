set -x

CONFIG=configs/vit/upernet/upernet_vit_tiny_12_512_slide_160k_ade20k_ms2.py
PRETRAINED=/mnt/petrelfs/wangjiahao/DoiT/pretrained/vit-tiny-in1k.pth
WORD_DIR=work_dirs/lv4_train_upernet_vit_tiny_512_160k_ade20k
GPUS=8
PORT=${PORT:-29278}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p gvembodied \
    --job-name=lv4_train_upernet_vit_tiny_512_160k_ade20k \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        tools/train.py $CONFIG --work-dir $WORD_DIR --seed 0 --deterministic \
        --options model.pretrained=$PRETRAINED --launcher pytorch