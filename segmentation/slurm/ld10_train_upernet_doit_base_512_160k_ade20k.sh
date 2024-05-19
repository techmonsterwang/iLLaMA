set -x

CONFIG=configs/doit/upernet/upernet_doit_base_12_512_slide_160k_ade20k_ms1.py
PRETRAINED=/mnt/petrelfs/wangjiahao/DoiT/pretrained/illama-base-in1k-81.6.pth
WORD_DIR=work_dirs/ld10_train_upernet_doit_base_512_160k_ade20k
GPUS=8
PORT=${PORT:-29274}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p eval \
    --job-name=ld10_train_upernet_doit_base_512_160k_ade20k \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --preempt \
    --quotatype=reserved \
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        tools/train.py $CONFIG --work-dir $WORD_DIR --seed 0 --deterministic \
        --options model.pretrained=$PRETRAINED --launcher pytorch