set -x

CONFIG=configs/vit/upernet/upernet_vit_tiny_12_512_slide_160k_ade20k_ms.py
CHECKPOINT=/xun/lian/wan/fuzhi/dao.pth
GPUS=4
PORT=${PORT:-29278}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p gvembodied \
    --job-name=lv1_test_upernet_vit_tiny_512_160k_ade20k_ms \
    --gres=gpu:4 \
    --cpus-per-task=16 \
    --quotatype=spot \
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        tools/test.py $CONFIG $CHECKPOINT --eval mIoU --aug-test --launcher pytorch