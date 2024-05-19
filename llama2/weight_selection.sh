set -x

srun -p gvembodied \
    --job-name=read_llama2 \
    --gres=gpu:1 \
    --cpus-per-task=16 \
    --preempt \
    --quotatype=spot \
    python llama2/weight_selection.py