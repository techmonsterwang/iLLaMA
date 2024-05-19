srun -p gvembodied \
    --job-name=feat_vis_rank \
    --gres=gpu:1 \
    --cpus-per-task=24 \
    --quotatype=reserved \
    python feat_vis_rank.py