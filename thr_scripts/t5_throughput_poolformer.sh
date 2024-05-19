cd ../
root_imagenet='/mnt/petrelfs/wangjiahao/datasets/classificaton/imagenet/'

for i in $(seq 1 3)  
do   
echo $(($i * 3 + 1));  
srun -p gvembodied \
    -w SH-IDC1-10-140-1-57 \
    --job-name=t5_throughput \
    --gres=gpu:1 \
    --cpus-per-task=24 \
    --quotatype=reserved \
    python3 throughput.py $root_imagenet --model poolformer_s12 -b 1024 --img-size 224 --num-classes 1000
done   

for i in $(seq 1 3)  
do   
echo $(($i * 3 + 1));  
srun -p gvembodied \
    -w SH-IDC1-10-140-1-57 \
    --job-name=t5_throughput \
    --gres=gpu:1 \
    --cpus-per-task=24 \
    --quotatype=reserved \
    python3 throughput.py $root_imagenet --model poolformer_m36 -b 1024 --img-size 224 --num-classes 1000
done   

for i in $(seq 1 3)  
do   
echo $(($i * 3 + 1));  
srun -p gvembodied \
    -w SH-IDC1-10-140-1-57 \
    --job-name=t5_throughput \
    --gres=gpu:1 \
    --cpus-per-task=24 \
    --quotatype=reserved \
    python3 throughput.py $root_imagenet --model poolformer_m48 -b 1024 --img-size 224 --num-classes 1000
done   