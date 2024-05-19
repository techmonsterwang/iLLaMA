root_imagenet='/mnt/petrelfs/wangjiahao/datasets/classificaton/imagenet/'
RESUME='/mnt/petrelfs/wangjiahao/DoiT/pretrained/illama-tiny-in1k-75.0.pth'

python main.py --model illama_tiny --eval true \
--resume /path/to/model \
--data_path $root_imagenet





