# [Adapting LLaMA Decoder to Vision Transformer](https://arxiv.org/pdf/2404.06773)

<p align="center">
<a href="https://arxiv.org/pdf/2404.06773" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2404.06773-b31b1b.svg?style=flat" /></a>
</p>


<p align="center">
<img src="https://github.com/hpcaitech/Open-Sora/assets/48375204/8c6c1428-89e9-41d3-862d-2c2be7d65656" width="376"> <br>
<small>Image credit: DALL·E</small>
</p>


This is a PyTorch implementation of iLLaMA proposed by our paper "[Adapting LLaMA Decoder to Vision Transformer](https://arxiv.org/abs/2404.06773)". 


![iLLaMA first figure](https://github.com/hpcaitech/Open-Sora/assets/48375204/59f7af9a-679c-46ea-a428-c7bf27c0ecea)
Figure 1: Left: iLLaMA architecture. Right: our design roadmap. Colored and gray bars
represent the results of the tiny and base regimes, with the red line depicting the training loss of the
tiny regime. iLLaMA strives to process visual tokens using standard LLaMa components, e.g., causal
self-attention. The proposed PS [cls] and soft mask strategy help overcome training challenges. 

<br>

![iLLaMA second figure](https://github.com/hpcaitech/Open-Sora/assets/48375204/6dffefaa-cb27-49ba-a258-1953bdaa7330)
Figure 2: (a) mask in causal self-attention. (b) mask in causal self-attention with our post-sequence
class token (PS [cls]) method. (c) modified causal mask.

<br>

![iLLaMA third figure](https://github.com/hpcaitech/Open-Sora/assets/48375204/f3b46c50-c807-4997-81d4-257b6168e5f7)
Figure 3: (a) Soft mask gradually transitions from a bi-directional mask into a causal mask during
training through a constant or linear schedule. (b) Ablation results of training loss and test accuracy.



## Requirements
PyTorch and timm 0.5.4 (`pip install timm==0.5.4`).

Data preparation: ImageNet with the following folder structure, you can extract ImageNet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```


## Models
### iLLaMA on ImageNet-1K
| Model | Pre-trained dataset | Resolution | Params | MACs | Top1 Acc |
| :---     | :---     |   :---:    |  :---: |  :---:  |  :---:  |
| [illama_tiny](https://github.com/techmonsterwang/iLLaMA/releases/download/model/mambaout_femto.pth) | - | 224 | 5.7M | 1.3G | 75.0 |
| [illama_small](https://github.com/techmonsterwang/iLLaMA/releases/download/model/mambaout_tiny.pth) | - | 224 | 21.9M | 4.6G | 79.9 |
| [illama_base](https://github.com/techmonsterwang/iLLaMA/releases/download/model/mambaout_small.pth) | - | 224 | 86.3M | 17.6G | 81.6 |
| [illama_base](https://github.com/techmonsterwang/iLLaMA/releases/download/model/mambaout_small.pth) | - | 384 | 86.3M | 55.5G | 83.0 |
| [illama_base](https://github.com/techmonsterwang/iLLaMA/releases/download/model/mambaout_small.pth) | ImageNet-21K | 224 | 86.3M | 17.6G | 83.6 |
| [illama_base](https://github.com/techmonsterwang/iLLaMA/releases/download/model/mambaout_small.pth) | ImageNet-21K | 384 | 86.3M | 55.5G | 85.0 |
| [illama_large](https://github.com/techmonsterwang/iLLaMA/releases/download/model/mambaout_small.pth) | ImageNet-21K | 224 | 310.2M | 62.8G | 84.8 |
| [illama_large](https://github.com/techmonsterwang/iLLaMA/releases/download/model/mambaout_small.pth) | ImageNet-21K | 384 | 310.2M | 194.7G | 86.0 |




## Evaluate

To evaluate models on 224 resolution, run:

```bash
MODEL=illama_tiny
RESUME='/your/path/to/illama-tiny-in1k-75.0.pth'

python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --model $MODEL --eval true \
    --data_path $root_imagenet \
    --resume $RESUME
```

To evaluate models on 384 resolution, run:

```bash
MODEL=illama_base
RESUME='/your/path/to/illama-base-in1k-384-83.0.pth'

python -m torch.distributed.launch --nproc_per_node=2 main_soft_fthr.py \
    --model $MODEL --input_size 384 --eval true \
    --data_path $root_imagenet \
    --resume $RESUME
```

## Train
We use batch size of 4096 by default with 8 GPUs. 


```bash
bash scripts/train_illama_tiny_in1k.sh
```
Training scripts of other models are shown in [scripts](/scripts/).


## Initialization Using LLaMA2-7B (Optional)
We use weight selection method to select weights from LLaMA2-7B. 

```bash
python llama2/weight_selection.py
```

Then we use the selected weights to initialize our iLLaMA-T/S/B. 

```bash
bash scripts/train_illama_tiny_from_llama2.sh
```
Training scripts of other models are shown in [scripts](/scripts/). 


## Bibtex
```
@article{wang2024adapting,
  title={Adapting LLaMA Decoder to Vision Transformer},
  author={Wang, Jiahao and Shao, Wenqi and Chen, Mengzhao and Wu, Chengyue and Liu, Yong and Zhang, Kaipeng and Zhang, Songyang and Chen, Kai and Luo, Ping},
  journal={arXiv preprint arXiv:2404.06773},
  year={2024}
}
```

## Acknowledgment

Our implementation is based on [pytorch-image-models](https://github.com/huggingface/pytorch-image-models), [LLaMA](https://github.com/meta-llama/llama), [Dropout](https://github.com/facebookresearch/dropout), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), and [metaformer](https://github.com/sail-sg/metaformer).
