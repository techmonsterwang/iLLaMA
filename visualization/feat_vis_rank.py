import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from functools import partial
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/mnt/petrelfs/wangjiahao/DoiT')
from PIL import Image
import torchvision.transforms as T
import numpy as np
from visualization.models_vis.vit_vis import VisionTransformer as ViT
from visualization.models_vis.doit_vis import VisionTransformer as DoiT
from visualization.models_vis.doit_vis import RMSNorm

# ------------------------------------------------------------
# 1.create data
# ------------------------------------------------------------
val_dataset_path = '/mnt/petrelfs/wangjiahao/datasets/classificaton/imagenet/val'

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_dataset = ImageFolder(val_dataset_path, transform=transform)

# 随机抽取 30 张图片
random_indices = np.random.choice(len(val_dataset), 30, replace=False)
subset_dataset = torch.utils.data.Subset(val_dataset, random_indices)
subset_loader = DataLoader(subset_dataset, batch_size=1, shuffle=False)



# ------------------------------------------------------------
# 2.create model
# ------------------------------------------------------------
print("Creating Model!")
vit_t = ViT(
    patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, 
    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
checkpoint = torch.load('/mnt/petrelfs/wangjiahao/DoiT/pretrained/vit-tiny-in1k-73.8.pth', map_location='cpu')
if 'model' in checkpoint.keys():
    checkpoint = checkpoint['model']
vit_t.load_state_dict(checkpoint, strict=True)

doit_t = DoiT(
    patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
checkpoint = torch.load('/mnt/petrelfs/wangjiahao/DoiT/pretrained/illama-tiny-in1k-75.0.pth', map_location='cpu')
if 'model' in checkpoint.keys():
    checkpoint = checkpoint['model']
doit_t.load_state_dict(checkpoint, strict=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_t.to(device)
doit_t.to(device)
vit_t.eval()
doit_t.eval()



# ------------------------------------------------------------
# 4.hook the input
# ------------------------------------------------------------
for i in vit_t.state_dict().keys():
    print(i)
for i in doit_t.state_dict().keys():
    print(i)

# propagate through the model
print('Forward propagation started!')

attn_map_vectors_vit_t = [[] for _ in range(12)]
attn_map_vectors_doit_t = [[] for _ in range(12)]

with torch.no_grad():
    for inputs, _ in subset_loader:
        inputs = inputs.to(device)

        # inference vit-tiny
        output_vit_t, attn_map_list_vit_t = vit_t(inputs)
        for i in range(12):
            attn_map_vectors_vit_t[i].append(attn_map_list_vit_t[i].clone())

        # inference doit-tiny
        output_doit_t, attn_map_list_doit_t = doit_t(inputs)
        for i in range(12):
            attn_map_vectors_doit_t[i].append(attn_map_list_doit_t[i].clone())

# 计算每个位置的平均值
attn_map_mean_vit_t = [torch.mean(torch.stack(attn_map_vectors_vit_t[i]), dim=0) for i in range(12)]
attn_map_mean_doit_t = [torch.mean(torch.stack(attn_map_vectors_doit_t[i]), dim=0) for i in range(12)]

print('Forward propagation completed!')


print('attn_map_mean_vit_t, shape:', len(attn_map_mean_vit_t))
print('attn_map_mean_doit_t, shape:', len(attn_map_mean_doit_t))

print('attn_map_mean_vit_t layer 1, shape:', attn_map_mean_vit_t[0].shape)
print('attn_map_mean_doit_t layer 1, shape:', attn_map_mean_doit_t[0].shape)
print('attn_map_mean_vit_t layer 4, shape:', attn_map_mean_vit_t[3].shape)
print('attn_map_mean_doit_t layer 4, shape:', attn_map_mean_doit_t[3].shape)
print('attn_map_mean_vit_t layer 8, shape:', attn_map_mean_vit_t[7].shape)
print('attn_map_mean_doit_t layer 8, shape:', attn_map_mean_doit_t[7].shape)
print('attn_map_mean_vit_t layer 12, shape:', attn_map_mean_vit_t[11].shape)
print('attn_map_mean_doit_t layer 12, shape:', attn_map_mean_doit_t[11].shape)


print('extracting attention map of layer 1, 4, 8, 12, head 1:')
attn_map_vit_t_layer1_head1 = attn_map_mean_vit_t[0][0,0,:,:]
attn_map_vit_t_layer4_head1 = attn_map_mean_vit_t[3][0,0,:,:]
attn_map_vit_t_layer8_head1 = attn_map_mean_vit_t[7][0,0,:,:]
attn_map_vit_t_layer12_head1 = attn_map_mean_vit_t[11][0,0,:,:]

attn_map_doit_t_layer1_head1 = attn_map_mean_doit_t[0][0,0,:,:]
attn_map_doit_t_layer4_head1 = attn_map_mean_doit_t[3][0,0,:,:]
attn_map_doit_t_layer8_head1 = attn_map_mean_doit_t[7][0,0,:,:]
attn_map_doit_t_layer12_head1 = attn_map_mean_doit_t[11][0,0,:,:]

attn_map_vit_t_layer1_head2 = attn_map_mean_vit_t[0][0,1,:,:]
attn_map_vit_t_layer4_head2 = attn_map_mean_vit_t[3][0,1,:,:]
attn_map_vit_t_layer8_head2 = attn_map_mean_vit_t[7][0,1,:,:]
attn_map_vit_t_layer12_head2 = attn_map_mean_vit_t[11][0,1,:,:]

attn_map_doit_t_layer1_head2 = attn_map_mean_doit_t[0][0,1,:,:]
attn_map_doit_t_layer4_head2 = attn_map_mean_doit_t[3][0,1,:,:]
attn_map_doit_t_layer8_head2 = attn_map_mean_doit_t[7][0,1,:,:]
attn_map_doit_t_layer12_head2 = attn_map_mean_doit_t[11][0,1,:,:]

attn_map_vit_t_layer1_head3 = attn_map_mean_vit_t[0][0,2,:,:]
attn_map_vit_t_layer4_head3 = attn_map_mean_vit_t[3][0,2,:,:]
attn_map_vit_t_layer8_head3 = attn_map_mean_vit_t[7][0,2,:,:]
attn_map_vit_t_layer12_head3 = attn_map_mean_vit_t[11][0,2,:,:]

attn_map_doit_t_layer1_head3 = attn_map_mean_doit_t[0][0,2,:,:]
attn_map_doit_t_layer4_head3 = attn_map_mean_doit_t[3][0,2,:,:]
attn_map_doit_t_layer8_head3 = attn_map_mean_doit_t[7][0,2,:,:]
attn_map_doit_t_layer12_head3 = attn_map_mean_doit_t[11][0,2,:,:]

print(attn_map_vit_t_layer1_head1.shape)
print(attn_map_doit_t_layer1_head1.shape)
print(attn_map_vit_t_layer4_head1.shape)
print(attn_map_doit_t_layer4_head1.shape)
print(attn_map_vit_t_layer8_head1.shape)
print(attn_map_doit_t_layer8_head1.shape)
print(attn_map_vit_t_layer12_head1.shape)
print(attn_map_doit_t_layer12_head1.shape)

print(attn_map_vit_t_layer1_head2.shape)
print(attn_map_doit_t_layer1_head2.shape)
print(attn_map_vit_t_layer4_head2.shape)
print(attn_map_doit_t_layer4_head2.shape)
print(attn_map_vit_t_layer8_head2.shape)
print(attn_map_doit_t_layer8_head2.shape)
print(attn_map_vit_t_layer12_head2.shape)
print(attn_map_doit_t_layer12_head2.shape)

print(attn_map_vit_t_layer1_head3.shape)
print(attn_map_doit_t_layer1_head3.shape)
print(attn_map_vit_t_layer4_head3.shape)
print(attn_map_doit_t_layer4_head3.shape)
print(attn_map_vit_t_layer8_head3.shape)
print(attn_map_doit_t_layer8_head3.shape)
print(attn_map_vit_t_layer12_head3.shape)
print(attn_map_doit_t_layer12_head3.shape)


# # ------------------------------------------------------------
# # 4.visualize the distribution of feature
# # ------------------------------------------------------------

# layer 1
# 奇异值分解
_, s_vit_1_1, _ = torch.svd(attn_map_vit_t_layer1_head1.cpu())
_, s_doit_1_1, _ = torch.svd(attn_map_doit_t_layer1_head1.cpu())
_, s_vit_1_2, _ = torch.svd(attn_map_vit_t_layer1_head2.cpu())
_, s_doit_1_2, _ = torch.svd(attn_map_doit_t_layer1_head2.cpu())
_, s_vit_1_3, _ = torch.svd(attn_map_vit_t_layer1_head3.cpu())
_, s_doit_1_3, _ = torch.svd(attn_map_doit_t_layer1_head3.cpu())

# 对奇异值向量按照从大到小的顺序排列
s_vit_1_1 = s_vit_1_1.sort(descending=True).values
s_doit_1_1 = s_doit_1_1.sort(descending=True).values
s_vit_1_2 = s_vit_1_2.sort(descending=True).values
s_doit_1_2 = s_doit_1_2.sort(descending=True).values
s_vit_1_3 = s_vit_1_3.sort(descending=True).values
s_doit_1_3 = s_doit_1_3.sort(descending=True).values

# 归一化并进行累加求和
singular_vit_t_1_1 = torch.cumsum(s_vit_1_1 / s_vit_1_1.sum(), dim=0)
singular_doit_t_1_1 = torch.cumsum(s_doit_1_1 / s_doit_1_1.sum(), dim=0)
singular_vit_t_1_2 = torch.cumsum(s_vit_1_2 / s_vit_1_2.sum(), dim=0)
singular_doit_t_1_2 = torch.cumsum(s_doit_1_2 / s_doit_1_2.sum(), dim=0)
singular_vit_t_1_3 = torch.cumsum(s_vit_1_3 / s_vit_1_3.sum(), dim=0)
singular_doit_t_1_3 = torch.cumsum(s_doit_1_3 / s_doit_1_3.sum(), dim=0)

# 画图
plt.figure(1)
plt.plot(range(1, 198), singular_vit_t_1_1.numpy(), color='#0071BC', label='ViT-T')
plt.plot(range(1, 198), singular_doit_t_1_1.numpy(), color='#EB9B78', label='iLLaMa-T')

plt.xlabel('singular value index')
plt.ylabel('normalized cumulative singular value')
plt.legend()

plt.tight_layout()
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer1_head1.pdf')
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer1_head1.svg')

plt.figure(2)
plt.plot(range(1, 198), singular_vit_t_1_2.numpy(), color='#0071BC', label='ViT-T')
plt.plot(range(1, 198), singular_doit_t_1_2.numpy(), color='#EB9B78', label='iLLaMa-T')

plt.xlabel('singular value index')
plt.ylabel('normalized cumulative singular value')
plt.legend()

plt.tight_layout()
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer1_head2.pdf')
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer1_head2.svg')

plt.figure(3)
plt.plot(range(1, 198), singular_vit_t_1_3.numpy(), color='#0071BC', label='ViT-T')
plt.plot(range(1, 198), singular_doit_t_1_3.numpy(), color='#EB9B78', label='iLLaMa-T')

plt.xlabel('singular value index')
plt.ylabel('normalized cumulative singular value')
plt.legend()

plt.tight_layout()
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer1_head3.pdf')
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer1_head3.svg')



# layer 4
# 奇异值分解
_, s_vit_4_1, _ = torch.svd(attn_map_vit_t_layer4_head1.cpu())
_, s_doit_4_1, _ = torch.svd(attn_map_doit_t_layer4_head1.cpu())
_, s_vit_4_2, _ = torch.svd(attn_map_vit_t_layer4_head2.cpu())
_, s_doit_4_2, _ = torch.svd(attn_map_doit_t_layer4_head2.cpu())
_, s_vit_4_3, _ = torch.svd(attn_map_vit_t_layer4_head3.cpu())
_, s_doit_4_3, _ = torch.svd(attn_map_doit_t_layer4_head3.cpu())

# 对奇异值向量按照从大到小的顺序排列
s_vit_4_1 = s_vit_4_1.sort(descending=True).values
s_doit_4_1 = s_doit_4_1.sort(descending=True).values
s_vit_4_2 = s_vit_4_2.sort(descending=True).values
s_doit_4_2 = s_doit_4_2.sort(descending=True).values
s_vit_4_3 = s_vit_4_3.sort(descending=True).values
s_doit_4_3 = s_doit_4_3.sort(descending=True).values

# 归一化并进行累加求和
singular_vit_t_4_1 = torch.cumsum(s_vit_4_1 / s_vit_4_1.sum(), dim=0)
singular_doit_t_4_1 = torch.cumsum(s_doit_4_1 / s_doit_4_1.sum(), dim=0)
singular_vit_t_4_2 = torch.cumsum(s_vit_4_2 / s_vit_4_2.sum(), dim=0)
singular_doit_t_4_2 = torch.cumsum(s_doit_4_2 / s_doit_4_2.sum(), dim=0)
singular_vit_t_4_3 = torch.cumsum(s_vit_4_3 / s_vit_4_3.sum(), dim=0)
singular_doit_t_4_3 = torch.cumsum(s_doit_4_3 / s_doit_4_3.sum(), dim=0)

# 画图
plt.figure(4)
plt.plot(range(1, 198), singular_vit_t_4_1.numpy(), color='#0071BC', label='ViT-T')
plt.plot(range(1, 198), singular_doit_t_4_1.numpy(), color='#EB9B78', label='iLLaMa-T')

plt.xlabel('singular value index')
plt.ylabel('normalized cumulative singular value')
plt.legend()

plt.tight_layout()
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer4_head1.pdf')
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer4_head1.svg')

plt.figure(5)
plt.plot(range(1, 198), singular_vit_t_4_2.numpy(), color='#0071BC', label='ViT-T')
plt.plot(range(1, 198), singular_doit_t_4_2.numpy(), color='#EB9B78', label='iLLaMa-T')

plt.xlabel('singular value index')
plt.ylabel('normalized cumulative singular value')
plt.legend()

plt.tight_layout()
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer4_head2.pdf')
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer4_head2.svg')

plt.figure(6)
plt.plot(range(1, 198), singular_vit_t_4_3.numpy(), color='#0071BC', label='ViT-T')
plt.plot(range(1, 198), singular_doit_t_4_3.numpy(), color='#EB9B78', label='iLLaMa-T')

plt.xlabel('singular value index')
plt.ylabel('normalized cumulative singular value')
plt.legend()

plt.tight_layout()
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer4_head3.pdf')
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer4_head3.svg')



# layer 8
# 奇异值分解
_, s_vit_8_1, _ = torch.svd(attn_map_vit_t_layer8_head1.cpu())
_, s_doit_8_1, _ = torch.svd(attn_map_doit_t_layer8_head1.cpu())
_, s_vit_8_2, _ = torch.svd(attn_map_vit_t_layer8_head2.cpu())
_, s_doit_8_2, _ = torch.svd(attn_map_doit_t_layer8_head2.cpu())
_, s_vit_8_3, _ = torch.svd(attn_map_vit_t_layer8_head3.cpu())
_, s_doit_8_3, _ = torch.svd(attn_map_doit_t_layer8_head3.cpu())

# 对奇异值向量按照从大到小的顺序排列
s_vit_8_1 = s_vit_8_1.sort(descending=True).values
s_doit_8_1 = s_doit_8_1.sort(descending=True).values
s_vit_8_2 = s_vit_8_2.sort(descending=True).values
s_doit_8_2 = s_doit_8_2.sort(descending=True).values
s_vit_8_3 = s_vit_8_3.sort(descending=True).values
s_doit_8_3 = s_doit_8_3.sort(descending=True).values

# 归一化并进行累加求和
singular_vit_t_8_1 = torch.cumsum(s_vit_8_1 / s_vit_8_1.sum(), dim=0)
singular_doit_t_8_1 = torch.cumsum(s_doit_8_1 / s_doit_8_1.sum(), dim=0)
singular_vit_t_8_2 = torch.cumsum(s_vit_8_2 / s_vit_8_2.sum(), dim=0)
singular_doit_t_8_2 = torch.cumsum(s_doit_8_2 / s_doit_8_2.sum(), dim=0)
singular_vit_t_8_3 = torch.cumsum(s_vit_8_3 / s_vit_8_3.sum(), dim=0)
singular_doit_t_8_3 = torch.cumsum(s_doit_8_3 / s_doit_8_3.sum(), dim=0)

# 画图
plt.figure(7)
plt.plot(range(1, 198), singular_vit_t_8_1.numpy(), color='#0071BC', label='ViT-T')
plt.plot(range(1, 198), singular_doit_t_8_1.numpy(), color='#EB9B78', label='iLLaMa-T')

plt.xlabel('singular value index')
plt.ylabel('normalized cumulative singular value')
plt.legend()

plt.tight_layout()
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer8_head1.pdf')
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer8_head1.svg')

plt.figure(8)
plt.plot(range(1, 198), singular_vit_t_8_2.numpy(), color='#0071BC', label='ViT-T')
plt.plot(range(1, 198), singular_doit_t_8_2.numpy(), color='#EB9B78', label='iLLaMa-T')

plt.xlabel('singular value index')
plt.ylabel('normalized cumulative singular value')
plt.legend()

plt.tight_layout()
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer8_head2.pdf')
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer8_head2.svg')

plt.figure(9)
plt.plot(range(1, 198), singular_vit_t_8_3.numpy(), color='#0071BC', label='ViT-T')
plt.plot(range(1, 198), singular_doit_t_8_3.numpy(), color='#EB9B78', label='iLLaMa-T')

plt.xlabel('singular value index')
plt.ylabel('normalized cumulative singular value')
plt.legend()

plt.tight_layout()
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer8_head3.pdf')
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer8_head3.svg')



# layer 12
# 奇异值分解
_, s_vit_12_1, _ = torch.svd(attn_map_vit_t_layer12_head1.cpu())
_, s_doit_12_1, _ = torch.svd(attn_map_doit_t_layer12_head1.cpu())
_, s_vit_12_2, _ = torch.svd(attn_map_vit_t_layer12_head2.cpu())
_, s_doit_12_2, _ = torch.svd(attn_map_doit_t_layer12_head2.cpu())
_, s_vit_12_3, _ = torch.svd(attn_map_vit_t_layer12_head3.cpu())
_, s_doit_12_3, _ = torch.svd(attn_map_doit_t_layer12_head3.cpu())

# 对奇异值向量按照从大到小的顺序排列
s_vit_12_1 = s_vit_12_1.sort(descending=True).values
s_doit_12_1 = s_doit_12_1.sort(descending=True).values
s_vit_12_2 = s_vit_12_2.sort(descending=True).values
s_doit_12_2 = s_doit_12_2.sort(descending=True).values
s_vit_12_3 = s_vit_12_3.sort(descending=True).values
s_doit_12_3 = s_doit_12_3.sort(descending=True).values

# 归一化并进行累加求和
singular_vit_t_12_1 = torch.cumsum(s_vit_12_1 / s_vit_12_1.sum(), dim=0)
singular_doit_t_12_1 = torch.cumsum(s_doit_12_1 / s_doit_12_1.sum(), dim=0)
singular_vit_t_12_2 = torch.cumsum(s_vit_12_2 / s_vit_12_2.sum(), dim=0)
singular_doit_t_12_2 = torch.cumsum(s_doit_12_2 / s_doit_12_2.sum(), dim=0)
singular_vit_t_12_3 = torch.cumsum(s_vit_12_3 / s_vit_12_3.sum(), dim=0)
singular_doit_t_12_3 = torch.cumsum(s_doit_12_3 / s_doit_12_3.sum(), dim=0)

# 画图
plt.figure(10)
plt.plot(range(1, 198), singular_vit_t_12_1.numpy(), color='#0071BC', label='ViT-T')
plt.plot(range(1, 198), singular_doit_t_12_1.numpy(), color='#EB9B78', label='iLLaMa-T')

plt.xlabel('singular value index')
plt.ylabel('normalized cumulative singular value')
plt.legend()

plt.tight_layout()
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer12_head1.pdf')
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer12_head1.svg')

plt.figure(11)
plt.plot(range(1, 198), singular_vit_t_12_2.numpy(), color='#0071BC', label='ViT-T')
plt.plot(range(1, 198), singular_doit_t_12_2.numpy(), color='#EB9B78', label='iLLaMa-T')

plt.xlabel('singular value index')
plt.ylabel('normalized cumulative singular value')
plt.legend()

plt.tight_layout()
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer12_head2.pdf')
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer12_head2.svg')

plt.figure(12)
plt.plot(range(1, 198), singular_vit_t_12_3.numpy(), color='#0071BC', label='ViT-T')
plt.plot(range(1, 198), singular_doit_t_12_3.numpy(), color='#EB9B78', label='iLLaMa-T')

plt.xlabel('singular value index')
plt.ylabel('normalized cumulative singular value')
plt.legend()

plt.tight_layout()
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer12_head3.pdf')
plt.savefig('./feat_rank_analysis/feat_vis_rank_analysis_layer12_head3.svg')