import torch
state_dict = torch.load('/mnt/petrelfs/wangjiahao/DoiT/llama2/1.pth', map_location='cpu')

print(state_dict.items())



