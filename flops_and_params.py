import torch
import torch.nn as nn
from thop import profile
from fvcore.nn import FlopCountAnalysis, flop_count_table
from functools import partial
from models.vision_transformer import VisionTransformer as ViT
from models.doit1 import VisionTransformer as DoiT1
from models.doit2 import VisionTransformer as DoiT2
from models.doit3 import VisionTransformer as DoiT3
from models.doit5 import VisionTransformer as DoiT5
from models.doit4 import VisionTransformer as DoiT4
from models.doit6 import VisionTransformer as DoiT6
from models.doit7 import VisionTransformer as DoiT7
from models.doit16 import VisionTransformer as DoiT16
from models.doit22 import VisionTransformer as DoiT22
from models.doit22_hr import VisionTransformer as DoiT22HR
from models.doit22 import RMSNorm


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


input = torch.randn(1, 3, 224, 224)

vit_t = ViT(
    patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(vit_t, inputs=(input, ))
flops = FlopCountAnalysis(vit_t, input)
print('--- --- ---')
print("vit_t: GFLOPs: {}".format(flops.total() / 1e9))
print("vit_t: number of params (M): {}".format(params / 1e6))


vit_b = ViT(
    patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(vit_b, inputs=(input, ))
flops = FlopCountAnalysis(vit_b, input)
print('--- --- ---')
print("vit_b: GFLOPs: {}".format(flops.total() / 1e9))
print("vit_b: number of params (M): {}".format(params / 1e6))

vit_l = ViT(
    patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(vit_l, inputs=(input, ))
flops = FlopCountAnalysis(vit_l, input)
print('--- --- ---')
print("vit_l: GFLOPs: {}".format(flops.total() / 1e9))
print("vit_l: number of params (M): {}".format(params / 1e6))

doit1_t = DoiT1(
    patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit1_t, inputs=(input, ))
flops = FlopCountAnalysis(doit1_t, input)
print('--- --- ---')
print("doit1_t: GFLOPs: {}".format(flops.total() / 1e9))
print("doit1_t: number of params (M): {}".format(params / 1e6))

doit1_b = DoiT1(
    patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit1_b, inputs=(input, ))
flops = FlopCountAnalysis(doit1_b, input)
print('--- --- ---')
print("doit1_b: GFLOPs: {}".format(flops.total() / 1e9))
print("doit1_b: number of params (M): {}".format(params / 1e6))


doit2_t = DoiT2(
    patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit2_t, inputs=(input, ))
flops = FlopCountAnalysis(doit2_t, input)
print('--- --- ---')
print("doit2_t: GFLOPs: {}".format(flops.total() / 1e9))
print("doit2_t: number of params (M): {}".format(params / 1e6))

doit2_b = DoiT2(
    patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit2_b, inputs=(input, ))
flops = FlopCountAnalysis(doit2_b, input)
print('--- --- ---')
print("doit2_b: GFLOPs: {}".format(flops.total() / 1e9))
print("doit2_b: number of params (M): {}".format(params / 1e6))


doit3_t = DoiT3(
    patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit3_t, inputs=(input, ))
flops = FlopCountAnalysis(doit3_t, input)
print('--- --- ---')
print("doit3_t: GFLOPs: {}".format(flops.total() / 1e9))
print("doit3_t: number of params (M): {}".format(params / 1e6))

doit3_b = DoiT3(
    patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit3_b, inputs=(input, ))
flops = FlopCountAnalysis(doit3_b, input)
print('--- --- ---')
print("doit3_b: GFLOPs: {}".format(flops.total() / 1e9))
print("doit3_b: number of params (M): {}".format(params / 1e6))


doit5_t = DoiT5(
    patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit5_t, inputs=(input, ))
flops = FlopCountAnalysis(doit5_t, input)
print('--- --- ---')
print("doit5_t: GFLOPs: {}".format(flops.total() / 1e9))
print("doit5_t: number of params (M): {}".format(params / 1e6))

doit5_b = DoiT5(
    patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit5_b, inputs=(input, ))
flops = FlopCountAnalysis(doit5_b, input)
print('--- --- ---')
print("doit5_b: GFLOPs: {}".format(flops.total() / 1e9))
print("doit5_b: number of params (M): {}".format(params / 1e6))


doit4_t = DoiT4(
    patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit4_t, inputs=(input, ))
flops = FlopCountAnalysis(doit4_t, input)
print('--- --- ---')
print("doit4_t: GFLOPs: {}".format(flops.total() / 1e9))
print("doit4_t: number of params (M): {}".format(params / 1e6))

doit4_b = DoiT4(
    patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit4_b, inputs=(input, ))
flops = FlopCountAnalysis(doit4_b, input)
print('--- --- ---')
print("doit4_b: GFLOPs: {}".format(flops.total() / 1e9))
print("doit4_b: number of params (M): {}".format(params / 1e6))


doit6_t = DoiT6(
    patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit6_t, inputs=(input, ))
flops = FlopCountAnalysis(doit6_t, input)
print('--- --- ---')
print("doit6_t: GFLOPs: {}".format(flops.total() / 1e9))
print("doit6_t: number of params (M): {}".format(params / 1e6))

doit6_b = DoiT6(
    patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit6_b, inputs=(input, ))
flops = FlopCountAnalysis(doit6_b, input)
print('--- --- ---')
print("doit6_b: GFLOPs: {}".format(flops.total() / 1e9))
print("doit6_b: number of params (M): {}".format(params / 1e6))


doit7_t = DoiT7(
    patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit7_t, inputs=(input, ))
flops = FlopCountAnalysis(doit7_t, input)
print('--- --- ---')
print("doit7_t: GFLOPs: {}".format(flops.total() / 1e9))
print("doit7_t: number of params (M): {}".format(params / 1e6))

doit7_b = DoiT7(
    patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit7_b, inputs=(input, ))
flops = FlopCountAnalysis(doit7_b, input)
print('--- --- ---')
print("doit7_b: GFLOPs: {}".format(flops.total() / 1e9))
print("doit7_b: number of params (M): {}".format(params / 1e6))


doit16_t = DoiT16(
    patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit16_t, inputs=(input, ))
flops = FlopCountAnalysis(doit16_t, input)
print('--- --- ---')
print("doit16_t: GFLOPs: {}".format(flops.total() / 1e9))
print("doit16_t: number of params (M): {}".format(params / 1e6))

doit16_s = DoiT16(
    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit16_s, inputs=(input, ))
flops = FlopCountAnalysis(doit16_s, input)
print('--- --- ---')
print("doit16_s: GFLOPs: {}".format(flops.total() / 1e9))
print("doit16_s: number of params (M): {}".format(params / 1e6))

doit16_b = DoiT16(
    patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit16_b, inputs=(input, ))
flops = FlopCountAnalysis(doit16_b, input)
print('--- --- ---')
print("doit16_b: GFLOPs: {}".format(flops.total() / 1e9))
print("doit16_b: number of params (M): {}".format(params / 1e6))

doit16_l = DoiT16(
    patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit16_l, inputs=(input, ))
flops = FlopCountAnalysis(doit16_l, input)
print('--- --- ---')
print("doit16_l: GFLOPs: {}".format(flops.total() / 1e9))
print("doit16_l: number of params (M): {}".format(params / 1e6))


doit22_t = DoiT22(
    patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit22_t, inputs=(input, ))
flops = FlopCountAnalysis(doit22_t, input)
print('--- --- ---')
print("doit22_t: GFLOPs: {}".format(flops.total() / 1e9))
print("doit22_t: number of params (M): {}".format(params / 1e6))

doit22_s = DoiT22(
    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit22_s, inputs=(input, ))
flops = FlopCountAnalysis(doit22_s, input)
print('--- --- ---')
print("doit22_s: GFLOPs: {}".format(flops.total() / 1e9))
print("doit22_s: number of params (M): {}".format(params / 1e6))

doit22_b = DoiT22(
    patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit22_b, inputs=(input, ))
flops = FlopCountAnalysis(doit22_b, input)
print('--- --- ---')
print("doit22_b: GFLOPs: {}".format(flops.total() / 1e9))
print("doit22_b: number of params (M): {}".format(params / 1e6))

doit22_l = DoiT22(
    patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit22_l, inputs=(input, ))
flops = FlopCountAnalysis(doit22_l, input)
print('--- --- ---')
print("doit22_l: GFLOPs: {}".format(flops.total() / 1e9))
print("doit22_l: number of params (M): {}".format(params / 1e6))


input = torch.randn(1, 3, 384, 384)


vit_b = ViT(
    img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(vit_b, inputs=(input, ))
flops = FlopCountAnalysis(vit_b, input)
print('--- --- ---')
print("vit_b_384: GFLOPs: {}".format(flops.total() / 1e9))
print("vit_b_384: number of params (M): {}".format(params / 1e6))

vit_l = ViT(
    img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(vit_l, inputs=(input, ))
flops = FlopCountAnalysis(vit_l, input)
print('--- --- ---')
print("vit_l_384: GFLOPs: {}".format(flops.total() / 1e9))
print("vit_l_384: number of params (M): {}".format(params / 1e6))

doit22_b = DoiT22(
    img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit22_b, inputs=(input, ))
flops = FlopCountAnalysis(doit22_b, input)
print('--- --- ---')
print("doit22_b_384: GFLOPs: {}".format(flops.total() / 1e9))
print("doit22_b_384: number of params (M): {}".format(params / 1e6))

doit22_l = DoiT22(
    img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
flops, params = profile(doit22_l, inputs=(input, ))
flops = FlopCountAnalysis(doit22_l, input)
print('--- --- ---')
print("doit22_l_384: GFLOPs: {}".format(flops.total() / 1e9))
print("doit22_l_384: number of params (M): {}".format(params / 1e6))