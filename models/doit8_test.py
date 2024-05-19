# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple
from einops import pack, rearrange, repeat, unpack

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'doit8test_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/doit8test_small_p16_224-15ec54c9.pth',
    ),
    'doit8test_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_doit8test_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'doit8test_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_doit8test_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'doit8test_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_doit8test_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'doit8test_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_doit8test_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'doit8test_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_doit8test_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'doit8test_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_doit8test_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'doit8test_huge_patch16_224': _cfg(),
    'doit8test_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'doit8test_small_resnet26d_224': _cfg(),
    'doit8test_small_resnet50d_s3_224': _cfg(),
    'doit8test_base_resnet26d_224': _cfg(),
    'doit8test_base_resnet50d_224': _cfg(),
}


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

        
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, multiple_of=256, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = int(2 * hidden_features / 3)
        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=False)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x):
        x = F.silu(self.fc1(x)) * self.fc3(x) # [B, N + num_reg + num_cls, 4D*2/3]
        # print(x.shape)
        x = self.fc2(x)
        return x
    

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # [32], float32
    t = torch.arange(end, device=freqs.device)   # [197], int64
    freqs = torch.outer(t, freqs).float()   # [197, 32], float32
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [197, 32], complex64
    return freqs_cis  # [197, 32], complex64

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim  # 4, since [bsz, 197, 3, 32], complex64
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])  
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]  # [1, 197, 1, 32], list
    return freqs_cis.view(*shape)  # [1, 197, 1, 32], complex64

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [bsz, 197, 3, 64], float32 → [bsz, 197, 3, 32, 2], float32 → [bsz, 197, 3, 32], complex64
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # [bsz, 197, 3, 64], float32 → [bsz, 197, 3, 32, 2], float32 → [bsz, 197, 3, 32], complex64
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)  # [1, 197, 1, 32], complex64
    # print('freqs_cis:', freqs_cis.shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # [bsz, 197, 3, 64], float32
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)  # [bsz, 197, 3, 64], float32
    return xq_out.type_as(xq), xk_out.type_as(xk)  # [bsz, 197, 3, 64], float32, [bsz, 197, 3, 64], float32


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4) # [3, B, N, self.num_heads, C // self.num_heads]
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) # [B, N, self.num_heads, C // self.num_heads]

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        q = q.transpose(1, 2)  # [B, self.num_heads, N, C // self.num_heads]
        k = k.transpose(1, 2)  # [B, self.num_heads, N, C // self.num_heads]
        v = v.transpose(1, 2)  # [B, self.num_heads, N, C // self.num_heads]
        attn = (q @ k.transpose(-2, -1)) * self.scale # [B, self.num_heads, N, N]
        if mask is not None:
            attn = attn + mask  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=RMSNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        x = x + self.drop_path(self.attn(self.norm1(x), freqs_cis, mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, num_reg_tokens=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=RMSNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # I add these two lines
        self.drop_rate=drop_rate
        attn_drop_rate=drop_rate
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # register
        self.reg_token = nn.Parameter(torch.zeros(1, num_reg_tokens, embed_dim))

        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # remove the original learnable positional embedding
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.depth = depth

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.reg_token, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.freqs_cis = precompute_freqs_cis(
            self.num_features // num_heads, num_patches + num_reg_tokens + 1
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'}
        return {'reg_token', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        freqs_cis = self.freqs_cis.to(x.device)
        # print('freqs_cis:', freqs_cis.shape)

        reg_tokens = self.reg_token.expand(B, -1, -1)  
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((x, reg_tokens, cls_tokens), dim=1)
        # print('x:', x.shape)
        # x = x + self.pos_embed
        x = self.pos_drop(x)
        N = x.shape[1]

        mask = None
        if N > 1:
            mask = torch.full((1, 1, N, N), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1).type_as(x)

        for blk in self.blocks:
            x = blk(x, freqs_cis, mask)

        x = self.norm(x)
        return x[:, -1]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
    def update_drop_path(self, drop_path_rate):
        self.drop_path = drop_path_rate
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        for i in range(self.depth):
            self.blocks[i].drop_path.drop_prob = dp_rates[i]
    
    def update_dropout(self, drop_rate):
        self.drop_rate = drop_rate
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

@register_model
def doit8test_tiny(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, num_reg_tokens=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), **kwargs)
    return model

@register_model
def doit8test_small(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, num_reg_tokens=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), **kwargs)
    return model

@register_model
def doit8test_base(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, num_reg_tokens=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), **kwargs)
    return model

@register_model
def doit8test_large(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, num_reg_tokens=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), **kwargs)
    return model


