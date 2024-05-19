from safetensors import safe_open
import os
import torch
import collections
import sys
from functools import partial
sys.path.insert(0, '/mnt/petrelfs/wangjiahao/DoiT')
from models.illama import VisionTransformer as iLLaMa
from models.illama import RMSNorm


def uniform_element_selection(wt, s_shape):
    assert wt.dim() == len(s_shape), "Tensors have different number of dimensions"
    ws = wt.clone()
    for dim in range(wt.dim()):
        assert wt.shape[dim] >= s_shape[dim], "Teacher's dimension should not be smaller than student's dimension"  # determine whether teacher is larger than student on this dimension
        if wt.shape[dim] % s_shape[dim] == 0:
            step = wt.shape[dim] // s_shape[dim]
            indices = torch.arange(s_shape[dim]) * step
        else:
            indices = torch.round(torch.linspace(0, wt.shape[dim]-1, s_shape[dim])).to(torch.int64)
        ws = torch.index_select(ws, dim, indices)
    assert ws.shape == s_shape
    return ws


# show iLLaMA keys
# illama_t = iLLaMa(
#     patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
#     norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
tensors_illama = torch.load('/mnt/petrelfs/wangjiahao/DoiT/pretrained/illama-tiny-in1k-75.0.pth', map_location='cpu')
if 'model' in tensors_illama.keys():
    tensors_illama = tensors_illama['model']
print("illama keys:")
print("\n".join(tensors_illama.keys()))


# load llama2-7b-hf
tensors_llama2 = {}
path="llama2-7b-hf"
for n in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
    file_name = os.path.join(path, n)
    with safe_open(file_name, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors_llama2[k] = f.get_tensor(k)
print("llama2 keys:")
print("\n".join(tensors_llama2.keys()))
print("load done")

# (576, 192)
print(tensors_illama["blocks.8.attn.qkv.weight"].shape)
# (4096, 4096)
print(tensors_llama2["model.layers.12.self_attn.q_proj.weight"].shape)
# (11008, 4096)
print(tensors_llama2["model.layers.12.mlp.up_proj.weight"].shape)
# (11008, 4096)
print(tensors_llama2["model.layers.12.mlp.gate_proj.weight"].shape)
# (4096)
print(tensors_llama2["model.layers.12.input_layernorm.weight"].shape)


# illama:
# blocks.8.norm1.weight
# blocks.8.attn.qkv.weight
# blocks.8.attn.qkv.bias
# blocks.8.attn.proj.weight
# blocks.8.attn.proj.bias
# blocks.8.norm2.weight
# blocks.8.mlp.fc1.weight
# blocks.8.mlp.fc2.weight
# blocks.8.mlp.fc3.weight

# llama2:
# model.layers.12.input_layernorm.weight
# model.layers.12.mlp.down_proj.weight
# model.layers.12.mlp.gate_proj.weight
# model.layers.12.mlp.up_proj.weight
# model.layers.12.post_attention_layernorm.weight
# model.layers.12.self_attn.k_proj.weight
# model.layers.12.self_attn.o_proj.weight
# model.layers.12.self_attn.q_proj.weight
# model.layers.12.self_attn.rotary_emb.inv_freq
# model.layers.12.self_attn.v_proj.weight

tensors_llama2_to_illama = collections.OrderedDict()
# for k,v in tensors_llama2.items():
for i in range(12):
    # norm
    tensors_llama2_to_illama["blocks." + str(i) + ".norm1.weight"] = \
        tensors_llama2["model.layers." + str(i) + ".input_layernorm.weight"]
    tensors_llama2_to_illama["blocks." + str(i) + ".norm2.weight"] = \
        tensors_llama2["model.layers." + str(i) + ".post_attention_layernorm.weight"]
    # attn
    tensors_llama2_to_illama["blocks." + str(i) + ".attn.qkv.weight"] = \
        torch.cat((tensors_llama2["model.layers." + str(i) + ".self_attn.q_proj.weight"], 
            tensors_llama2["model.layers." + str(i) + ".self_attn.k_proj.weight"], 
            tensors_llama2["model.layers." + str(i) + ".self_attn.v_proj.weight"]), dim=0)
    tensors_llama2_to_illama["blocks." + str(i) + ".attn.proj.weight"] = \
        tensors_llama2["model.layers." + str(i) + ".self_attn.o_proj.weight"]  
    # ffn 
    tensors_llama2_to_illama["blocks." + str(i) + ".mlp.fc1.weight"] = \
        tensors_llama2["model.layers." + str(i) + ".mlp.gate_proj.weight"]  
    tensors_llama2_to_illama["blocks." + str(i) + ".mlp.fc2.weight"] = \
        tensors_llama2["model.layers." + str(i) + ".mlp.down_proj.weight"]  
    tensors_llama2_to_illama["blocks." + str(i) + ".mlp.fc3.weight"] = \
        tensors_llama2["model.layers." + str(i) + ".mlp.up_proj.weight"]    
print("llama2_to_illama keys:")
print("\n".join(tensors_llama2_to_illama.keys()))

# save new
# torch.save(tensors_llama2_to_illama, 'llama2/1.pth')



# weight selection from llama2 to illama
student_illama_tiny = iLLaMa(
    patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
student_illama_small = iLLaMa(
    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)
student_illama_base = iLLaMa(
    patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(RMSNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.0, drop_rate=0.0)

teacher_weights = tensors_llama2_to_illama
student_weights_tiny = student_illama_tiny.state_dict()
student_weights_small = student_illama_small.state_dict()
student_weights_base = student_illama_base.state_dict()

# illama tiny weight selection
student_weight_selection_tiny = collections.OrderedDict()
for key in student_weights_tiny.keys():
    if "norm1.weight" in key or "norm2.weight" in key or "attn.qkv.weight" in key or "attn.proj.weight" in key or "mlp.fc1.weight" in key or "mlp.fc2.weight" in key or "mlp.fc3.weight" in key:
        print(f"initializing {key}")
        print("teacher_weights shape:", teacher_weights[key].shape)
        print("student_weights_tiny shape:", student_weights_tiny[key].shape)
        student_weight_selection_tiny[key] = uniform_element_selection(teacher_weights[key], student_weights_tiny[key].shape)
print("student_weights_tiny initialization done")

# illama small weight selection
student_weight_selection_small = collections.OrderedDict()
for key in student_weights_small.keys():
    if "norm1.weight" in key or "norm2.weight" in key or "attn.qkv.weight" in key or "attn.proj.weight" in key or "mlp.fc1.weight" in key or "mlp.fc2.weight" in key or "mlp.fc3.weight" in key:
        print(f"initializing {key}")
        print("teacher_weights shape:", teacher_weights[key].shape)
        print("student_weights_small shape:", student_weights_small[key].shape)
        student_weight_selection_small[key] = uniform_element_selection(teacher_weights[key], student_weights_small[key].shape)
print("student_weights_small initialization done")

# illama base weight selection
student_weight_selection_base = collections.OrderedDict()
for key in student_weights_base.keys():
    if "norm1.weight" in key or "norm2.weight" in key or "attn.qkv.weight" in key or "attn.proj.weight" in key or "mlp.fc1.weight" in key or "mlp.fc2.weight" in key or "mlp.fc3.weight" in key:
        print(f"initializing {key}")
        print("teacher_weights shape:", teacher_weights[key].shape)
        print("student_weights_base shape:", student_weights_base[key].shape)
        student_weight_selection_base[key] = uniform_element_selection(teacher_weights[key], student_weights_base[key].shape)
print("student_weights_base initialization done")


# check keys for selected tiny small base
for key in student_weight_selection_tiny.keys():
    assert key in student_weights_tiny, f"Key {key} not found in Model iLLaMA-T"
    assert student_weight_selection_tiny[key].shape == student_weights_tiny[key].shape, f"Shape mismatch for key {key}: {student_weight_selection_tiny[key].shape} != {student_weights_tiny[key].shape}"
for key in student_weight_selection_small.keys():
    assert key in student_weights_small, f"Key {key} not found in Model iLLaMA-S"
    assert student_weight_selection_small[key].shape == student_weights_small[key].shape, f"Shape mismatch for key {key}: {student_weight_selection_small[key].shape} != {student_weights_small[key].shape}"
for key in student_weight_selection_base.keys():
    assert key in student_weights_base, f"Key {key} not found in Model iLLaMA-B"
    assert student_weight_selection_base[key].shape == student_weights_base[key].shape, f"Shape mismatch for key {key}: {student_weight_selection_base[key].shape} != {student_weights_base[key].shape}"

print("All keys checked, you can use student_weight_selection_tiny, student_weight_selection_small, student_weight_selection_base.")

# save weight selection for illama
torch.save(student_weight_selection_tiny, 'llama2/pretrained/illama_ws_tiny.pth')
torch.save(student_weight_selection_small, 'llama2/pretrained/illama_ws_small.pth')
torch.save(student_weight_selection_base, 'llama2/pretrained/illama_ws_base.pth')
