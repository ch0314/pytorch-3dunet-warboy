import collections
import torch
import yaml # PyYAML
import os
import numpy as np
import onnx
import onnxruntime
from pytorch3dunet.unet3d.model import UNet3D


# --- 1. Setting --- 
checkpoint_path = "/users/chanmin/3D_Unet/pytorch-3dunet/best_checkpoint.pytorch"
config_path = "/users/chanmin/3D_Unet/pytorch-3dunet/resources/3DUnet_confocal_boundary/test_config.yml"
onnx_output_path = "unet3d.onnx"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"device: {device}")
print(f"Pytorch checkpoint: {checkpoint_path}")
print(f"Config yaml: {config_path}")
print(f"output onnx: {onnx_output_path}")

# --- 2. YAML Config Load ---
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Cannot find: {config_path}")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# --- 3. Model Info ---
model_config = config['model']
model_name = model_config.get('name', 'UNet3D')
in_channels = model_config.get('in_channels')
out_channels = model_config.get('out_channels')
f_maps = model_config.get('f_maps', 64) 
layer_order = model_config.get('layer_order') 
num_groups = model_config.get('num_groups') 
final_sigmoid = model_config.get('final_sigmoid') 

model = UNet3D(in_channels=1,
                        out_channels=1,
                        final_sigmoid=True,
                        f_maps=32,
                        layer_order="gcr",
                        num_groups=8,
                        )


# --- 4. Weight Load---
checkpoint = torch.load(checkpoint_path, map_location=device)

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint
    print("Warning: state_dict is checkpoint itself")

model.load_state_dict(state_dict)

# --- 5. ONNX Export ---
model.eval()

dummy_input_shape = (1, 1, 128, 128, 128)
dummy_input = torch.randn(dummy_input_shape, dtype=torch.float32)

torch.onnx.export(model,
                    dummy_input,
                    onnx_output_path,
                    export_params=True,
                    opset_version=13,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input' : {0 : 'batch_size', 2 : 'depth', 3 : 'height', 4 : 'width'},
                                  'output' : {0 : 'batch_size', 2 : 'depth', 3 : 'height', 4 : 'width'}}
                    )
onnx_model = onnx.load(onnx_output_path)
onnx.checker.check_model(onnx_model)