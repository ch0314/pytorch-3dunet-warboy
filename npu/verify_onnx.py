import torch
import yaml # PyYAML
import os
import numpy as np
import onnx
import onnxruntime
from pytorch3dunet.unet3d.model import UNet3D

onnx_file_path = "/users/chanmin/3D_Unet/pytorch-3dunet/npu/unet3d.onnx"
checkpoint_path = "/users/chanmin/3D_Unet/pytorch-3dunet/best_checkpoint.pytorch"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dummy_input_shape = (1, 1, 320, 960, 1000)

ort_session = onnxruntime.InferenceSession(onnx_file_path, providers=['CUDAExecutionProvider' if device.type == 'cuda' else 'CPUExecutionProvider'])
verify_input_np = np.random.randn(*dummy_input_shape).astype(np.float32)
ort_inputs = {ort_session.get_inputs()[0].name: verify_input_np}
ort_outputs = ort_session.run(None, ort_inputs)
onnx_output = ort_outputs[0]
print(f"ONNX Runtime Output, shape: {onnx_output.shape}")


model = UNet3D(in_channels=1,
                out_channels=1,
                final_sigmoid=True,
                f_maps=32,
                layer_order="gcr",
                num_groups=8,
                )
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)

verify_input_torch = torch.from_numpy(verify_input_np).to(device)
model.eval()
with torch.no_grad():
    pytorch_output_tensor = model(verify_input_torch)
    pytorch_output = pytorch_output_tensor.cpu().numpy()
    print(f"PyTorch model Output shape: {pytorch_output.shape}")


if np.allclose(pytorch_output, onnx_output, atol=1e-4):
    print("Sucess!")
else:
    mae = np.mean(np.abs(pytorch_output - onnx_output))
    print(f"False onnx, MAE: {mae}")


