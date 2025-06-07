import os
import onnx
import numpy as np

from tqdm import tqdm
from furiosa.optimizer import optimize_model
from furiosa.quantizer import (
    CalibrationMethod, Calibrator,quantize,
)
from npu.preprocess import preprocess_h5_for_npu

# 1. Load ONNX Model
model = onnx.load("unet3d.onnx")

# 2. ONNX Graph Optimization
model = optimize_model(
    model = model, opset_version=13, input_shapes={"input": [1, 1, 128, 128, 128]}
)

# FuriosaAI SDK Calibrator: onnx model, calibration method
calibrator = Calibrator(model, CalibrationMethod.MIN_MAX_ASYM)

# 3. Load dataset
data_dir = "dataset/validation"
yaml_config_path = "/users/chanmin/3D_Unet/pytorch-3dunet/resources/3DUnet_confocal_boundary/test_config.yml"
calibration_dataset = [f for f in os.listdir(data_dir) if f.endswith('.h5')]


# 4. Preprocessing
for data_name in tqdm(calibration_dataset):
    data_path = os.path.join(data_dir, data_name)

    label, input = preprocess_h5_for_npu(data_path, yaml_config_path)
    
    np.expand_dims(input, axis=0)

    calibrator.collect_data([[input]])

# Calculate Calibration Ranges
calibration_range = calibrator.compute_range()
# Qauntization
quantized_model = quantize(model, calibration_range)

with open("unet3d_i8.onnx", "wb") as f:
    f.write(bytes(quantized_model))