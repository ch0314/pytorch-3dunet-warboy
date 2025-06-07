import h5py
import yaml
import numpy as np
import torch
import os
from scipy.ndimage import zoom
from pytorch3dunet.augment import transforms
from pytorch3dunet.datasets.utils import calculate_stats
from pytorch3dunet.unet3d.utils import get_logger

h5_file_path = "/users/chanmin/3D_Unet/pytorch-3dunet/npu/dataset/validation/N_420_ds2x.h5"
yaml_config_path = "/users/chanmin/3D_Unet/pytorch-3dunet/resources/3DUnet_confocal_boundary/test_config.yml"

def image_reshape(original_image: torch.tensor, target_shape = (128, 128, 128)):
    current_shape = original_image.shape
    if current_shape != target_shape:
        factors = [ts / cs for ts, cs in zip(target_shape, current_shape)]
    resized_image_data = zoom(original_image, factors, order=1, mode='nearest')

    return resized_image_data


def preprocess_h5_for_npu(h5_file_path: str, yaml_config_path: str, device_str: str = 'cpu'):
    with open(yaml_config_path, 'r') as f:
        config = yaml.safe_load(f)
    transformer_config = config['loaders']['test']['transformer']

    # Loading
    with h5py.File(h5_file_path, 'r') as f:
        raw_image_data = f["raw"][:]
        label = f["label"][:]


    # Resize
    reshaped_image_data = image_reshape(raw_image_data)
    reshaped_label = image_reshape(label)

    # Standardize
    stats = calculate_stats(reshaped_image_data)

    transformer = transforms.Transformer(transformer_config, stats)
    raw_transform_fn = transformer.raw_transform()
    transformed_tensor = raw_transform_fn(reshaped_image_data)

    if transformed_tensor.ndim == 4: 
        processed_tensor = transformed_tensor.unsqueeze(0)
    
    # To numpy
    processed_numpy = processed_tensor.cpu().numpy().astype(np.float32)

    return reshaped_label, processed_numpy


if __name__ == '__main__':
    preprocess_h5_for_npu(h5_file_path, yaml_config_path)