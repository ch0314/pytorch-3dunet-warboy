import os, subprocess, asyncio
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from npu.preprocess import preprocess_h5_for_npu
from furiosa.runtime.sync import create_runner

def save_numpy_as_image(numpy_array, filename):
    plt.imshow(numpy_array, cmap='gray')
    plt.title(f'Slice {filename}')
    plt.savefig(filename)
    plt.close()
    print(f"Saved image: {filename}")


def furiosa_runtime_sync(model_path, input_, threshold = 0.8):
    with create_runner(model_path, device = "warboy(2)*1") as runner:
        print(f"Running inference once for input shape: {input_.shape}")
        start_time = time.time()
        preds = runner.run([input_])
        end_time = time.time()
        print(f"Inference finished in {end_time - start_time:.4f} seconds. output shape: {preds[0].shape}")

        predicted_boundary = (np.squeeze(preds[0]) > threshold).astype(np.uint8)
        mip_predicted_image = np.max(predicted_boundary, axis=0)
        save_numpy_as_image(mip_predicted_image, f"result/{data_name}_predicted_mip.png")

        if input_ is not None:
            mip_input_image = np.max(np.squeeze(input_), axis=0)
            save_numpy_as_image(mip_input_image, f"result/{data_name}_mip.png")

        print("done with inference")
        return predicted_boundary


model_path = "unet3d_i8.onnx"
data_dir = "dataset/test"

if os.path.exists("result"):
    subprocess.run(["rm", "-rf", "result"])
os.makedirs("result")

# Preprocessing
yaml_config_path = "/users/chanmin/3D_Unet/pytorch-3dunet/resources/3DUnet_confocal_boundary/test_config.yml"
dataset = [f for f in os.listdir(data_dir) if f.endswith('.h5')]

for data_name in dataset:
    data_path = os.path.join(data_dir, data_name)
    label, input = preprocess_h5_for_npu(data_path, yaml_config_path)
    output = furiosa_runtime_sync(model_path, input)
