import os
import numpy as np
import h5py
from npu.preprocess import preprocess_h5_for_npu
from furiosa.runtime.sync import create_runner
from sklearn.metrics import f1_score  

def calculate_dice_score(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    return f1_score(y_true_flat, y_pred_flat, average='macro')

def evaluate_threshold(model_path, data_paths, yaml_config_path):
    thresholds = np.arange(0.1, 0.9, 0.05)  
    with create_runner(model_path, device="warboy(2)*1") as runner:
        best_threshold = 0.5
        best_dice_score = 0
        for data_path in data_paths:
            true_label, input_for_npu = preprocess_h5_for_npu(data_path, yaml_config_path)
            preds = runner.run([input_for_npu])[0]

            for threshold in thresholds:
                predicted_mask = (np.squeeze(preds) > threshold).astype(np.uint8)

                if true_label.shape != predicted_mask.shape:
                    print(f"Warning: Shape mismatch between true label ({true_label.shape}) and prediction ({predicted_mask.shape}) for {data_path}. Skipping.")
                    continue

                dice_score = calculate_dice_score(true_label, predicted_mask)
                if dice_score > best_dice_score:
                    best_dice_score = dice_score
                    best_threshold = threshold
            
    return best_threshold

def find_optimal_threshold(model_path, validation_data_dir, yaml_config_path, thresholds):
    validation_files = [os.path.join(validation_data_dir, f) for f in os.listdir(validation_data_dir) if f.endswith('.h5')]

    best_threshold = evaluate_threshold(model_path, validation_files, yaml_config_path)

    return best_threshold

model_path = "unet3d_i8.onnx"
validation_data_dir = "dataset/validation"  
yaml_config_path = "/users/chanmin/3D_Unet/pytorch-3dunet/resources/3DUnet_confocal_boundary/test_config.yml"
thresholds = np.arange(0.1, 0.9, 0.05)  

optimal_threshold = find_optimal_threshold(model_path, validation_data_dir, yaml_config_path, thresholds)
print(optimal_threshold)