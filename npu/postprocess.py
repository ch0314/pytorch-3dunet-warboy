import os, subprocess, asyncio
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def save_numpy_as_image(numpy_array, filename, slice_index=0):
    plt.figure()
    plt.imshow(numpy_array[slice_index], cmap='gray')
    plt.title(f'Slice {slice_index}')
    plt.savefig(filename)
    plt.close()
    print(f"Saved image: {filename}")

def predict_boundary(pred, data_name, threshold=0.01):
    if(pred.dim == 4 and pred.shape[0] == 1): # squeeze channel dimension
        np.squeeze(pred)
    predicted_boundary = (pred > threshold).astype(np.uint8)
    save_numpy_as_image(predicted_boundary, f"result/{data_name}_predicted_slice_0.png", 0)