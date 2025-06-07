#!/bin/bash

# Define the ONNX model path
MODEL_PATH="npu_pipeline/examples/confocal_boundary/models/unet3d_i8.onnx"

# Define the NPU device target
DEVICE_TARGET="warboy(2)*1"

# Define arrays for batch sizes and worker counts
BATCH_SIZES=(1 2 4 8)
WORKER_COUNTS=(2 4 8)

# Loop through each batch size
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    # Loop through each worker count
    for WORKER_COUNT in "${WORKER_COUNTS[@]}"; do
        echo "======================================================================"
        echo "Running benchmark with Batch Size: ${BATCH_SIZE}, Worker Count: ${WORKER_COUNT}"
        echo "======================================================================"

        furiosa-bench "${MODEL_PATH}" -n 5 -b "${BATCH_SIZE}" -w "${WORKER_COUNT}" -t 1 -d "${DEVICE_TARGET}"

        echo "" # Add an empty line for better readability between runs
    done
done

echo "All benchmarks completed."