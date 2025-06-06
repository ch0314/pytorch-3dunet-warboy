#!/usr/bin/env python3
import argparse
import asyncio
import yaml
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.loader import H5Loader
from src.preprocessor import Preprocessor
from src.model_exporter import ModelExporter
from src.quantizer import Quantizer
from src.benchmark import Benchmark
from src.inference_engine import InferenceEngine
from src.postprocessor import Postprocessor

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='NPU Pipeline for 3D U-Net')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['sync', 'async'], default='sync',
                        help='Inference mode')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark')
    parser.add_argument('--export-only', action='store_true',
                        help='Only export and quantize model')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup output directory
    output_dir = config.get('loaders', {}).get('output_dir', 'output')
    models_dir = os.path.join(os.path.dirname(args.config), 'models')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Step 1: Export model to ONNX
    logger.info("Step 1: Exporting model to ONNX")
    exporter = ModelExporter(config)
    onnx_path = os.path.join(models_dir, 'unet3d.onnx')
    # exported_path = exporter.export_to_onnx(onnx_path)
    
    # Step 2: Quantize model
    logger.info("Step 2: Quantizing model")
    quantizer = Quantizer(config)
    quantized_path = os.path.join(models_dir, 'unet3d_i8.onnx')
    # quantized_path = quantizer.quantize_model(
    #     exported_path,
    #     quantized_path,
    #     calibration_samples=5
    # )
    
    if args.export_only:
        logger.info("Model export and quantization completed")
        return
    
    # Step 3: Run benchmark (optional)
    if args.benchmark:
        logger.info("Step 3: Running benchmark")
        benchmark = Benchmark(config)
        results = benchmark.run_comprehensive_benchmark(
            quantized_path,
            batch_sizes=[1, 2, 4],
            worker_nums=[1, 2, 4, 8]
        )
        
        # Find optimal configuration
        optimal_batch, optimal_workers = benchmark.find_optimal_configuration(results)
        
        # Save benchmark results
        benchmark_path = os.path.join(output_dir, 'benchmark_results.yaml')
        with open(benchmark_path, 'w') as f:
            yaml.dump(results, f)
        logger.info(f"Benchmark results saved to: {benchmark_path}")
    
    # Step 4: Run inference
    logger.info("Step 4: Running inference")
    engine = InferenceEngine(config, quantized_path)
    
    if args.mode == 'sync':
        results = engine.run_inference_sync()
    else:
        results = asyncio.run(engine.run_inference_async())
    
    # Save results
    results_path = os.path.join(output_dir, 'inference_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    logger.info(f"Inference results saved to: {results_path}")
    
    # Save predictions as H5 files
    for result in results['results']:
        file_name = result['file_name']
        prediction = result['prediction']
        
        pred_path = os.path.join(output_dir, file_name.replace('.h5', '_pred.h5'))
        import h5py
        with h5py.File(pred_path, 'w') as f:
            f.create_dataset('prediction', data=prediction, compression='gzip')
            
    logger.info("Pipeline completed successfully!")


if __name__ == '__main__':
    main()