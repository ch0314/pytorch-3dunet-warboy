#!/usr/bin/env python3
import argparse
import asyncio
import yaml
import os
import sys
from pathlib import Path
import h5py

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
    parser.add_argument('--batch-size', type=int, default=None,
                    help='Override batch size from config')
    parser.add_argument('--num-workers', type=int, default=None,
                    help='Override number of workers from config')

    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if args.batch_size:
        config['loaders']['batch_size'] = args.batch_size
        logger.info(f"Overriding batch size to: {args.batch_size}")
    
    if args.num_workers:
        config['loaders']['num_workers'] = args.num_workers
        logger.info(f"Overriding num workers to: {args.num_workers}")
        
    # Setup output directory
    output_dir = config.get('loaders', {}).get('output_dir', 'output')
    models_dir = os.path.join(os.path.dirname(args.config), 'models')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Step 1: Export model to ONNX
    logger.info("Step 1: Exporting model to ONNX")
    exporter = ModelExporter(config)
    onnx_path = os.path.join(models_dir, 'unet3d.onnx')
    exported_path = exporter.export_to_onnx(onnx_path)
    
    # Step 2: Quantize model
    logger.info("Step 2: Quantizing model")
    quantizer = Quantizer(config)
    quantized_path = os.path.join(models_dir, 'unet3d_i8.onnx')
    quantized_path = quantizer.quantize_model(
        exported_path,
        quantized_path,
        calibration_samples=5
    )
    
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
    
    try:
        if args.mode == 'sync':
            results = engine.run_inference_sync()
        else:
            # Use asyncio.run for async mode
            results = asyncio.run(engine.run_inference_async())
            
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return
    
    # Save results
    results_path = os.path.join(output_dir, 'inference_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    logger.info(f"Inference results saved to: {results_path}")
    
    # Save predictions as H5 files
    successful_results = [r for r in results['results'] if r.get('status') == 'success']
    
    for result in successful_results:
        file_name = result['file_name']
        prediction = result.get('prediction')
        
        if prediction is not None:
            pred_path = os.path.join(output_dir, file_name.replace('.h5', '_pred.h5'))
            with h5py.File(pred_path, 'w') as f:
                f.create_dataset('prediction', data=prediction, compression='gzip')
                
                # Save probability map if available
                if 'probability' in result:
                    f.create_dataset('probability', data=result['probability'], compression='gzip')
                    
                # Save metrics if available
                if 'metrics' in result:
                    for key, value in result['metrics'].items():
                        f.attrs[key] = value
                        
            logger.info(f"Saved prediction: {pred_path}")
    
    # Print summary
    logger.info(f"\nPipeline Summary:")
    logger.info(f"  Total files: {len(results['results'])}")
    logger.info(f"  Successful: {len(successful_results)}")
    logger.info(f"  Failed: {len(results['results']) - len(successful_results)}")
    
    # Print metrics summary if available
    if successful_results and 'metrics' in successful_results[0]:
        dice_scores = [r['metrics']['dice'] for r in successful_results if 'metrics' in r]
        if dice_scores:
            avg_dice = sum(dice_scores) / len(dice_scores)
            logger.info(f"  Average Dice score: {avg_dice:.4f}")
    
    logger.info("Pipeline completed successfully!")


if __name__ == '__main__':
    main()