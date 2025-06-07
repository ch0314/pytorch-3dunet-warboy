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
from src.model_exporter import ModelExporter
from src.quantizer import Quantizer
from src.benchmark import Benchmark
from src.inference_engine import InferenceEngine
from src.basic_async_engine import BasicAsyncEngine

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
    parser.add_argument('--mode', type=str, 
                        choices=['benchmark', 'sync', 'async'], 
                        default='sync',
                        help='Pipeline execution mode')
    parser.add_argument('--export-only', action='store_true',
                        help='Only export and quantize model')
    parser.add_argument('--compare-all', action='store_true',
                        help='Run all modes for comparison')
    parser.add_argument('--force-export', action='store_true',
                        help='Force model export even if files exist')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup directories
    output_dir = config.get('loaders', {}).get('output_dir', 'output')
    models_dir = os.path.join(os.path.dirname(args.config), 'models')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Step 1: Setup model (export/quantize if needed)
    onnx_path, quantized_path = setup_model(config, models_dir, args.export_only, args.force_export)
    
    if args.export_only:
        return
    
    # Step 2: Run selected mode or comparison
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

def setup_model(config, models_dir, export_only, force_export):
    """Setup model with export/quantization (skip if already exists)"""
    
    onnx_path = os.path.join(models_dir, 'unet3d.onnx')
    quantized_path = os.path.join(models_dir, 'unet3d_i8.onnx')
    
    # Check if models already exist
    onnx_exists = os.path.exists(onnx_path)
    quantized_exists = os.path.exists(quantized_path)
    
    if not force_export and onnx_exists and quantized_exists:
        logger.info(f"✅ Using existing models:")
        logger.info(f"  ONNX: {onnx_path}")
        logger.info(f"  Quantized: {quantized_path}")
        return onnx_path, quantized_path
    
    logger.info("🔧 Setting up model...")
    
    # Export model to ONNX if needed
    if force_export or not onnx_exists:
        logger.info("📤 Exporting model to ONNX...")
        exporter = ModelExporter(config)
        exported_path = exporter.export_to_onnx(onnx_path)
    else:
        exported_path = onnx_path
        logger.info(f"✅ Using existing ONNX model: {onnx_path}")
    
    # Quantize model if needed
    if force_export or not quantized_exists:
        logger.info("🔢 Quantizing model...")
        quantizer = Quantizer(config)
        quantized_path = quantizer.quantize_model(
            exported_path,
            quantized_path,
            calibration_samples=5
        )
    else:
        logger.info(f"✅ Using existing quantized model: {quantized_path}")
    
    if export_only:
        logger.info("✅ Model export and quantization completed")
        
    return exported_path, quantized_path


async def run_single_mode(mode, config, quantized_path, output_dir):
    """Execute single mode"""
    logger.info(f"🚀 Running {mode} mode...")
    
    if mode == 'benchmark':
        await run_benchmark_mode(config, quantized_path, output_dir)
    elif mode == 'sync':
        await run_sync_mode(config, quantized_path, output_dir)
    elif mode == 'async':
        await run_basic_async_mode(config, quantized_path, output_dir)
    else:
        logger.error(f"❌ Unknown mode: {mode}")


async def run_benchmark_mode(config, quantized_path, output_dir):
    """Benchmark mode execution"""
    logger.info("📊 Running benchmark...")
    
    benchmark = Benchmark(config)
    results = benchmark.run_comprehensive_benchmark(
        quantized_path.replace('_i8.onnx', '.onnx'),
        batch_sizes=[1, 2, 4],
        worker_nums=[1, 2, 4, 8]
    )
    
    # Save results
    benchmark_path = os.path.join(output_dir, 'benchmark_results.yaml')
    with open(benchmark_path, 'w') as f:
        yaml.dump(results, f)
    
    optimal_batch, optimal_workers = benchmark.find_optimal_configuration(results)
    logger.info(f"✅ Benchmark completed. Optimal: batch={optimal_batch}, workers={optimal_workers}")


async def run_sync_mode(config, quantized_path, output_dir):
    """Synchronous mode execution"""
    logger.info("🔄 Running sync mode...")
    
    engine = InferenceEngine(config, quantized_path)
    results = engine.run_inference_sync()
    
    save_results(results, output_dir, 'sync')
    logger.info("✅ Sync mode completed")


async def run_basic_async_mode(config, quantized_path, output_dir):
    """Basic asynchronous mode execution"""
    logger.info("⚡ Running basic async mode...")
    
    engine = BasicAsyncEngine(config, quantized_path)
    results = await engine.run_basic_async_inference()
    
    # Print performance analysis
    engine.print_performance_analysis()
    
    save_results(results, output_dir, 'async')
    logger.info("✅ Basic async mode completed")


async def run_all_modes_comparison(config, quantized_path, output_dir):
    """Execute all modes for comparison"""
    logger.info("🏁 Running all modes for comparison...")
    
    comparison_results = {}
    
    # 1. Sync mode
    logger.info("\n" + "="*50)
    logger.info("1️⃣  SYNC MODE")
    logger.info("="*50)
    
    engine_sync = InferenceEngine(config, quantized_path)
    sync_results = engine_sync.run_inference_sync()
    comparison_results['sync'] = sync_results
    
    # 2. Basic Async mode
    logger.info("\n" + "="*50)
    logger.info("2️⃣  BASIC ASYNC MODE")
    logger.info("="*50)
    
    engine_basic_async = BasicAsyncEngine(config, quantized_path)
    basic_async_results = await engine_basic_async.run_basic_async_inference()
    engine_basic_async.print_performance_analysis()
    comparison_results['async'] = basic_async_results
    
    # 3. Performance comparison
    print_performance_comparison(comparison_results)
    
    # Save all results
    comparison_path = os.path.join(output_dir, 'mode_comparison.yaml')
    with open(comparison_path, 'w') as f:
        yaml.dump(comparison_results, f)
    
    logger.info(f"✅ All modes completed. Results saved to {comparison_path}")


def print_performance_comparison(results):
    """Print performance comparison results"""
    print("\n" + "="*60)
    print("📊 PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    
    for mode, result in results.items():
        total_time = result.get('total_time', 0)
        num_results = len(result.get('results', []))
        
        print(f"\n{mode.upper()}:")
        print(f"  ⏱️  Total time: {total_time:.2f}s")
        print(f"  📦 Files processed: {num_results}")
        if total_time > 0:
            print(f"  🚀 Throughput: {num_results/total_time:.2f} files/sec")
        
        # Basic async specific metrics
        if mode == 'async' and 'metrics' in result:
            metrics = result['metrics']
            if 'speedup_factor' in metrics:
                print(f"  ⚡ Pipeline speedup: {metrics['speedup_factor']:.2f}x")
                print(f"  📈 Pipeline efficiency: {metrics['pipeline_efficiency']:.1%}")
    
    # Relative performance comparison
    if 'sync' in results and 'async' in results:
        sync_time = results['sync'].get('total_time', 0)
        basic_async_time = results['async'].get('total_time', 0)
        
        if sync_time > 0 and basic_async_time > 0:
            speedup = sync_time / basic_async_time
            print(f"\n🎯 Basic Async vs Sync speedup: {speedup:.2f}x")


def save_results(results, output_dir, mode_name):
    """Save inference results"""
    # Save detailed results
    results_path = os.path.join(output_dir, f'{mode_name}_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    
    # Save predictions as H5 files
    for result in results.get('results', []):
        if 'file_name' in result and 'prediction' in result:
            file_name = result['file_name']
            prediction = result['prediction']
            
            pred_path = os.path.join(output_dir, f"{mode_name}_{file_name.replace('.h5', '_pred.h5')}")
            
            import h5py
            with h5py.File(pred_path, 'w') as f:
                f.create_dataset('prediction', data=prediction, compression='gzip')
                if 'metrics' in result:
                    # Save metrics as well
                    for key, value in result['metrics'].items():
                        f.attrs[key] = value


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    main()