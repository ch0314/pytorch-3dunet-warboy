import argparse
import asyncio
import sys
import os
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from npu_pipeline.src.pipeline import UNet3DPipeline


async def main():
    parser = argparse.ArgumentParser(
        description='Run NPU Pipeline with pytorch-3dunet config format'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to test_config.yml')
    parser.add_argument('--device', type=str, default='warboy(2)*1',
                        help='NPU device specification')
    parser.add_argument('--calibration-samples', type=int, default=5,
                        help='Number of calibration samples')
    parser.add_argument('--export', action='store_true',
                        help='ONNX export')
    parser.add_argument('--quantize', action='store_true',
                        help='ONNX quantization')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark after quantization')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print(f"Loading configuration from: {args.config}")
    pipeline = UNet3DPipeline(args.config)
    
    # Print configuration summary
    print("\nConfiguration Summary:")
    print(f"  Model: {pipeline.model_config.get('name', 'UNet3D')}")
    print(f"  Input channels: {pipeline.model_config.get('in_channels', 1)}")
    print(f"  Output channels: {pipeline.model_config.get('out_channels', 1)}")
    print(f"  Patch shape: {pipeline.patch_config['patch_shape']}")
    print(f"  Stride shape: {pipeline.patch_config['stride_shape']}")
    print(f"  Halo shape: {pipeline.patch_config['halo_shape']}")
    print(f"  Output directory: {pipeline.output_dir}")
    
    # Step 1: Export to ONNX
    if args.export:
        print("\n=== Step 1: Exporting to ONNX ===")
        onnx_path = pipeline.export_and_optimize_onnx()
        print(f"Exported to: {onnx_path}")
    else:
        # Assume ONNX already exists
        pipeline.optimized_onnx_path = os.path.join(
            pipeline.output_dir, "unet3d_optimized.onnx"
        )
    
    # Step 2: Quantize
    if args.quantize:
        print("\n=== Step 2: Quantizing model ===")
        
        quantized_path = pipeline.quantize_model(
            calibration_samples=args.calibration_samples
        )
        print(f"Quantized model saved to: {quantized_path}")
    else:
        # Assume quantized model exists
        pipeline.quantized_onnx_path = os.path.join(
            pipeline.output_dir, "unet3d_i8.onnx"
        )
    
    # Step 3: Benchmark (optional)
    if args.benchmark:
        print("\n=== Step 3: Running benchmark ===")
        metrics = pipeline.benchmark_optimized(
            device=args.device,
            run_furiosa_bench=True
        )
        
        print("\nBenchmark Results:")
        print(f"  Patches per second: {metrics['patches_per_second']:.2f}")
        print(f"  Average latency per patch: {metrics['avg_latency_per_patch']*1000:.2f} ms")
        print(f"  Batch size: {metrics['batch_size']}")
        
        if 'furiosa_bench' in metrics:
            print("\nFuriosa-bench metrics:")
            for key, value in metrics['furiosa_bench'].items():
                print(f"  {key}: {value}")
    
    # Step 4: Run inference
    print("\n=== Step 4: Running inference ===")
    results = await pipeline.run_inference_async(
        device=args.device,
        save_predictions=True
    )
    
    print(f"\nInference completed!")
    print(f"  Total files: {results['total_files']}")
    print(f"  Output directory: {results['output_dir']}")
    
    # Print metrics if available
    successful_results = [r for r in results['results'] if 'metrics' in r]
    if successful_results:
        dice_scores = [r['metrics']['dice_score'] for r in successful_results]
        avg_dice = sum(dice_scores) / len(dice_scores)
        print(f"  Average Dice score: {avg_dice:.4f}")
    
    # Save summary
    summary_path = os.path.join(pipeline.output_dir, "inference_summary.yaml")
    with open(summary_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    asyncio.run(main())