import onnx
# from onnx import optimizer
import subprocess
import json
import numpy as np
import os
from typing import Dict, Any, List, Tuple
from furiosa.optimizer import optimize_model
from pytorch3dunet.unet3d.utils import get_logger
logger = get_logger(__name__)


class ModelOptimizer:
    """
    Model optimization utilities using furiosa tools
    """
    
    def optimize_onnx(
        self, 
        input_path: str, 
        output_path: str,
        input_shape: List[int]
    ) -> str:
        """
        Optimize ONNX model for NPU deployment
        """
        # Load model
        model = onnx.load(input_path)
        
        # # Apply ONNX optimizations
        # optimized_model = optimizer.optimize(model, [
        #     'eliminate_identity',
        #     'eliminate_nop_transpose',
        #     'eliminate_nop_pad',
        #     'eliminate_unused_initializer',
        #     'eliminate_deadend',
        #     'fuse_consecutive_squeezes',
        #     'fuse_consecutive_transposes',
        #     'fuse_add_bias_into_conv',
        #     'fuse_transpose_into_gemm'
        # ])
        
        # Apply Furiosa-specific optimizations
        optimized_model = optimize_model(
            model=model,
            opset_version=13,
            input_shapes={"input": [1, 1] + input_shape}
        )
        
        # Save optimized model
        onnx.save(optimized_model, output_path)
        
        return output_path
    
    def run_furiosa_bench(
        self, 
        model_path: str,
        device: str,
        batch_size: int = 1,
        num_iterations: int = 1000
    ) -> Dict[str, Any]:
        """
        Run furiosa-bench and parse results
        """
        cmd = [
            "furiosa-bench",
            model_path,
            "--workload", "T",
            "-n", str(num_iterations),
            "-b", str(batch_size),
            "-w", "8",
            "-t", "8",
            "-d", device,
            "--output-format", "json"
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Parse JSON output
            metrics = json.loads(result.stdout)
            
            return {
                'throughput': metrics.get('throughput', 0),
                'latency_mean': metrics.get('latency_mean', 0),
                'latency_p99': metrics.get('latency_p99', 0),
                'npu_utilization': metrics.get('npu_utilization', 0)
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"furiosa-bench failed: {e.stderr}")
            return {}
        except json.JSONDecodeError:
            # Fallback to text parsing if JSON not available
            return self._parse_furiosa_bench_text(result.stdout)
    
    def _parse_furiosa_bench_text(self, output: str) -> Dict[str, Any]:
        """Parse text output from furiosa-bench"""
        metrics = {}
        
        for line in output.split('\n'):
            if 'Throughput' in line:
                # Extract throughput value
                parts = line.split(':')
                if len(parts) > 1:
                    value = parts[1].strip().split()[0]
                    metrics['throughput'] = float(value)
            elif 'Latency' in line and 'mean' in line:
                # Extract mean latency
                parts = line.split(':')
                if len(parts) > 1:
                    value = parts[1].strip().split()[0]
                    metrics['latency_mean'] = float(value)
        
        return metrics
    
    def find_best_compiler_settings(
        self,
        model_path: str,
        compiler_options: List[Dict[str, Any]],
        batch_size: int
    ) -> Dict[str, Any]:
        """
        Test different compiler settings and find the best
        """
        best_settings = None
        best_throughput = 0
        
        for option in compiler_options:
            logger.info(f"Testing compiler option: {option['option']}")
            
            # Run benchmark with specific compiler options
            env = os.environ.copy()
            if option['args']:
                env['FURIOSA_COMPILER_FLAGS'] = ' '.join(option['args'])
            
            metrics = self.run_furiosa_bench(
                model_path,
                "warboy(2)*1",
                batch_size
            )
            
            throughput = metrics.get('throughput', 0)
            if throughput > best_throughput:
                best_throughput = throughput
                best_settings = option
        
        return best_settings
    
    def analyze_performance(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Analyze performance metrics and provide optimization suggestions
        """
        suggestions = []
        
        # Check NPU utilization
        npu_util = metrics.get('npu_utilization', 0)
        if npu_util < 80:
            suggestions.append(
                f"Low NPU utilization ({npu_util}%). Consider:"
                f"\n  - Increasing batch size"
                f"\n  - Using more workers"
                f"\n  - Optimizing preprocessing pipeline"
            )
        
        # Check latency variation
        if 'latency_p99' in metrics and 'latency_mean' in metrics:
            variation = metrics['latency_p99'] / metrics['latency_mean']
            if variation > 1.5:
                suggestions.append(
                    "High latency variation detected. Consider:"
                    "\n  - Using fixed batch sizes"
                    "\n  - Enabling memory pinning"
                )
        
        # Check throughput
        patches_per_second = metrics.get('patches_per_second', 0)
        if patches_per_second < 100:  # Threshold depends on model
            suggestions.append(
                "Low throughput. Consider:"
                "\n  - Using larger patches"
                "\n  - Reducing model complexity"
                "\n  - Using INT8 quantization"
            )
        
        return suggestions
