import subprocess
import json
import os
from typing import Dict, List, Optional, Tuple
from .model_exporter import ModelExporter
from .quantizer import Quantizer
import logging

logger = logging.getLogger(__name__)


class Benchmark:
    """
    Automated benchmarking with furiosa-bench
    """
    
    def __init__(self, config: Dict):
        """
        Initialize benchmark
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_exporter = ModelExporter(config)
        self.quantizer = Quantizer(config)
        
    def run_comprehensive_benchmark(
        self,
        base_onnx_path: str,
        batch_sizes: List[int] = [1, 2, 4, 8],
        worker_nums: List[int] = [1, 2, 4, 8],
        device: str = "warboy(2)*1",
        num_iterations: int = 1000
    ) -> Dict[str, Dict]:
        """
        Run comprehensive benchmark with different configurations
        
        Args:
            base_onnx_path: Base ONNX model path
            batch_sizes: List of batch sizes to test
            worker_nums: List of worker numbers to test
            device: NPU device specification
            num_iterations: Number of iterations per test
            
        Returns:
            Dictionary of benchmark results
        """
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Re-export and quantize for different batch size
            onnx_path = base_onnx_path.replace('.onnx', f'_b{batch_size}.onnx')
            quantized_path = self._prepare_model_for_batch(base_onnx_path, onnx_path, batch_size)
            
            results[f'batch_{batch_size}'] = {}
            
            for worker_num in worker_nums:
                logger.info(f"  Testing {worker_num} workers")
                
                # Run benchmark
                metrics = self._run_furiosa_bench(
                    quantized_path,
                    batch_size,
                    worker_num,
                    device,
                    num_iterations
                )
                
                results[f'batch_{batch_size}'][f'workers_{worker_num}'] = metrics
                
        return results
    
    def _prepare_model_for_batch(self, base_path: str, output_path: str, batch_size: int) -> str:
        """
        Prepare model for specific batch size
        """
        # For batch size 1, use existing model
        if batch_size == 1 and os.path.exists(base_path.replace('.onnx', '_i8.onnx')):
            return base_path.replace('.onnx', '_i8.onnx')
            
        # TODO: Re-export with different batch size if needed
        # For now, assume dynamic batch size
        quantized_path = output_path.replace('.onnx', '_i8.onnx')
        
        if not os.path.exists(quantized_path):
            # Quantize the model
            self.quantizer.quantize_model(base_path, quantized_path)
            
        return quantized_path
    
    def _run_furiosa_bench(
        self,
        model_path: str,
        batch_size: int,
        worker_num: int,
        device: str,
        num_iterations: int
    ) -> Dict:
        """
        Run furiosa-bench and parse results
        """
        cmd = [
            "furiosa-bench",
            model_path,
            "--workload", "T",  # Throughput mode
            "-n", str(num_iterations),
            "-b", str(batch_size),
            "-w", str(worker_num),
            "-t", str(worker_num),  # Number of threads
            "-d", device
        ]
        
        try:
            # Run benchmark
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse output
            metrics = self._parse_bench_output(result.stdout)
            
            logger.info(f"    Throughput: {metrics.get('throughput', 'N/A')} inferences/sec")
            logger.info(f"    Latency (mean): {metrics.get('latency_mean', 'N/A')} ms")
            
            return metrics
            
        except subprocess.CalledProcessError as e:
            logger.error(f"furiosa-bench failed: {e.stderr}")
            return {"error": str(e)}
            
    def _parse_bench_output(self, output: str) -> Dict:
        """
        Parse furiosa-bench output
        """
        metrics = {}
        
        lines = output.strip().split('\n')
        
        for line in lines:
            # Parse throughput
            if 'Throughput' in line and 'inferences/sec' in line:
                try:
                    value = float(line.split(':')[1].strip().split()[0])
                    metrics['throughput'] = value
                except:
                    pass
                    
            # Parse latency
            elif 'Latency' in line:
                if 'mean' in line:
                    try:
                        value = float(line.split(':')[1].strip().split()[0])
                        metrics['latency_mean'] = value
                    except:
                        pass
                elif 'p99' in line:
                    try:
                        value = float(line.split(':')[1].strip().split()[0])
                        metrics['latency_p99'] = value
                    except:
                        pass
                        
        return metrics
    
    def find_optimal_configuration(self, results: Dict) -> Tuple[int, int]:
        """
        Find optimal batch size and worker configuration
        
        Args:
            results: Benchmark results dictionary
            
        Returns:
            Tuple of (optimal_batch_size, optimal_workers)
        """
        best_throughput = 0
        best_config = (1, 1)
        
        for batch_key, batch_results in results.items():
            batch_size = int(batch_key.split('_')[1])
            
            for worker_key, metrics in batch_results.items():
                if 'error' not in metrics:
                    throughput = metrics.get('throughput', 0)
                    
                    if throughput > best_throughput:
                        best_throughput = throughput
                        workers = int(worker_key.split('_')[1])
                        best_config = (batch_size, workers)
                        
        logger.info(f"Optimal configuration: batch_size={best_config[0]}, workers={best_config[1]}")
        logger.info(f"Best throughput: {best_throughput} inferences/sec")
        
        return best_config