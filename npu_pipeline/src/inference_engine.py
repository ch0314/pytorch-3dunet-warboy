import numpy as np
import time
import asyncio
from typing import Dict, Any, List
from furiosa.runtime import create_runner
from furiosa.runtime.sync import create_runner as create_runner_sync
from pytorch3dunet.unet3d.utils import get_logger
logger = get_logger(__name__)


class InferenceEngine:
    """
    Optimized inference engine with patch-based processing
    """
    
    def __init__(
        self, 
        model_path: str, 
        device: str, 
        num_workers: int,
        patch_config: Dict[str, Any]
    ):
        self.model_path = model_path
        self.device = device
        self.num_workers = num_workers
        self.patch_config = patch_config
        self.batch_size = patch_config['batch_size']
        
    async def infer_batch_async(self, batch_data: np.ndarray) -> List[np.ndarray]:
        """
        Asynchronous batch inference
        """
        async with create_runner(
            self.model_path,
            device=self.device,
            worker_num=self.num_workers
        ) as runner:
            # Process batch
            outputs = await runner.run(batch_data)
            return outputs
    
    def infer_batch(self, batch_data: np.ndarray) -> List[np.ndarray]:
        """
        Synchronous batch inference
        """
        with create_runner_sync(
            self.model_path,
            device=self.device,
            worker_num=self.num_workers
        ) as runner:
            outputs = runner.run(batch_data)
            return outputs
    
    def benchmark_patch_based(self, num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark with realistic patch-based workload
        """
        # Create dummy patches
        patch_shape = self.patch_config['patch_shape']
        halo_shape = self.patch_config['halo_shape']
        input_shape = [
            patch_shape[0] + 2 * halo_shape[0],  # 80 + 2*16 = 112
            patch_shape[1] + 2 * halo_shape[1],  # 170 + 2*32 = 234
            patch_shape[2] + 2 * halo_shape[2]   # 170 + 2*32 = 234
        ]

        dummy_batch = np.random.randn(
            self.batch_size, 1, *input_shape
        ).astype(np.uint8)
        
        # Warmup
        with create_runner_sync(
            self.model_path,
            device=self.device,
            worker_num=self.num_workers
        ) as runner:
            for _ in range(10):
                _ = runner.run(dummy_batch)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                _ = runner.run(dummy_batch)
            end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_latency_per_batch = total_time / num_iterations
        avg_latency_per_patch = avg_latency_per_batch / self.batch_size
        patches_per_second = (num_iterations * self.batch_size) / total_time
        
        return {
            'total_time': total_time,
            'avg_latency_per_batch': avg_latency_per_batch,
            'avg_latency_per_patch': avg_latency_per_patch,
            'patches_per_second': patches_per_second,
            'batch_size': self.batch_size,
            'num_iterations': num_iterations
        }