import asyncio
import numpy as np
import time
import gc
import psutil
import os
from typing import Dict, List, Optional, Tuple
from furiosa.runtime import create_runner
import logging

logger = logging.getLogger(__name__)


class BasicAsyncEngine:
    """
    Basic Async Engine - Memory-efficient streaming implementation
    """
    
    def __init__(self, config: Dict, quantized_model_path: str):
        self.config = config
        self.model_path = quantized_model_path
        
        # Basic configuration
        self.device = config.get('device', 'warboy(2)*1')
        self.batch_size = config.get('loaders', {}).get('batch_size', 1)
        self.num_workers = config.get('loaders', {}).get('num_workers', 4)
        
        # Memory settings
        self.chunk_size = config.get('chunk_size', 5)  # Process 5 patches at a time
        
        # Memory monitoring
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Performance metrics
        self.metrics = {
            'prep_times': [],
            'npu_times': [],
            'memory_usage': []
        }
        
        # Initialize components
        from .loader import H5Loader
        from .preprocessor import Preprocessor
        from .postprocessor import Postprocessor
        
        self.loader = H5Loader()
        self.preprocessor = Preprocessor(config)
        self.postprocessor = Postprocessor(config)
        
        logger.info(f"🔧 Initial memory usage: {self.initial_memory:.1f} MB")
        logger.info(f"📦 Chunk size: {self.chunk_size} patches")
    
    def _log_memory_usage(self, stage: str):
        """Log current memory usage"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - self.initial_memory
        self.metrics['memory_usage'].append(current_memory)
        
        logger.debug(f"💾 {stage}: {current_memory:.1f} MB (+{memory_increase:.1f} MB)")
        
        # Warning if memory usage is too high
        if current_memory > 8000:  # 8GB threshold
            logger.warning(f"⚠️  High memory usage detected: {current_memory:.1f} MB")
        
        return current_memory
    
    async def run_basic_async_inference(self) -> Dict:
        """
        Run basic async inference with streaming processing
        """
        self._log_memory_usage("Start")
        
        # Load test data
        data_dict = self.loader.load_from_config(self.config)
        self._log_memory_usage("After data loading")
        
        if 'test' not in data_dict:
            raise ValueError("No test data found in config")
        
        results = []
        overall_start = time.time()
        
        async with create_runner(
            self.model_path,
            device=self.device,
            worker_num=self.num_workers
        ) as runner:
            
            self._log_memory_usage("After runner creation")
            
            for i, data_info in enumerate(data_dict['test']):
                logger.info(f"Processing file {i+1}/{len(data_dict['test'])}: {data_info['file_name']}")
                
                # Execute streaming async pipeline
                result = await self._process_volume_streaming(runner, data_info)
                results.append(result)
                
                # Force garbage collection after each volume
                gc.collect()
                self._log_memory_usage(f"After processing {data_info['file_name']}")
        
        overall_time = time.time() - overall_start
        
        return {
            'results': results,
            'mode': 'basic_async_streaming',
            'device': self.device,
            'total_time': overall_time,
            'metrics': self._calculate_summary_metrics()
        }
    
    async def _process_volume_streaming(self, runner, data_info: Dict) -> Dict:
        """
        Process single volume using streaming approach
        """
        volume = data_info['data']
        label = data_info.get('label')
        file_name = data_info['file_name']
        
        start_time = time.time()
        
        # Get patch count without creating patches
        total_patches = self.preprocessor.get_patch_count(volume)
        volume_size = volume.nbytes / 1024 / 1024  # MB
        
        logger.info(f"📦 Volume: {volume.shape} ({volume_size:.1f} MB)")
        logger.info(f"📦 Will process {total_patches} patches in streaming mode")
        
        # Use streaming processor
        all_predictions = []
        processed_count = 0
        
        # Process volume in chunks using streaming
        async for chunk_predictions in self._streaming_pipeline(runner, volume, total_patches):
            all_predictions.extend(chunk_predictions)
            processed_count += len(chunk_predictions)
            
            # Log progress
            progress = (processed_count / total_patches) * 100
            logger.info(f"🚀 Progress: {processed_count}/{total_patches} ({progress:.1f}%)")
            
            # Memory cleanup
            gc.collect()
            self._log_memory_usage(f"After chunk {processed_count//self.chunk_size}")
        
        # Postprocess all predictions
        logger.info("🎯 Starting postprocessing...")
        result = self.postprocessor.process_predictions(
            all_predictions,
            volume.shape,
            label
        )
        
        total_time = time.time() - start_time
        result['file_name'] = file_name
        result['processing_time'] = total_time
        
        logger.info(f"✅ {file_name} completed in {total_time:.2f}s")
        return result
    
    async def _streaming_pipeline(self, runner, volume: np.ndarray, total_patches: int):
        """
        Streaming async pipeline that processes chunks of patches
        """
        # Create streaming processor
        chunk_generator = self.preprocessor.process_volume_streaming(
            volume, 
            chunk_size=self.chunk_size
        )
        
        # Pipeline queues (very small)
        prep_queue = asyncio.Queue(maxsize=2)
        npu_queue = asyncio.Queue(maxsize=2)
        
        async def chunk_preprocessing_worker():
            """Process chunks of patches"""
            try:
                chunk_idx = 0
                for patch_chunk in chunk_generator:
                    start_time = time.time()
                    
                    # Process each patch in the chunk
                    processed_patches = []
                    for patch_info in patch_chunk:
                        # Use existing prepare_batch method
                        patch_batch = self.preprocessor.prepare_batch(
                            [patch_info['patch']], 
                            for_inference=True
                        )
                        
                        processed_patches.append({
                            'patch_idx': patch_info['patch_idx'],
                            'slice_indices': patch_info['slice_indices'],
                            'preprocessed': patch_batch[0],  # Extract from batch
                            'original_patch_info': patch_info
                        })
                    
                    prep_time = time.time() - start_time
                    self.metrics['prep_times'].append(prep_time)
                    
                    await prep_queue.put({
                        'chunk_idx': chunk_idx,
                        'patches': processed_patches,
                        'prep_time': prep_time
                    })
                    
                    chunk_idx += 1
                    logger.debug(f"🔧 Preprocessed chunk {chunk_idx} ({len(processed_patches)} patches)")
                
                await prep_queue.put(None)  # End signal
                logger.info("✅ Chunk preprocessing completed")
                
            except Exception as e:
                logger.error(f"❌ Chunk preprocessing error: {e}")
                await prep_queue.put(None)
                raise
        
        async def npu_worker():
            """Process chunks on NPU"""
            try:
                while True:
                    chunk_data = await prep_queue.get()
                    if chunk_data is None:
                        await npu_queue.put(None)
                        break
                    
                    # Process patches in the chunk as batches
                    chunk_predictions = []
                    patches = chunk_data['patches']
                    
                    # Process in mini-batches
                    for i in range(0, len(patches), self.batch_size):
                        batch_patches = patches[i:i + self.batch_size]
                        
                        # Prepare NPU batch
                        batch_data = np.stack([p['preprocessed'] for p in batch_patches])
                        
                        # NPU inference
                        start_time = time.time()
                        batch_outputs = await runner.run([batch_data])
                        npu_time = time.time() - start_time
                        
                        self.metrics['npu_times'].append(npu_time)
                        
                        # Collect predictions
                        for j, patch in enumerate(batch_patches):
                            prediction = batch_outputs[0][j] if len(batch_outputs[0]) > j else batch_outputs[0][0]
                            
                            chunk_predictions.append({
                                'patch_idx': patch['patch_idx'],
                                'prediction': prediction.copy(),
                                'patch_info': patch['original_patch_info'],
                                'npu_time': npu_time / len(batch_patches)
                            })
                        
                        # Cleanup batch data immediately
                        del batch_data
                        del batch_outputs
                    
                    await npu_queue.put({
                        'chunk_idx': chunk_data['chunk_idx'],
                        'predictions': chunk_predictions
                    })
                    
                    logger.debug(f"🚀 NPU processed chunk {chunk_data['chunk_idx']}")
                
                logger.info("✅ NPU processing completed")
                
            except Exception as e:
                logger.error(f"❌ NPU worker error: {e}")
                await npu_queue.put(None)
                raise
        
        async def result_collector():
            """Collect and yield results"""
            try:
                while True:
                    chunk_result = await npu_queue.get()
                    if chunk_result is None:
                        break
                    
                    # Yield predictions for this chunk
                    yield chunk_result['predictions']
                    
                    logger.debug(f"🎯 Collected chunk {chunk_result['chunk_idx']}")
                
                logger.info("✅ Result collection completed")
                
            except Exception as e:
                logger.error(f"❌ Result collector error: {e}")
                raise
        
        # Create async generator for results
        result_gen = result_collector()
        
        # Start workers
        workers = [
            asyncio.create_task(chunk_preprocessing_worker()),
            asyncio.create_task(npu_worker()),
        ]
        
        # Yield results as they become available
        try:
            async for chunk_predictions in result_gen:
                yield chunk_predictions
        finally:
            # Cleanup workers
            for worker in workers:
                if not worker.done():
                    worker.cancel()
            
            await asyncio.gather(*workers, return_exceptions=True)
    
    def _calculate_summary_metrics(self) -> Dict:
        """Calculate performance metrics summary"""
        if not self.metrics['prep_times'] or not self.metrics['npu_times']:
            return {}
        
        summary = {
            'avg_prep_time': np.mean(self.metrics['prep_times']),
            'avg_npu_time': np.mean(self.metrics['npu_times']),
            'total_prep_time': sum(self.metrics['prep_times']),
            'total_npu_time': sum(self.metrics['npu_times']),
            'num_chunks': len(self.metrics['prep_times']),
        }
        
        # Memory metrics
        if self.metrics['memory_usage']:
            summary['peak_memory_mb'] = max(self.metrics['memory_usage'])
            summary['memory_increase_mb'] = max(self.metrics['memory_usage']) - self.initial_memory
        
        return summary
    
    def print_performance_analysis(self):
        """Print performance analysis results"""
        metrics = self._calculate_summary_metrics()
        
        if not metrics:
            print("❌ No performance metrics available")
            return
        
        print(f"\n📊 Streaming Async Pipeline Performance Analysis:")
        print(f"  📦 Chunks processed: {metrics['num_chunks']}")
        print(f"  ⏱️  Average prep time: {metrics['avg_prep_time']:.3f}s")
        print(f"  🚀 Average NPU time: {metrics['avg_npu_time']:.3f}s")
        
        if 'peak_memory_mb' in metrics:
            print(f"  💾 Peak memory: {metrics['peak_memory_mb']:.1f} MB (+{metrics['memory_increase_mb']:.1f} MB)")
        
        print(f"  ✅ Memory-efficient streaming processing completed")