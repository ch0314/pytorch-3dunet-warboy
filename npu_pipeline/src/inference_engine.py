import numpy as np
from typing import Dict, List, Optional, Union
from furiosa.runtime import create_runner
from furiosa.runtime.sync import create_runner as create_runner_sync
import asyncio
import time
from .loader import H5Loader
from .preprocessor import Preprocessor
from .postprocessor import Postprocessor
import logging

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    NPU inference engine with sync/async support
    """
    
    def __init__(self, config: Dict, quantized_model_path: str):
        """
        Initialize inference engine
        
        Args:
            config: Configuration dictionary
            quantized_model_path: Path to quantized ONNX model
        """
        self.config = config
        self.model_path = quantized_model_path
        
        # Initialize components
        self.loader = H5Loader()
        self.preprocessor = Preprocessor(config)
        self.postprocessor = Postprocessor(config)
        
        # Extract device config
        self.device = config.get('device', 'warboy(2)*1')
        self.batch_size = config.get('loaders', {}).get('batch_size', 1)
        self.num_workers = config.get('loaders', {}).get('num_workers', 8)
        
        logger.info(f"InferenceEngine initialized - device: {self.device}, batch_size: {self.batch_size}, workers: {self.num_workers}")
        
    def run_inference_sync(self) -> Dict:
        """
        Run synchronous inference on test data
        
        Returns:
            Dictionary of results
        """
        # Load test data
        data_dict = self.loader.load_from_config(self.config)
        
        if 'test' not in data_dict:
            raise ValueError("No test data found in config")
            
        results = []
        
        # Create sync runner
        logger.info("Creating sync runner...")
        with create_runner_sync(
            self.model_path,
            device=self.device,
            worker_num=self.num_workers
        ) as runner:
            logger.info("Runner created successfully")
            
            for i, data_info in enumerate(data_dict['test']):
                logger.info(f"Processing file {i+1}/{len(data_dict['test'])}: {data_info['file_name']}")
                result = self._process_volume_sync(runner, data_info)
                results.append(result)
                
        return {
            'results': results,
            'mode': 'sync',
            'device': self.device
        }
    
    def _process_volume_sync(self, runner, data_info: Dict) -> Dict:
        """
        Process single volume synchronously with detailed timing
        """
        volume = data_info['data']
        label = data_info.get('label')
        file_name = data_info['file_name']
        
        logger.info(f"Processing {file_name} - shape: {volume.shape}")
        
        try:
            # Time preprocessing
            preprocess_start = time.time()
            patches_info = self.preprocessor.process_volume(volume)
            preprocess_time = time.time() - preprocess_start
            logger.info(f"Preprocessing took {preprocess_time:.2f}s for {len(patches_info)} patches")
            
            # Run inference on patches
            predictions = []
            inference_start = time.time()
            
            # Process in smaller batches if too many patches
            total_patches = len(patches_info)
            processed_patches = 0
            
            for i in range(0, total_patches, self.batch_size):
                # Log progress every 10 batches
                if processed_patches % (10 * self.batch_size) == 0:
                    logger.info(f"Processing patches {processed_patches}/{total_patches}")
                
                # Prepare batch
                batch_start = time.time()
                batch_patches = patches_info[i:i + self.batch_size]
                batch_data = self.preprocessor.prepare_batch(
                    [p['patch'] for p in batch_patches],
                    for_inference=True
                )
                
                # Verify data type and shape
                if batch_data.dtype != np.uint8:
                    logger.warning(f"Converting batch data from {batch_data.dtype} to uint8")
                    batch_data = batch_data.astype(np.uint8)
                
                logger.debug(f"Batch shape: {batch_data.shape}, dtype: {batch_data.dtype}")
                
                # Run inference
                npu_start = time.time()
                batch_predictions = runner.run(batch_data)
                npu_time = time.time() - npu_start
                
                # Log timing for first few batches
                if i < 3 * self.batch_size:
                    batch_time = time.time() - batch_start
                    logger.info(f"Batch {i//self.batch_size}: prep={batch_time-npu_time:.3f}s, npu={npu_time:.3f}s")
                
                # Store predictions with metadata
                for j, pred in enumerate(batch_predictions):
                    predictions.append({
                        'prediction': pred,
                        'patch_info': batch_patches[j]
                    })
                
                processed_patches += len(batch_patches)
                    
            inference_time = time.time() - inference_start
            logger.info(f"Inference took {inference_time:.2f}s for {total_patches} patches")
            logger.info(f"Average time per patch: {inference_time/total_patches:.3f}s")
            
            # Postprocess predictions
            postprocess_start = time.time()
            result = self.postprocessor.process_predictions(
                predictions,
                volume.shape,
                label
            )
            postprocess_time = time.time() - postprocess_start
            logger.info(f"Postprocessing took {postprocess_time:.2f}s")
            
            result['file_name'] = file_name
            result['status'] = 'success'
            result['timing'] = {
                'preprocessing': preprocess_time,
                'inference': inference_time,
                'postprocessing': postprocess_time,
                'total': preprocess_time + inference_time + postprocess_time
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}", exc_info=True)
            result = {
                'file_name': file_name,
                'status': 'error',
                'error': str(e)
            }
        
        return result