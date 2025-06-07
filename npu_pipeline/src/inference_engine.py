import numpy as np
from typing import Dict, List, Optional, Union
from furiosa.runtime import create_runner
from furiosa.runtime.sync import create_runner as create_runner_sync
import asyncio
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
        with create_runner_sync(
            self.model_path,
            device=self.device,
            worker_num=self.num_workers
        ) as runner:
            
            for data_info in data_dict['test']:
                result = self._process_volume_sync(runner, data_info)
                results.append(result)
                
        return {
            'results': results,
            'mode': 'sync',
            'device': self.device
        }
    
    async def run_inference_async(self) -> Dict:
        """
        Run asynchronous inference on test data
        
        Returns:
            Dictionary of results
        """
        # Load test data
        data_dict = self.loader.load_from_config(self.config)
        
        if 'test' not in data_dict:
            raise ValueError("No test data found in config")
            
        results = []
        
        # Create async runner
        async with create_runner(
            self.model_path,
            device=self.device,
            worker_num=self.num_workers
        ) as runner:
            
            # Process volumes concurrently
            tasks = []
            for data_info in data_dict['test']:
                task = self._process_volume_async(runner, data_info)
                tasks.append(task)
                
            results = await asyncio.gather(*tasks)
            
        return {
            'results': results,
            'mode': 'async',
            'device': self.device
        }
    
    def _process_volume_sync(self, runner, data_info: Dict) -> Dict:
        """
        Process single volume synchronously
        """
        volume = data_info['data']
        label = data_info.get('label')
        file_name = data_info['file_name']
        
        logger.info(f"Processing {file_name}")
        
        # Preprocess volume into patches
        patches_info = self.preprocessor.process_volume(volume)
        
        # Run inference on patches
        predictions = []
        
        for i in range(0, len(patches_info), self.batch_size):
            # Prepare batch
            batch_patches = patches_info[i:i + self.batch_size]
            batch_data = self.preprocessor.prepare_batch(
                [p['patch'] for p in batch_patches],
                for_inference=True  # This ensures UINT8 conversion
            )
            
            # Verify data type
            if batch_data.dtype != np.uint8:
                logger.warning(f"Converting batch data from {batch_data.dtype} to uint8")
                batch_data = batch_data.astype(np.uint8)
            
            # Run inference
            batch_predictions = runner.run(batch_data)
            
            # Store predictions with metadata
            for j, pred in enumerate(batch_predictions):
                predictions.append({
                    'prediction': pred,
                    'patch_info': batch_patches[j]
                })
                
        # Postprocess predictions
        result = self.postprocessor.process_predictions(
            predictions,
            volume.shape,
            label
        )
        
        result['file_name'] = file_name
        
        return result
    
    async def _process_volume_async(self, runner, data_info: Dict) -> Dict:
        """
        Process single volume asynchronously
        """
        volume = data_info['data']
        label = data_info.get('label')
        file_name = data_info['file_name']
        
        logger.info(f"Processing {file_name}")
        
        # Preprocess volume into patches
        patches_info = self.preprocessor.process_volume(volume)
        
        # Run inference on patches
        predictions = []
        
        # Process batches concurrently
        batch_tasks = []
        
        for i in range(0, len(patches_info), self.batch_size):
            # Prepare batch
            batch_patches = patches_info[i:i + self.batch_size]
            batch_data = self.preprocessor.prepare_batch(
                [p['patch'] for p in batch_patches],
                for_inference=True  # This ensures UINT8 conversion
            )
            
            # Verify data type
            if batch_data.dtype != np.uint8:
                logger.warning(f"Converting batch data from {batch_data.dtype} to uint8")
                batch_data = batch_data.astype(np.uint8)
            
            # Create async task
            task = runner.run(batch_data)
            batch_tasks.append((task, batch_patches))
            
        # Wait for all batches
        for task, batch_patches in batch_tasks:
            batch_predictions = await task
            
            # Store predictions with metadata
            for j, pred in enumerate(batch_predictions):
                predictions.append({
                    'prediction': pred,
                    'patch_info': batch_patches[j]
                })
                
        # Postprocess predictions
        result = self.postprocessor.process_predictions(
            predictions,
            volume.shape,
            label
        )
        
        result['file_name'] = file_name
        
        return result