import onnx
import numpy as np
from typing import List, Dict, Optional
from furiosa.quantizer import (
    CalibrationMethod, Calibrator, quantize,
    ModelEditor, TensorType
)
from tqdm import tqdm
from .loader import H5Loader
from .preprocessor import Preprocessor
import logging

logger = logging.getLogger(__name__)


class Quantizer:
    """
    Quantize ONNX model for INT8 NPU deployment
    """
    
    def __init__(self, config: Dict):
        """
        Initialize quantizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.loader = H5Loader()
        self.preprocessor = Preprocessor(config)
        
    def quantize_model(
        self, 
        onnx_path: str,
        output_path: str,
        calibration_samples: int = 100,
        calibration_method: str = 'MIN_MAX_ASYM'
    ) -> str:
        """
        Quantize ONNX model using validation dataset
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save quantized model
            calibration_samples: Number of calibration samples
            calibration_method: Calibration method name
            
        Returns:
            Path to quantized model
        """
        # Load calibration data
        calibration_data = self._load_calibration_data(calibration_samples)
        
        # Load ONNX model
        model = onnx.load(onnx_path)
        
        # IMPORTANT: Convert input type to UINT8 before calibration
        editor = ModelEditor(model)
        editor.convert_input_type("input", TensorType.UINT8)
        
        # Create calibrator
        method = getattr(CalibrationMethod, calibration_method)
        calibrator = Calibrator(model, method)
        
        # Collect calibration data
        logger.info(f"Collecting calibration data with {len(calibration_data)} samples")
        for data in tqdm(calibration_data, desc="Calibrator"):
            calibrator.collect_data([[data]])
            
        # Compute calibration ranges
        calibration_range = calibrator.compute_range()
        
        # Quantize model
        quantized_model = quantize(model, calibration_range)
        
        # Save quantized model
        with open(output_path, "wb") as f:
            f.write(bytes(quantized_model))
            
        logger.info(f"Quantized model saved to: {output_path}")
        
        return output_path
    
    def _load_calibration_data(self, num_samples: int) -> List[np.ndarray]:
        """
        Load calibration data from validation dataset
        """
        # Load validation data from config
        data_dict = self.loader.load_from_config(self.config)
        
        if 'valid' not in data_dict:
            raise ValueError("No validation data found in config")
            
        calibration_patches = []
        
        for data_info in data_dict['valid']:
            volume = data_info['data']
            
            # Process volume into patches
            patches = self.preprocessor.process_volume(volume)
            
            for patch_info in patches:
                patch = patch_info['patch']
                
                # IMPORTANT: For calibration, we need float32 data
                # The preprocessor returns data after transforms
                if patch.dtype != np.float32:
                    patch = patch.astype(np.float32)
                    
                # Normalize if needed
                if np.max(patch) > 1.0:
                    patch = patch / 255.0
                    
                # Add batch and channel dimensions if needed
                if patch.ndim == 3:
                    patch = np.expand_dims(patch, 0)
                if patch.ndim == 4:
                    patch = np.expand_dims(patch, 0)
                    
                calibration_patches.append(patch)
                
                if len(calibration_patches) >= num_samples:
                    return calibration_patches[:num_samples]
                    
        logger.info(f"Loaded {len(calibration_patches)} calibration patches")
        
        return calibration_patches