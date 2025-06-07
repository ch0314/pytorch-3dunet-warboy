import torch
import onnx
import numpy as np
from typing import Dict, Tuple, Optional
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.utils import load_checkpoint
from furiosa.optimizer import optimize_model
import logging

logger = logging.getLogger(__name__)


class ModelExporter:
    """
    Export PyTorch model to optimized ONNX format for NPU
    """
    
    def __init__(self, config: Dict):
        """
        Initialize model exporter
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.checkpoint_path = config.get('model_path')
        
        # Extract patch configuration
        test_config = config.get('loaders', {}).get('test', {})
        slice_config = test_config.get('slice_builder', {})
        
        self.patch_shape = slice_config.get('patch_shape', [80, 170, 170])
        self.halo_shape = slice_config.get('halo_shape', [0, 0, 0])
        self.batch_size = config.get('loaders', {}).get('batch_size', 1)
        
        # Calculate actual input shape with halo
        self.input_shape = self._calculate_input_shape()
        
    def _calculate_input_shape(self) -> Tuple[int, ...]:
        """
        Calculate input shape considering patch and halo
        """
        input_shape = []
        for p, h in zip(self.patch_shape, self.halo_shape):
            input_shape.append(p + 2 * h)
            
        return tuple(input_shape)
    
    def export_to_onnx(self, output_path: str) -> str:
        """
        Export model to ONNX format
        
        Args:
            output_path: Path to save ONNX model
            
        Returns:
            Path to exported ONNX file
        """
        # Load PyTorch model
        model = self._load_pytorch_model()
        
        # Create dummy input
        in_channels = self.model_config.get('in_channels', 1)
        batch_size = self.batch_size
        dummy_input = torch.randn(batch_size, in_channels, *self.input_shape)
        
        logger.info(f"Exporting model with input shape: {dummy_input.shape}")
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model exported to: {output_path}")
        
        # Verify ONNX model
        if self._verify_onnx(output_path):
            # Optimize for NPU
            optimized_path = self._optimize_for_npu(output_path)
            return optimized_path
        else:
            raise RuntimeError("ONNX verification failed")
            
    def _load_pytorch_model(self) -> torch.nn.Module:
        """
        Load PyTorch model from checkpoint
        """
        # Create model
        model = get_model(self.model_config)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.eval()
        
        logger.info(f"Loaded model from: {self.checkpoint_path}")
        
        return model
    
    def _verify_onnx(self, model_path: str) -> bool:
        """
        Verify ONNX model
        """
        try:
            # Check model
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            
            # Run shape inference
            from onnx import shape_inference
            onnx_model = shape_inference.infer_shapes(onnx_model)
            
            logger.info("ONNX model verification passed")
            return True
            
        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")
            return False
            
    def _optimize_for_npu(self, onnx_path: str) -> str:
        """
        Optimize ONNX model for NPU deployment
        """
        # Load model
        model = onnx.load(onnx_path)
        
        # Create optimized path
        optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
        
        # Apply Furiosa optimizations
        in_channels = self.model_config.get('in_channels', 1)
        optimized_model = optimize_model(
            model=model,
            opset_version=13,
            input_shapes={"input": [self.batch_size, in_channels] + list(self.input_shape)}
        )
        
        # Save optimized model
        onnx.save(optimized_model, optimized_path)
        
        logger.info(f"Optimized model saved to: {optimized_path}")
        
        return optimized_path