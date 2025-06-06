import torch
import yaml
import numpy as np
import os
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.utils import load_checkpoint
import onnx
import onnxruntime
from pytorch3dunet.unet3d.utils import get_logger
from furiosa.optimizer import optimize_model
logger = get_logger(__name__)


class ModelExporter:
    """Handle PyTorch model export to ONNX format"""
    
    def __init__(self, config: dict, checkpoint_path: str):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.model = self._load_model()
        self.onnx_path = None
        self.optimized_onnx_path = None
        
    def _load_model(self):
        """Load PyTorch model from checkpoint"""
        # Create model
        model = get_model(self.config['model'])
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    
    def export(self, output_path: str) -> str:
        """
        Export model to ONNX format
        
        Args:
            output_path: Path to save ONNX model
            dummy_input_shape: Shape of dummy input for tracing
            
        Returns:
            Path to exported ONNX file
        """      
        self.onnx_path = output_path

        patch_shape = self.config['loaders']['test']['slice_builder']['patch_shape']  # [80, 170, 170]
        halo_shape = self.config['loaders']['test']['slice_builder']['halo_shape']  # [16, 32, 32]
    
        # Calculate actual input shape with halo
        self.input_shape = [
            patch_shape[0] + 2 * halo_shape[0],  # 80 + 2*16 = 112
            patch_shape[1] + 2 * halo_shape[1],  # 170 + 2*32 = 234
            patch_shape[2] + 2 * halo_shape[2]   # 170 + 2*32 = 234
        ]

        # Create dummy input
        dummy_input_shape = (1, 1, *self.input_shape)  # (1, 1, 80, 170, 170)
        dummy_input = torch.randn(dummy_input_shape, dtype=torch.float32, requires_grad=False)
        print(f" patch shape: {dummy_input_shape}")
    
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            self.onnx_path,
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
        
        logger.info(f"Model exported to: {self.onnx_path}, verifying...")

        # verification
        if self.verify_onnx():
            # Furiosa litmus
            # logger.info("\n=== Furiosa litmus ===")
            # os.system(f"furiosa litmus {output_path}")
            self.optimize_onnx()
            return self.optimized_onnx_path
        else:
            logger.error("\nONNX export failed")

    
    def verify_onnx(self):
        import onnxruntime as ort
        from onnx import shape_inference
        
        logger.info(f"\nverifying: {self.onnx_path}")
        
        # 1. ONNX model load
        model = onnx.load(self.onnx_path)
        
        # 2. Shape inference 
        try:
            model_with_shapes = shape_inference.infer_shapes(model)
            onnx.save(model_with_shapes, self.onnx_path.replace('.onnx', '_shaped.onnx'))
            logger.info("Shape inference success")
        except Exception as e:
            logger.info(f"Shape inference warning: {e}")
        
        # 3. ONNX verification
        try:
            onnx.checker.check_model(model)
            logger.info("ONNX model verification succeed")
        except Exception as e:
            logger.info(f"ONNX verification failed: {e}")
            return False
        
        # 4. ONNX Runtime
        try:
            session = ort.InferenceSession(self.onnx_path)

            input_info = session.get_inputs()[0]
            
            dummy_input_shape = (1, 1, *self.input_shape)
            test_input = np.random.randn(*dummy_input_shape).astype(np.float32)
            outputs = session.run(None, {input_info.name: test_input})
            
            logger.info(f"\noutput shape: {outputs[0].shape}")
            logger.info("ONNX Runtime test passed")

            return True
            
        except Exception as e:
            logger.error(f"ONNX Runtime test failed: {e}")
            return False
        
    def optimize_onnx(self) -> str:
        model = onnx.load(self.onnx_path)

        self.optimized_onnx_path = self.onnx_path.replace('.onnx', '_optimized.onnx')
        logger.info(f"Optimizing ONNX model for NPU: {self.optimized_onnx_path}")
        optimized_model = optimize_model(
            model=model,
            opset_version=13,
            input_shapes={"input": [1, 1] + self.input_shape}
        )
        onnx.save(optimized_model, self.optimized_onnx_path)

        return self.optimized_onnx_path