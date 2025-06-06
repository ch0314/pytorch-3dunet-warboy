import numpy as np
from typing import List, Dict, Any, Tuple
from pytorch3dunet.unet3d.utils import get_logger
logger = get_logger(__name__)


class Postprocessor:
    """
    Postprocessor that reconstructs full volume from patches
    """
    
    def __init__(self, config: Dict[str, Any], patch_config: Dict[str, Any]):
        self.config = config
        self.patch_config = patch_config
        self.halo_shape = tuple(patch_config['halo_shape'])
        self.threshold = config.get('prediction_threshold', 0.5)
        self.apply_sigmoid = config.get('model', {}).get('final_sigmoid', True)
        
    def reconstruct_from_patches(
        self, 
        patch_predictions: List[Dict[str, Any]],
        original_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Reconstruct full volume from overlapping patches
        
        Uses weighted averaging in overlapping regions
        """
        # Initialize output volume and weight map
        output_volume = np.zeros(original_shape, dtype=np.float32)
        weight_map = np.zeros(original_shape, dtype=np.float32)
        
        for patch_data in patch_predictions:
            prediction = patch_data['prediction']
            slice_idx = patch_data['info']['slice_idx']
            
            # Remove batch/channel dimensions if present
            if prediction.ndim == 5:
                prediction = prediction[0, 0]
            elif prediction.ndim == 4:
                prediction = prediction[0]
            
            # Apply sigmoid if needed
            if not self.apply_sigmoid:
                prediction = self._sigmoid(prediction)
            
            # Remove halo from prediction
            prediction = self._remove_halo(prediction)
            
            # Create weight for this patch (higher in center, lower at edges)
            weight = self._create_patch_weight(prediction.shape)
            
            # Add to output volume
            output_volume[slice_idx] += prediction * weight
            weight_map[slice_idx] += weight
        
        # Normalize by weights
        output_volume = np.divide(
            output_volume, 
            weight_map, 
            out=np.zeros_like(output_volume), 
            where=weight_map != 0
        )
        
        # Apply threshold for binary segmentation
        if self.config.get('binary_output', True):
            output_volume = (output_volume > self.threshold).astype(np.uint8)
        
        logger.info(f"Reconstructed volume shape: {output_volume.shape}")
        return output_volume
    
    def _remove_halo(self, patch: np.ndarray) -> np.ndarray:
        """Remove halo from patch"""
        slices = []
        for halo in self.halo_shape:
            if halo > 0:
                slices.append(slice(halo, -halo))
            else:
                slices.append(slice(None))
        
        return patch[tuple(slices)]
    
    def _create_patch_weight(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Create weight map for patch (Gaussian-like weighting)
        Higher weight in center, lower at edges for smooth blending
        """
        weight = np.ones(shape, dtype=np.float32)
        
        # Create smooth transition at edges
        for dim in range(3):
            edge_size = min(shape[dim] // 4, 20)  # Transition zone size
            
            # Create 1D weight profile
            profile = np.ones(shape[dim])
            if edge_size > 0:
                # Smooth transition at start
                transition = np.linspace(0.1, 1.0, edge_size)
                profile[:edge_size] = transition
                # Smooth transition at end
                profile[-edge_size:] = transition[::-1]
            
            # Apply to weight map
            for i in range(shape[dim]):
                if dim == 0:
                    weight[i, :, :] *= profile[i]
                elif dim == 1:
                    weight[:, i, :] *= profile[i]
                else:
                    weight[:, :, i] *= profile[i]
        
        return weight
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid function"""
        return 1 / (1 + np.exp(-x))