import numpy as np
from typing import List, Dict, Tuple, Optional
from pytorch3dunet.datasets.utils import SliceBuilder, calculate_stats
from pytorch3dunet.augment import transforms
import torch
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Preprocess data with patching, padding, and transformations
    """
    
    def __init__(self, config: Dict):
        """
        Initialize preprocessor with config
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        test_config = config.get('loaders', {}).get('test', {})
        
        # Extract slice builder config
        slice_builder_config = test_config.get('slice_builder', {})
        self.patch_shape = tuple(slice_builder_config.get('patch_shape', [80, 170, 170]))
        self.stride_shape = tuple(slice_builder_config.get('stride_shape', [80, 170, 170]))
        self.halo_shape = tuple(slice_builder_config.get('halo_shape', [0, 0, 0]))
        
        # Extract transformer config
        self.transformer_config = test_config.get('transformer', {})
        
    def process_volume(self, volume: np.ndarray) -> List[Dict]:
        """
        Process a single volume into patches
        
        Args:
            volume: 3D numpy array
            
        Returns:
            List of patch dictionaries
        """
        # Create slice builder
        slice_builder = SliceBuilder(
            raw_dataset=volume,
            label_dataset=None,
            patch_shape=self.patch_shape,
            stride_shape=self.stride_shape
        )
        
        patches = []
        
        for idx, slice_indices in enumerate(slice_builder.raw_slices):
            # Extract patch with halo
            patch_with_halo = self._extract_patch_with_halo(volume, slice_indices)
            
            # Apply transformations
            transformed_patch = self._apply_transforms(patch_with_halo)
            
            patch_info = {
                'patch': transformed_patch,
                'slice_indices': slice_indices,
                'patch_idx': idx,
                'original_shape': volume.shape
            }
            
            patches.append(patch_info)
            
        logger.info(f"Extracted {len(patches)} patches from volume of shape {volume.shape}")
        return patches
    
    def _extract_patch_with_halo(self, volume: np.ndarray, slice_indices: Tuple) -> np.ndarray:
        """
        Extract patch with halo (padding) from volume
        """
        # Calculate indices with halo
        padded_indices = []
        padding_needed = []
        
        for i, (idx, halo) in enumerate(zip(slice_indices, self.halo_shape)):
            # Skip channel dimension if present
            if len(slice_indices) == 4 and i == 0:
                padded_indices.append(idx)
                padding_needed.append((0, 0))
            else:
                dim_idx = i if len(slice_indices) == 3 else i - 1
                volume_size = volume.shape[i]
                
                # Calculate padded indices
                start = max(0, idx.start - halo)
                stop = min(volume_size, idx.stop + halo)
                padded_indices.append(slice(start, stop))
                
                # Calculate padding needed
                pad_before = halo - (idx.start - start)
                pad_after = halo - (stop - idx.stop)
                padding_needed.append((pad_before, pad_after))
        
        # Extract patch
        patch = volume[tuple(padded_indices)]
        
        # Apply padding if needed
        if any(p != (0, 0) for p in padding_needed):
            patch = np.pad(patch, padding_needed, mode='reflect')
            
        return patch
    
    def _apply_transforms(self, patch: np.ndarray) -> np.ndarray:
        """
        Apply transformations to patch
        """
        # Calculate statistics
        stats = calculate_stats(patch)
        
        # Create transformer
        transformer = transforms.Transformer(self.transformer_config, stats)
        raw_transform = transformer.raw_transform()
        
        # Apply transformations
        transformed = raw_transform(patch)
        
        # Convert to numpy if tensor
        if isinstance(transformed, torch.Tensor):
            transformed = transformed.cpu().numpy()
            
        return transformed
    
    def prepare_batch(self, patches: List[np.ndarray], for_inference: bool = True) -> np.ndarray:
        """
        Prepare batch for NPU inference
        
        Args:
            patches: List of preprocessed patches
            for_inference: If True, convert to UINT8 for NPU
            
        Returns:
            Batched numpy array
        """
        # Ensure all patches have same shape
        if not patches:
            raise ValueError("No patches to batch")
            
        # Add batch dimension if needed
        processed_patches = []
        for patch in patches:
            if patch.ndim == 3:
                patch = np.expand_dims(patch, 0)  # Add channel dim
            if patch.ndim == 4:
                patch = np.expand_dims(patch, 0)  # Add batch dim
            processed_patches.append(patch)
            
        # Stack into batch
        batch = np.vstack(processed_patches)
        
        # Convert to UINT8 for NPU inference
        if for_inference:
            # Normalize to [0, 255] range if needed
            if batch.dtype == np.float32:
                # Assuming data is already normalized to [0, 1] or standardized
                # If standardized, denormalize first
                if np.min(batch) < 0 or np.max(batch) > 1:
                    # Data is standardized, convert to [0, 1]
                    batch = (batch - np.min(batch)) / (np.max(batch) - np.min(batch))
                
                # Convert to [0, 255] and then to uint8
                batch = (batch * 255).astype(np.uint8)
            elif batch.dtype != np.uint8:
                # Convert to uint8
                batch = batch.astype(np.uint8)
                
        logger.debug(f"Prepared batch with shape {batch.shape} and dtype {batch.dtype}")
        
        return batch