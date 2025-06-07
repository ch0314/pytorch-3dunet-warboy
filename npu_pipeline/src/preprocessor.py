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
        
        # Cache for stats to avoid recalculation
        self._stats_cache = None
        
    def process_volume(self, volume: np.ndarray) -> List[Dict]:
        """
        Process a single volume into patches
        
        Args:
            volume: 3D numpy array
            
        Returns:
            List of patch dictionaries
        """
        # Calculate stats once for the entire volume
        if self._stats_cache is None:
            logger.info("Calculating volume statistics...")
            self._stats_cache = calculate_stats(volume)
            logger.info(f"Stats: mean={self._stats_cache['mean']:.3f}, std={self._stats_cache['std']:.3f}")
        
        # Create slice builder
        slice_builder = SliceBuilder(
            raw_dataset=volume,
            label_dataset=None,
            patch_shape=self.patch_shape,
            stride_shape=self.stride_shape
        )
        
        patches = []
        
        # Create transformer once
        transformer = transforms.Transformer(self.transformer_config, self._stats_cache)
        raw_transform = transformer.raw_transform()
        
        for idx, slice_indices in enumerate(slice_builder.raw_slices):
            # Extract patch with halo
            patch_with_halo = self._extract_patch_with_halo(volume, slice_indices)
            
            # Apply transformations
            transformed_patch = self._apply_transforms_cached(patch_with_halo, raw_transform)
            
            patch_info = {
                'patch': transformed_patch,
                'slice_indices': slice_indices,
                'patch_idx': idx,
                'original_shape': volume.shape
            }
            
            patches.append(patch_info)
            
        logger.info(f"Extracted {len(patches)} patches from volume of shape {volume.shape}")
        return patches
    
    def _apply_transforms_cached(self, patch: np.ndarray, transform) -> np.ndarray:
        """
        Apply transformations using cached transformer
        """
        # Apply transformations
        transformed = transform(patch)
        
        # Convert to numpy if tensor
        if isinstance(transformed, torch.Tensor):
            transformed = transformed.cpu().numpy()
            
        return transformed
    
    def _extract_patch_with_halo(self, volume: np.ndarray, slice_indices: Tuple) -> np.ndarray:
        """
        Extract patch with halo (padding) from volume - optimized version
        """
        # Fast path for no halo
        if all(h == 0 for h in self.halo_shape):
            return volume[slice_indices]
        
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
    
    def prepare_batch(self, patches: List[np.ndarray], for_inference: bool = True) -> np.ndarray:
        """
        Prepare batch for NPU inference - optimized version
        
        Args:
            patches: List of preprocessed patches
            for_inference: If True, convert to UINT8 for NPU
            
        Returns:
            Batched numpy array
        """
        # Ensure all patches have same shape
        if not patches:
            raise ValueError("No patches to batch")
            
        # Fast path for single patch
        if len(patches) == 1:
            patch = patches[0]
            if patch.ndim == 3:
                patch = np.expand_dims(patch, (0, 1))
            elif patch.ndim == 4:
                patch = np.expand_dims(patch, 0)
                
            if for_inference and patch.dtype != np.uint8:
                patch = self._convert_to_uint8(patch)
                
            return patch
            
        # Process multiple patches
        processed_patches = []
        for patch in patches:
            if patch.ndim == 3:
                patch = patch[np.newaxis, np.newaxis, ...]  # Add batch and channel dims
            elif patch.ndim == 4:
                patch = patch[np.newaxis, ...]  # Add batch dim
            processed_patches.append(patch)
            
        # Stack into batch
        batch = np.concatenate(processed_patches, axis=0)
        
        # Convert to UINT8 for NPU inference
        if for_inference and batch.dtype != np.uint8:
            batch = self._convert_to_uint8(batch)
                
        return batch
    
    def _convert_to_uint8(self, data: np.ndarray) -> np.ndarray:
        """
        Convert data to uint8 efficiently
        """
        if data.dtype == np.float32:
            # Check if data is normalized to [0, 1]
            data_min, data_max = data.min(), data.max()
            
            if data_min >= -0.1 and data_max <= 1.1:
                # Data is approximately in [0, 1]
                return np.clip(data * 255, 0, 255).astype(np.uint8)
            else:
                # Data needs normalization
                data_normalized = (data - data_min) / (data_max - data_min)
                return (data_normalized * 255).astype(np.uint8)
        else:
            return data.astype(np.uint8)