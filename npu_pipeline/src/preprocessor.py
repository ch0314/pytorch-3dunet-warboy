import numpy as np
import torch
import h5py
import os
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional
from pytorch3dunet.augment import transforms
from pytorch3dunet.datasets.utils import calculate_stats
from pytorch3dunet.datasets.utils import SliceBuilder
from pytorch3dunet.unet3d.utils import get_logger
logger = get_logger(__name__)


class Preprocessor:
    """
    Unified preprocessor for both calibration and inference
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Extract configurations
        self.model_config = config.get('model', {})
        test_config = config.get('loaders', {}).get('test', {})
        
        # Slice builder (patch) configuration
        slice_builder_config = test_config.get('slice_builder', {})
        self.patch_shape = tuple(slice_builder_config.get('patch_shape', [80, 170, 170]))
        self.stride_shape = tuple(slice_builder_config.get('stride_shape', [80, 170, 170]))
        self.halo_shape = tuple(slice_builder_config.get('halo_shape', [16, 32, 32]))
        
        # Transformer configuration
        self.transformer_config = test_config.get('transformer', {})
        
        # Batch size
        self.batch_size = config.get('loaders', {}).get('batch_size', 1)
        
        logger.info(f"Preprocessor initialized with patch_shape: {self.patch_shape}")
    
    def extract_patches_from_volume(self, volume: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract patches from a volume using SliceBuilder
        
        Returns:
            List of patch dictionaries containing patch data and metadata
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
            # Extract patch without halo
            patch = volume[slice_indices]
            
            # Extract patch with halo if specified
            if any(h > 0 for h in self.halo_shape):
                patch_with_halo = self._extract_patch_with_halo(
                    volume, 
                    slice_indices, 
                    self.halo_shape
                )
            else:
                patch_with_halo = patch
            
            patches.append({
                'patch': patch,
                'patch_with_halo': patch_with_halo,
                'slice_indices': slice_indices,
                'patch_idx': idx
            })
        
        return patches
    
    def _extract_patch_with_halo(
        self, 
        volume: np.ndarray, 
        slice_indices: tuple,
        halo_shape: tuple
    ) -> np.ndarray:
        """Extract patch with halo (context) around it"""
        padded_indices = []
        padding_needed = []
        
        # Check if we have channel dimension
        has_channel = len(slice_indices) == 4
        
        for i, (idx, halo) in enumerate(zip(slice_indices, halo_shape)):
            if has_channel and i == 0:
                # First dimension is channel - no halo
                padded_indices.append(idx)
                padding_needed.append((0, 0))
            else:
                # Spatial dimensions
                dim_idx = i if not has_channel else i - 1
                volume_size = volume.shape[i]
                
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
    
    def process_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing transformations to a single patch
        
        Args:
            patch: Raw patch data
            
        Returns:
            Preprocessed patch ready for inference
        """
        # Calculate statistics for this patch
        stats = calculate_stats(patch)
        
        # Create transformer
        transformer = transforms.Transformer(self.transformer_config, stats)
        raw_transform = transformer.raw_transform()
        
        # Apply transformations
        transformed = raw_transform(patch)
        
        # Convert to numpy if needed
        if torch.is_tensor(transformed):
            transformed = transformed.cpu().numpy()
        
        # Ensure correct shape (add batch dim if needed)
        if transformed.ndim == 3:
            transformed = np.expand_dims(transformed, 0)
            transformed = np.expand_dims(transformed, 0)
        elif transformed.ndim == 4:
            transformed = np.expand_dims(transformed, 0)
        
        return transformed.astype(np.float32)
    
    def load_and_process_patches_from_file(
        self, 
        file_path: str,
        max_patches: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Load patches from a single H5 file and process them
        
        Args:
            file_path: Path to H5 file
            max_patches: Maximum number of patches to extract
            
        Returns:
            List of processed patches
        """
        processed_patches = []
        
        # Load raw data
        with h5py.File(file_path, 'r') as f:
            raw_data = f['raw'][:]
        
        # Extract patches
        patches = self.extract_patches_from_volume(raw_data)
        
        # Process patches
        for i, patch_info in enumerate(patches):
            if max_patches and i >= max_patches:
                break
                
            # Use patch with halo for processing
            patch_data = patch_info['patch_with_halo']
            
            # Process patch
            processed = self.process_patch(patch_data)
            processed_patches.append(processed)
        
        return processed_patches

    
    def prepare_batch(self, patches: List[np.ndarray]) -> np.ndarray:
        """
        Prepare a batch of patches for inference
        
        Args:
            patches: List of processed patches
            
        Returns:
            Batched array ready for NPU inference
        """
        # Remove batch dimension from individual patches
        patches_unbatched = [p[0] if p.ndim == 5 else p for p in patches]
        
        # Stack into batch
        batch = np.stack(patches_unbatched)
        
        return batch
    
    
    def preprocess_directory(
        self,
        directory_path: str,
        max_patches_per_file: Optional[int] = None,
        file_extensions: List[str] = ['.h5', '.hdf5', '.hdf', '.hd5']
    ) -> List[np.ndarray]:
        """
        Preprocess all H5 files in a directory
        
        Args:
            directory_path: Path to directory containing H5 files
            max_patches_per_file: Maximum number of patches to extract per file
            file_extensions: List of valid H5 file extensions
            
        Returns:
            Dictionary mapping file paths to their processed patches
        """
        import os
        from pathlib import Path
        
        # Convert to Path object for easier handling
        dir_path = Path(directory_path)
        
        # Verify directory exists
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Find all H5 files in directory
        h5_files = []
        for ext in file_extensions:
            h5_files.extend(dir_path.glob(f"*{ext}"))
        
        if not h5_files:
            print(f"No H5 files found in {directory_path}")
            return {}
        
        # Process each file
        results = []
        total_patches = 0
        
        print(f"Found {len(h5_files)} H5 files to process")
        
        for file_path in h5_files:
            try:
                print(f"Processing: {file_path.name}")
                
                # Process patches from file
                processed_patches = self.load_and_process_patches_from_file(
                    str(file_path),
                    max_patches=max_patches_per_file
                )
                
                # Store results
                results.extend(processed_patches)
                total_patches += len(processed_patches)
                
                print(f"  - Extracted {len(processed_patches)} patches")
                
            except Exception as e:
                print(f"  - Error processing {file_path.name}: {str(e)}")
                continue
        
        print(f"\nTotal patches processed: {total_patches}")
        return results