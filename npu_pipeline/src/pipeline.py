import os
import yaml
import torch
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue

from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.utils import load_checkpoint
from pytorch3dunet.datasets.utils import SliceBuilder, calculate_stats
from pytorch3dunet.augment import transforms

from .model_exporter import ModelExporter
from .preprocessor import Preprocessor
from .quantizer import FuriosaQuantizer
from .postprocessor import Postprocessor
from .inference_engine import InferenceEngine
from .model_optimizer import ModelOptimizer
from pytorch3dunet.unet3d.utils import get_logger
logger = get_logger(__name__)


class UNet3DPipeline:
    """
    NPU Pipeline compatible with pytorch-3dunet configuration format
    """
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline from pytorch-3dunet style config
        
        Args:
            config_path: Path to test_config.yml
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Extract paths and settings from config
        self.model_path = self.config.get('model_path')
        self.checkpoint_path = self.model_path  # Alias for compatibility
        self.output_dir = self.config.get('loaders', {}).get('output_dir', 'output')
        
        # Extract model config
        self.model_config = self.config.get('model', {})
        
        # Extract slice builder (patch) config
        slice_builder_config = self.config.get('loaders', {}).get('test', {}).get('slice_builder', {})
        self.patch_config = {
            'patch_shape': slice_builder_config.get('patch_shape', [80, 170, 170]),
            'stride_shape': slice_builder_config.get('stride_shape', [80, 170, 170]),
            'halo_shape': slice_builder_config.get('halo_shape', [16, 32, 32]),
            'batch_size': self.config.get('loaders', {}).get('batch_size', 1)
        }
        
        # Extract file paths
        self.valid_file_paths = self.config.get('loaders', {}).get('valid', {}).get('file_paths', [])
        self.test_file_paths = self.config.get('loaders', {}).get('test', {}).get('file_paths', [])
        
        # Initialize components
        self.model = None
        self.model_exporter = ModelExporter(self.config, self.checkpoint_path)
        self.quantizer = FuriosaQuantizer()
        self.preprocessor = Preprocessor(self.config)
        self.model_optimizer = ModelOptimizer()
        self.postprocessor = Postprocessor(self.config, self.patch_config)
        self.inference_engine = None
        
        # Paths for generated files
        self.onnx_path = os.path.join(self.output_dir, "unet3d.onnx")
        self.optimized_onnx_path = None
        self.quantized_onnx_path = None
        
        # Data loading settings
        self.num_workers = self.config.get('loaders', {}).get('num_workers', 8)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def export_and_optimize_onnx(self) -> str:
        """Export PyTorch model to ONNX and optimize for NPU"""

        self.optimized_onnx_path = self.model_exporter.export(self.onnx_path)
        
        return self.optimized_onnx_path
    
    def quantize_model(
        self,
        output_path: Optional[str] = None,
        calibration_samples: int = 100,
    ) -> str:
        """
        Quantize model using unified preprocessor
        """
        if self.optimized_onnx_path is None:
            raise ValueError("Model must be exported first")
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, "unet3d_i8.onnx")
        
        
        # Use unified preprocessor to load calibration data
        calibration_patches = self.preprocessor.preprocess_directory(
            self.valid_file_paths,
            calibration_samples
        )

        print(calibration_patches[0].shape)

        # Quantize
        self.quantized_onnx_path = self.quantizer.quantize(
            self.optimized_onnx_path,
            calibration_patches,
            output_path
        )
        
        return self.quantized_onnx_path
    
    def benchmark_optimized(
        self, 
        device: str = "warboy(2)*1",
        run_furiosa_bench: bool = True
    ) -> Dict[str, Any]:
        """Run benchmark with current configuration"""
        if self.quantized_onnx_path is None:
            self.quantized_onnx_path = self.onnx_path.replace('.onnx', '_i8.onnx')
        
        logger.info("Running benchmark...")
        
        # Initialize inference engine
        if self.inference_engine is None:
            self.inference_engine = InferenceEngine(
                self.quantized_onnx_path,
                device,
                self.num_workers,
                self.patch_config
            )
        
        # Run benchmark
        metrics = self.inference_engine.benchmark_patch_based()
        
        # Run furiosa-bench if requested
        if run_furiosa_bench:
            furiosa_metrics = self.model_optimizer.run_furiosa_bench(
                self.quantized_onnx_path,
                device,
                batch_size=self.patch_config['batch_size']
            )
            metrics['furiosa_bench'] = furiosa_metrics
        
        return metrics
    
    async def run_inference_async(
        self,
        device: str = "warboy(2)*1",
        save_predictions: bool = True
    ) -> Dict[str, Any]:
        """Run inference on test files specified in config"""
        if self.quantized_onnx_path is None:
            raise ValueError("Model must be quantized first")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize inference engine
        if self.inference_engine is None:
            self.inference_engine = InferenceEngine(
                self.quantized_onnx_path,
                device,
                self.num_workers,
                self.patch_config
            )
        
        # Collect all test files
        all_test_files = []
        h5_files = [os.path.join(self.test_file_paths, f) 
                    for f in os.listdir(self.test_file_paths) 
                    if f.endswith('.h5')]
        all_test_files.extend(h5_files)
        print("all_test_files: ", all_test_files )
        logger.info(f"Found {len(all_test_files)} test files")
        
        # Process files
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for file_path in all_test_files:
                future = executor.submit(
                    self._process_single_file,
                    file_path,
                    save_predictions
                )
                futures.append((file_path, future))
            
            
            # Collect results
            for file_path, future in tqdm(futures, desc="Processing files"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    results.append({
                        'file_path': file_path,
                        'error': str(e)
                    })
        
        return {
            'results': results,
            'total_files': len(all_test_files),
            'output_dir': self.output_dir,
            'device': device,
            'patch_config': self.patch_config
        }
    
    def _process_single_file(
        self,
        file_path: str,
        save_predictions: bool
    ) -> Dict[str, Any]:
        """Process a single test file"""
        filename = os.path.basename(file_path)
        
        # Load data
        with h5py.File(file_path, 'r') as f:
            raw_data = f['raw'][:]
            label_data = f['label'][:] if 'label' in f else None
        
        original_shape = raw_data.shape
        
        # Get transformer config
        transformer_config = self.config.get('loaders', {}).get('test', {}).get('transformer', {})
        
        # Extract patches
        patches_info = self._extract_patches(raw_data)
        
        # Process patches in batches
        all_predictions = []
        batch_size = self.patch_config['batch_size']
        
        for i in range(0, len(patches_info), batch_size):
            batch_patches = patches_info[i:i + batch_size]
            
            # Prepare batch
            batch_data = []
            for patch_info in batch_patches:
                # Use patch with halo for inference
                patch_data = patch_info['patch_with_halo']
                
                # Apply transforms
                processed = self._apply_transforms(patch_data, transformer_config)
                batch_data.append(processed[0])  # Remove batch dim temporarily
            
            # Stack into batch
            if batch_data:
                batch_array = np.stack(batch_data)
                
                # Run inference
                predictions = self.inference_engine.infer_batch(batch_array)
                
                # Store predictions with metadata
                for j, pred in enumerate(predictions):
                    all_predictions.append({
                        'prediction': pred,
                        'slice_indices': batch_patches[j]['slice_indices'],
                        'patch_idx': batch_patches[j]['patch_idx']
                    })
        
        # Reconstruct volume from patches
        reconstructed = self._reconstruct_volume(
            all_predictions,
            original_shape
        )
        
        # Save predictions
        if save_predictions:
            output_filename = filename.replace('.h5', '_predictions.h5')
            output_path = os.path.join(self.output_dir, output_filename)
            
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('predictions', data=reconstructed, compression='gzip')
                f.create_dataset('raw', data=raw_data, compression='gzip')
            
            logger.info(f"Saved predictions to {output_path}")
        
        # Prepare result
        result = {
            'file_path': file_path,
            'filename': filename,
            'num_patches': len(patches_info),
            'original_shape': original_shape,
            'prediction_shape': reconstructed.shape
        }
        
        # Calculate metrics if ground truth available
        if label_data is not None:
            from sklearn.metrics import f1_score
            
            # Apply threshold
            threshold = self.config.get('prediction_threshold', 0.5)
            pred_binary = (reconstructed > threshold).astype(np.uint8)
            
            # Calculate Dice score
            dice_score = f1_score(
                label_data.flatten(),
                pred_binary.flatten(),
                average='macro'
            )
            
            result['metrics'] = {
                'dice_score': dice_score
            }
        
        return result
    
    def _reconstruct_volume(
        self,
        predictions: List[Dict[str, Any]],
        original_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Reconstruct full volume from patches"""
        # Initialize output volume
        output = np.zeros(original_shape, dtype=np.float32)
        weight_map = np.zeros(original_shape, dtype=np.float32)
        
        halo_shape = self.patch_config['halo_shape']
        
        for pred_info in predictions:
            prediction = pred_info['prediction']
            slice_indices = pred_info['slice_indices']
            
            # Remove batch and channel dimensions if present
            if prediction.ndim == 5:
                prediction = prediction[0, 0]
            elif prediction.ndim == 4:
                prediction = prediction[0]
            
            # Remove halo from prediction
            if any(h > 0 for h in halo_shape):
                prediction = self._remove_halo(prediction, halo_shape)
            
            # Add to output with weighting
            weight = self._create_weight_map(prediction.shape)
            output[slice_indices] += prediction * weight
            weight_map[slice_indices] += weight
        
        # Normalize by weights
        output = np.divide(
            output,
            weight_map,
            out=np.zeros_like(output),
            where=weight_map != 0
        )
        
        return output
    
    def _remove_halo(self, patch: np.ndarray, halo_shape: tuple) -> np.ndarray:
        """Remove halo from patch"""
        slices = []
        for i, halo in enumerate(halo_shape):
            if halo > 0:
                slices.append(slice(halo, -halo))
            else:
                slices.append(slice(None))
        return patch[tuple(slices)]
    
    def _create_weight_map(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Create weight map for smooth blending"""
        weight = np.ones(shape, dtype=np.float32)
        
        # Apply Gaussian-like weighting
        for dim in range(len(shape)):
            edge_size = min(shape[dim] // 4, 20)
            if edge_size > 0:
                # Create 1D weight profile
                profile = np.ones(shape[dim])
                transition = np.linspace(0.3, 1.0, edge_size)
                profile[:edge_size] = transition
                profile[-edge_size:] = transition[::-1]
                
                # Apply to weight map
                for i in range(shape[dim]):
                    if dim == 0:
                        weight[i, ...] *= profile[i]
                    elif dim == 1:
                        weight[:, i, ...] *= profile[i]
                    else:
                        weight[:, :, i] *= profile[i]
        
        return weight    

    def _load_calibration_patches_from_paths(
        self, 
        file_paths: List[str],
        num_samples: int
    ) -> List[np.ndarray]:
        """Load calibration patches from file paths"""
        calibration_patches = []
        
        # Get transformer config
        transformer_config = self.config.get('loaders', {}).get('test', {}).get('transformer', {})
        
        for file_path in file_paths:
            if os.path.isdir(file_path):
                # If directory, get all h5 files
                h5_files = [f for f in os.listdir(file_path) if f.endswith('.h5')]
                h5_paths = [os.path.join(file_path, f) for f in h5_files]
            else:
                h5_paths = [file_path]
            
            for h5_path in tqdm(h5_paths, desc="Loading calibration data"):
                if not os.path.exists(h5_path):
                    logger.warning(f"File not found: {h5_path}")
                    continue
                
                with h5py.File(h5_path, 'r') as f:
                    raw_data = f['raw'][:]
                
                # Extract patches using SliceBuilder
                patches = self._extract_patches(raw_data)
                
                # Process each patch
                for patch_info in patches:
                    patch_data = patch_info['patch_with_halo']
                    
                    # Apply transformations
                    processed_patch = self._apply_transforms(
                        patch_data, 
                        transformer_config
                    )
                    
                    calibration_patches.append(processed_patch)
                    
                    if len(calibration_patches) >= num_samples:
                        return calibration_patches
        
        logger.info(f"Loaded {len(calibration_patches)} calibration patches")
        return calibration_patches
    
    def _extract_patches(self, volume: np.ndarray) -> List[Dict[str, Any]]:
        """Extract patches using pytorch-3dunet's SliceBuilder"""
        # Create slice builder
        slice_builder = SliceBuilder(
            raw_dataset=volume,
            label_dataset=None,
            patch_shape=self.patch_config['patch_shape'],
            stride_shape=self.patch_config['stride_shape']
        )
        
        patches = []
        halo_shape = self.patch_config['halo_shape']
        
        for idx, slice_indices in enumerate(slice_builder.raw_slices):
            # Extract patch without halo
            patch = volume[slice_indices]
            
            # Extract patch with halo if specified
            if any(h > 0 for h in halo_shape):
                patch_with_halo = self._extract_with_halo(
                    volume, 
                    slice_indices, 
                    halo_shape
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
    
    def _extract_with_halo(
        self, 
        volume: np.ndarray, 
        slice_indices: tuple,
        halo_shape: tuple
    ) -> np.ndarray:
        """Extract patch with halo"""
        padded_indices = []
        padding_needed = []
        
        # Handle both 3D and 4D cases (with/without channel dimension)
        spatial_dims = len(slice_indices) if len(slice_indices) == 3 else len(slice_indices) - 1
        
        for i, (idx, halo) in enumerate(zip(slice_indices, halo_shape)):
            if i < len(slice_indices) - spatial_dims:
                # Channel dimension - no halo
                padded_indices.append(idx)
                padding_needed.append((0, 0))
            else:
                # Spatial dimensions
                start = max(0, idx.start - halo)
                stop = min(volume.shape[i], idx.stop + halo)
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
    
    def _apply_transforms(
        self, 
        patch: np.ndarray, 
        transformer_config: Dict[str, Any]
    ) -> np.ndarray:
        """Apply transformations to patch"""
        # Calculate stats for this patch
        stats = calculate_stats(patch)
        
        # Create transformer
        transformer = transforms.Transformer(transformer_config, stats)
        raw_transform = transformer.raw_transform()
        
        # Apply transformations
        transformed = raw_transform(patch)
        
        # Convert to numpy if needed
        if torch.is_tensor(transformed):
            transformed = transformed.cpu().numpy()
        
        # Ensure correct shape (add batch dim if needed)
        if transformed.ndim == 4:
            transformed = np.expand_dims(transformed, 0)
        
        return transformed.astype(np.uint8)