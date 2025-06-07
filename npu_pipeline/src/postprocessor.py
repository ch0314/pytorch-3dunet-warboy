import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import f1_score, jaccard_score
import logging

logger = logging.getLogger(__name__)


class Postprocessor:
    """
    Postprocess predictions: reconstruct volume and evaluate
    """
    
    def __init__(self, config: Dict):
        """
        Initialize postprocessor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Extract configurations
        test_config = config.get('loaders', {}).get('test', {})
        slice_config = test_config.get('slice_builder', {})
        
        self.patch_shape = tuple(slice_config.get('patch_shape', [80, 170, 170]))
        self.stride_shape = tuple(slice_config.get('stride_shape', [80, 170, 170]))
        self.halo_shape = tuple(slice_config.get('halo_shape', [0, 0, 0]))
        
        # Model config
        self.model_config = config.get('model', {})
        self.final_sigmoid = self.model_config.get('final_sigmoid', True)
        self.threshold = config.get('threshold', 0.5)
        
    def process_predictions(
        self,
        predictions: List[Dict],
        original_shape: Tuple[int, ...],
        label: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Process predictions: reconstruct and evaluate
        
        Args:
            predictions: List of prediction dictionaries
            original_shape: Original volume shape
            label: Ground truth label (optional)
            
        Returns:
            Results dictionary
        """
        # Reconstruct volume from patches
        reconstructed = self._reconstruct_volume(predictions, original_shape)
        
        # Apply sigmoid if needed
        if self.final_sigmoid:
            reconstructed = self._sigmoid(reconstructed)
            
        # Apply threshold for binary segmentation
        segmentation = (reconstructed > self.threshold).astype(np.uint8)
        
        result = {
            'prediction': segmentation,
            'probability': reconstructed,
            'original_shape': original_shape,
            'num_patches': len(predictions)
        }
        
        # Evaluate if label provided
        if label is not None:
            metrics = self._evaluate(segmentation, label)
            result['metrics'] = metrics
            
        return result
    
    def _reconstruct_volume(
        self,
        predictions: List[Dict],
        original_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Reconstruct full volume from patches
        """
        # Initialize output volume and weight map
        output = np.zeros(original_shape, dtype=np.float32)
        weight_map = np.zeros(original_shape, dtype=np.float32)
        
        for pred_dict in predictions:
            prediction = pred_dict['prediction']
            patch_info = pred_dict['patch_info']
            slice_indices = patch_info['slice_indices']
            
            # Remove batch and channel dimensions
            if prediction.ndim == 5:
                prediction = prediction[0, 0]
            elif prediction.ndim == 4:
                prediction = prediction[0]
            elif prediction.ndim == 3 and len(slice_indices) == 3:
                pass  # Already correct shape
            else:
                logger.warning(f"Unexpected prediction shape: {prediction.shape}")
                
            # Remove halo from prediction
            if any(h > 0 for h in self.halo_shape):
                prediction = self._remove_halo(prediction)
                
            # Create weight for smooth blending
            weight = self._create_weight_map(prediction.shape)
            
            # Add to output with weight
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
    
    def _remove_halo(self, patch: np.ndarray) -> np.ndarray:
        """
        Remove halo from patch
        """
        slices = []
        for halo in self.halo_shape:
            if halo > 0:
                slices.append(slice(halo, -halo))
            else:
                slices.append(slice(None))
                
        return patch[tuple(slices)]
    
    def _create_weight_map(self, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Create weight map for smooth blending
        """
        weight = np.ones(shape, dtype=np.float32)
        
        # Apply Gaussian-like weighting for smooth blending
        for dim in range(len(shape)):
            edge_size = min(shape[dim] // 4, 20)
            
            if edge_size > 0:
                # Create 1D weight profile
                profile = np.ones(shape[dim])
                
                # Smooth transition at edges
                transition = np.linspace(0.1, 1.0, edge_size)
                profile[:edge_size] = transition
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
        """
        Apply sigmoid function
        """
        return 1 / (1 + np.exp(-x))
    
    def _evaluate(self, prediction: np.ndarray, label: np.ndarray) -> Dict:
        """
        Evaluate prediction against ground truth
        """
        # Flatten arrays
        pred_flat = prediction.flatten()
        label_flat = label.flatten()
        
        # Check if labels are binary or multiclass
        unique_labels = np.unique(label_flat)
        n_classes = len(unique_labels)
        
        metrics = {}
        
        if n_classes == 2 and set(unique_labels) == {0, 1}:
            # Binary classification
            dice = f1_score(label_flat, pred_flat, average='binary')
            iou = jaccard_score(label_flat, pred_flat, average='binary')
            
            # Pixel-wise accuracy
            accuracy = np.mean(pred_flat == label_flat)
            
            # Sensitivity and specificity
            tp = np.sum((pred_flat == 1) & (label_flat == 1))
            tn = np.sum((pred_flat == 0) & (label_flat == 0))
            fp = np.sum((pred_flat == 1) & (label_flat == 0))
            fn = np.sum((pred_flat == 0) & (label_flat == 1))
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics = {
                'dice': dice,
                'iou': iou,
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity
            }
        else:
            # Multiclass classification
            logger.warning(f"Detected {n_classes} classes in labels: {unique_labels}")
            
            # For multiclass, convert prediction to binary (foreground vs background)
            # Assuming 0 is background and all others are foreground
            pred_binary = (pred_flat > 0).astype(int)
            label_binary = (label_flat > 0).astype(int)
            
            dice = f1_score(label_binary, pred_binary, average='binary')
            iou = jaccard_score(label_binary, pred_binary, average='binary')
            accuracy = np.mean(pred_binary == label_binary)
            
            metrics = {
                'dice': dice,
                'iou': iou,
                'accuracy': accuracy,
                'n_classes_in_label': n_classes,
                'unique_labels': unique_labels.tolist()
            }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics