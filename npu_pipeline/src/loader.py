import os
import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)


class H5Loader:
    """
    Load H5 files from directory or config paths and convert to numpy arrays
    """
    
    def __init__(self, internal_path: str = 'raw'):
        """
        Initialize H5 loader
        
        Args:
            internal_path: H5 internal dataset path (default: 'raw')
        """
        self.internal_path = internal_path
        self.supported_extensions = ['.h5', '.hdf5', '.hdf', '.hd5']
        
    def load_from_directory(self, directory_path: str) -> List[Dict[str, Union[np.ndarray, str]]]:
        """
        Load all H5 files from a directory
        
        Args:
            directory_path: Path to directory containing H5 files
            
        Returns:
            List of dictionaries containing numpy arrays and file paths
        """
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")
        
        h5_files = []
        for ext in self.supported_extensions:
            h5_files.extend(dir_path.glob(f"*{ext}"))
        
        return self._load_files(h5_files)
    
    def load_from_config(self, config: Dict) -> Dict[str, List[Dict[str, Union[np.ndarray, str]]]]:
        """
        Load H5 files based on config file paths
        
        Args:
            config: Configuration dictionary containing file paths
            
        Returns:
            Dictionary with 'train', 'valid', 'test' keys containing loaded data
        """
        results = {}
        
        # Extract paths from config
        loaders_config = config.get('loaders', {})
        
        # Load validation data if exists
        valid_paths = loaders_config.get('valid', {}).get('file_paths', [])
        if valid_paths:
            if isinstance(valid_paths, str):
                valid_paths = [valid_paths]
            results['valid'] = self._load_from_paths(valid_paths)
            
        # Load test data if exists
        test_paths = loaders_config.get('test', {}).get('file_paths', [])
        if test_paths:
            if isinstance(test_paths, str):
                test_paths = [test_paths]
            results['test'] = self._load_from_paths(test_paths)
            
        return results
    
    def _load_from_paths(self, paths: List[str]) -> List[Dict[str, Union[np.ndarray, str]]]:
        """
        Load H5 files from list of paths (can be files or directories)
        """
        all_files = []
        
        for path in paths:
            path_obj = Path(path)
            if path_obj.is_dir():
                # If directory, get all H5 files
                for ext in self.supported_extensions:
                    all_files.extend(path_obj.glob(f"*{ext}"))
            elif path_obj.is_file() and path_obj.suffix in self.supported_extensions:
                all_files.append(path_obj)
            else:
                logger.warning(f"Path not found or not supported: {path}")
                
        return self._load_files(all_files)
    
    def _load_files(self, file_paths: List[Path]) -> List[Dict[str, Union[np.ndarray, str]]]:
        """
        Load H5 files and return numpy arrays
        """
        data_list = []
        
        for file_path in file_paths:
            try:
                with h5py.File(file_path, 'r') as f:
                    if self.internal_path not in f:
                        logger.warning(f"Dataset '{self.internal_path}' not found in {file_path}")
                        continue
                        
                    data = f[self.internal_path][:]
                    
                    # Also load label if exists
                    label = None
                    if 'label' in f:
                        label = f['label'][:]
                    
                    data_dict = {
                        'data': data,
                        'label': label,
                        'file_path': str(file_path),
                        'file_name': file_path.name,
                        'shape': data.shape
                    }
                    
                    data_list.append(data_dict)
                    logger.info(f"Loaded {file_path.name} with shape {data.shape}")
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        return data_list