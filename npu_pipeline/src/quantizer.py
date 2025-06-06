import onnx
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from furiosa.quantizer import (
    CalibrationMethod, Calibrator, quantize, 
    ModelEditor, TensorType, get_pure_input_names
)
import tempfile

from pytorch3dunet.unet3d.utils import get_logger
logger = get_logger(__name__)


class FuriosaQuantizer:
    """
    Optimized quantizer that tests multiple calibration methods
    """
    
    def quantize(
        self,
        onnx_path: str,
        calibration_data: List[np.ndarray],
        output_path: str
    ) -> Tuple[str, str]:

        # Load model once
        model = onnx.load(onnx_path)

        # Create calibrator
        calibrator = Calibrator(model, CalibrationMethod.MIN_MAX_ASYM)
        
        # Collect calibration data
        for i, data in enumerate(tqdm(
            calibration_data, 
            leave=False
        )):
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            calibrator.collect_data([[data]])
        
        # Calculate calibration ranges
        calibration_range = calibrator.compute_range()

        editor = ModelEditor(model)
        editor.convert_input_type("input", TensorType.UINT8)
        
        # Quantize model
        quantized_model = quantize(model, calibration_range)

        if output_path is None:
            output_path = onnx_path.replace('.onnx', '_i8.onnx')

        with open(output_path, "wb") as f:
            f.write(bytes(quantized_model))

        return output_path
    