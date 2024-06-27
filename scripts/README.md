# Scripts

This directory contains scripts for converting, compressing, and optimizing AI models for deployment on NVIDIA hardware.

## Scripts

### convert_to_tensorrt.py
Converts an ONNX model to TensorRT.

#### Usage
```bash
python convert_to_tensorrt.py --model_path <path_to_onnx_model> --output_path <path_to_save_tensorrt_engine>

