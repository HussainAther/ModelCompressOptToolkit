# AI Model Compression and Optimization Toolkit

This repository provides tools and scripts for compressing and optimizing AI models for deployment on NVIDIA hardware using TensorRT. The toolkit includes scripts for converting models to TensorRT, compressing them, optimizing their performance, and benchmarking their inference times.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Usage](#usage)
  - [Convert a Model to TensorRT](#convert-a-model-to-tensorrt)
  - [Compress a Model](#compress-a-model)
  - [Optimize a Model](#optimize-a-model)
  - [Benchmark a Model](#benchmark-a-model)
- [Scripts](#scripts)
- [Benchmark Results](#benchmark-results)
- [Contributing](#contributing)
- [License](#license)

## Directory Structure
```
ai-model-compression-optimization-toolkit/
│
├── models/
│   ├── README.md
│   └── sample_model.onnx
│
├── scripts/
│   ├── convert_to_tensorrt.py
│   ├── compress_model.py
│   ├── optimize_model.py
│   └── README.md
│
├── benchmarks/
│   ├── benchmark_results.md
│   └── run_benchmarks.py
│
├── README.md
└── requirements.txt
```

## Getting Started

### Requirements
- Python 3.6+
- NVIDIA GPU
- CUDA Toolkit
- cuDNN
- TensorRT

### Installation
First, ensure you have the necessary dependencies installed. You can install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

## Usage

### Convert a Model to TensorRT
To convert an ONNX model to TensorRT:

```bash
python scripts/convert_to_tensorrt.py --model_path models/sample_model.onnx --output_path models/sample_model.trt
```

### Compress a Model
To compress a TensorRT model:

```bash
python scripts/compress_model.py --model_path models/sample_model.trt --output_path models/sample_model_compressed.trt
```

### Optimize a Model
To optimize a TensorRT model:

```bash
python scripts/optimize_model.py --model_path models/sample_model_compressed.trt --output_path models/sample_model_optimized.trt
```

### Benchmark a Model
To benchmark the performance of a TensorRT model:

```bash
python benchmarks/run_benchmarks.py --model_path models/sample_model_optimized.trt
```

## Scripts

### convert_to_tensorrt.py
Converts an ONNX model to TensorRT.

#### Usage
```bash
python convert_to_tensorrt.py --model_path <path_to_onnx_model> --output_path <path_to_save_tensorrt_engine>
```

### compress_model.py
Compresses a TensorRT model. This can include techniques like pruning, quantization, etc.

#### Usage
```bash
python compress_model.py --model_path <path_to_tensorrt_model> --output_path <path_to_save_compressed_model>
```

### optimize_model.py
Optimizes a TensorRT model. This can include techniques like layer fusion, tensor layout optimization, etc.

#### Usage
```bash
python optimize_model.py --model_path <path_to_tensorrt_model> --output_path <path_to_save_optimized_model>
```

## Benchmark Results
The benchmark results for the optimized models are stored in `benchmarks/benchmark_results.md`.

### Example Benchmark Results
| Model                        | Batch Size | Iterations | Average Inference Time (s) |
|------------------------------|------------|------------|----------------------------|
| sample_model_optimized.trt   | 1          | 100        | 0.005678                   |

## Contributing
We welcome contributions to improve this toolkit! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License
This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

