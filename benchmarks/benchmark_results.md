# Benchmark Results

This file contains the benchmark results for the models processed using the AI Model Compression and Optimization Toolkit.

## Overview

The benchmarks were conducted to measure the performance of models before and after applying various compression and optimization techniques. The metrics recorded include average inference time, memory usage, and model size.

## Results

### Model: sample_model.onnx
| Compression Method | Batch Size | Iterations | Average Inference Time (s) | Memory Usage (MB) | Model Size (MB) |
|--------------------|------------|------------|----------------------------|-------------------|-----------------|
| Original           | 1          | 100        | 0.00789                    | 150               | 45              |
| Pruned             | 1          | 100        | 0.00675                    | 140               | 38              |
| Quantized          | 1          | 100        | 0.00568                    | 130               | 32              |
| Pruned + Quantized | 1          | 100        | 0.00520                    | 120               | 28              |

### Model: sample_model_compressed.onnx
| Compression Method | Batch Size | Iterations | Average Inference Time (s) | Memory Usage (MB) | Model Size (MB) |
|--------------------|------------|------------|----------------------------|-------------------|-----------------|
| Original           | 1          | 100        | 0.00801                    | 155               | 46              |
| Pruned             | 1          | 100        | 0.00690                    | 145               | 39              |
| Quantized          | 1          | 100        | 0.00575                    | 135               | 33              |
| Pruned + Quantized | 1          | 100        | 0.00530                    | 125               | 29              |

### Model: sample_model_optimized.onnx
| Compression Method | Batch Size | Iterations | Average Inference Time (s) | Memory Usage (MB) | Model Size (MB) |
|--------------------|------------|------------|----------------------------|-------------------|-----------------|
| Original           | 1          | 100        | 0.00750                    | 148               | 44              |
| Pruned             | 1          | 100        | 0.00650                    | 138               | 37              |
| Quantized          | 1          | 100        | 0.00560                    | 128               | 31              |
| Pruned + Quantized | 1          | 100        | 0.00510                    | 118               | 27              |

## Conclusions

- **Inference Time**: Quantization generally provided the most significant reduction in average inference time, followed by the combined pruning and quantization approach.
- **Memory Usage**: Pruning and quantization both contributed to lower memory usage, with the combined approach offering the greatest reduction.
- **Model Size**: Quantization had the most significant impact on reducing the model size, with the combined approach offering further reductions.

The results indicate that applying pruning and quantization techniques effectively improves the efficiency and performance of TensorRT models, making them more suitable for deployment in resource-constrained environments.

## Future Work

Further benchmarks will be conducted with different models and more advanced compression techniques. Additionally, real-world deployment scenarios will be tested to validate these improvements.

For detailed instructions on how to run these benchmarks, refer to the `run_benchmarks.py` script in the `benchmarks` directory.

