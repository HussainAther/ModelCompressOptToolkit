import time
import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_file_path):
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def benchmark_model(engine, batch_size=1, iterations=100):
    context = engine.create_execution_context()
    input_shape = engine.get_binding_shape(0)
    output_shape = engine.get_binding_shape(1)
    input_size = trt.volume(input_shape) * batch_size
    output_size = trt.volume(output_shape) * batch_size

    d_input = cuda.mem_alloc(input_size * np.dtype(np.float32).itemsize)
    d_output = cuda.mem_alloc(output_size * np.dtype(np.float32).itemsize)

    h_input = np.random.random(input_size).astype(np.float32)
    h_output = np.empty(output_size, dtype=np.float32)

    cuda.memcpy_htod(d_input, h_input)

    times = []
    for _ in range(iterations):
        start_time = time.time()
        context.execute_v2([int(d_input), int(d_output)])
        cuda.memcpy_dtoh(h_output, d_output)
        times.append(time.time() - start_time)

    avg_time = sum(times) / iterations
    print(f"Average inference time: {avg_time:.6f} seconds")
    return avg_time

if __name__ == "__main__":
    import numpy as np

    parser = argparse.ArgumentParser(description="Benchmark a TensorRT model")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the TensorRT model")
    
    args = parser.parse_args()

    engine = load_engine(args.model_path)
    avg_time = benchmark_model(engine)
    print(f"Benchmark results for {args.model_path}:")
    print(f"Average Inference Time: {avg_time:.6f} seconds")

