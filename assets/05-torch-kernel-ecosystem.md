# Section 5: Torch Kernel Ecosystem

## Torch Profiler

A profiling tool that measures kernel execution time, VRAM consumption, and operation-level metrics. It identifies which operations bottleneck performance.

Key insights from profiling:

- Per-operation VRAM usage and execution time
- Kernel launch overhead
- Memory transfer costs between host and device

### ATen Operators

The **aten** namespace contains PyTorch's native CUDA kernels — written in C and CUDA for maximum hardware performance. These operators define how each mathematical operation is computed on GPU.

Source code is available in the [PyTorch GitHub repository](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/cuda).

## Torch Compile

A JIT compiler introduced in **PyTorch 2.0** that:

1. Traces Python code into a computation graph
2. Applies graph-level optimizations (operator fusion, memory planning)
3. Caches compiled subgraphs for subsequent calls

The compiler backend generates optimized subgraph functions, reducing Python overhead and enabling kernel fusion.

## Torch Helion

A higher-level wrapper around Triton that minimizes boilerplate by leveraging existing PyTorch APIs. It bridges the gap between PyTorch's user-friendly interface and Triton's kernel-level control.
