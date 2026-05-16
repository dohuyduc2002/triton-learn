# Section 6: Conclusion

## Real-World Applications

### LLM Inference Engines

Engines like **vLLM**, **SGLang**, and **Triton Inference Server** write custom kernels to manage **KV cache** efficiently during inference — a critical bottleneck in autoregressive generation.

### FlashAttention

Custom GPU kernels that **fuse attention operations** into a single kernel, eliminating redundant memory transfers between SRAM and DRAM during training. This reduces both memory footprint and computation time.

## When to Write Your Own GPU Kernel

Consider custom kernel development when:

- You deploy **open-source models** for training/inference and need to optimize specific operations
- Your team has **hardware-level expertise** in GPU architecture
- You need performance beyond what PyTorch's default kernels provide
- You want to understand how tensors are computed beneath the framework abstraction
