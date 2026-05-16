# Section 2: GPU Physical Architecture

## CUDA Core

The fundamental worker unit in a GPU. Each CUDA core performs simple arithmetic — addition and multiplication on binary values. Think of it as a minimal calculator.

## Warp

The smallest **scheduling unit** in a GPU — 32 threads sharing a single instruction stream. CUDA cores execute warp threads over one or more cycles depending on hardware availability. Cores are shared across warps and not permanently assigned to any single one.

The **Instruction Register** orchestrates all CUDA cores within a warp, directing computation per task.

## CPU vs GPU — Architectural Difference

| Aspect | CPU | GPU |
|--------|-----|-----|
| Core design | Complex control flow, branching, sequential | Simple control, same instruction across threads |
| Execution | Independent cores, few threads | Warps of 32 threads, thousands of cores |
| Suited for | General-purpose, latency-sensitive | Throughput-heavy, data-parallel workloads |

## Streaming Multiprocessor (SM)

The lowest physical compute unit. Each SM contains:

- Up to 4 warps executing concurrently
- Private registers
- L1/shared memory for low-latency access
- A warp scheduler queuing up to 64 warps (generation-dependent)

The SM handles individual instruction pieces broken from kernel tasks.

## Texture Processing Cluster (TPC)

Two consecutive SMs form a TPC, supported by:

- L2 cache partitions
- Memory controllers (responsible for fetching data from VRAM)

## VRAM (GPU DRAM)

GPU's dedicated memory, analogous to system RAM but:

1. Integrated directly into the GPU (not a motherboard slot)
2. Optimized for extremely high data-transfer bandwidth

| Type | Use Case |
|------|----------|
| GDDR (Graphics Double Data Rate) | Consumer GPUs |
| HBM (High Bandwidth Memory) | Data center GPUs |

## Memory Hierarchy

- **L1 Cache / Shared Memory**: Per-SM, lowest latency
- **L2 Cache**: Shared across all GPCs, lower latency than DRAM
- **DRAM (VRAM)**: Global memory, highest capacity, highest latency

## PCIe Interface

Universal standard for transferring data from external sources (host CPU/RAM) to the GPU.

## Simplified Architecture Summary

All compute units — from CUDA cores to SMs to GPCs — are constructed from billions of transistors. The hierarchy flows:

```
CUDA Cores → Warps → SM → TPC → GPC → Full GPU Chip
```

## NVIDIA Blackwell (GB202) — RTX 5090

**92.2 billion transistors** comprising:

| Component | Count | Per SM |
|-----------|-------|--------|
| CUDA Cores | 24,576 | 128 |
| RT Cores | 192 | 1 (3rd gen) |
| Tensor Cores | 768 | 4 (4th gen) |
| SMs | 144 | — |
| Register File | — | 256 KB |
| L1 Cache | — | 128 KB |

Each GPC contains 8 TPCs plus a Raster Engine.

### SM Internal Components

- **CUDA Cores**: Simple arithmetic (add, multiply)
- **Tensor Cores**: Matrix/tensor computation
- **LD/ST Units**: Load and Store for data transfer
- **SFU (Special Function Unit)**: Approximate complex functions (exp, sqrt, etc.)
