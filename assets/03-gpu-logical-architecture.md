# Section 3: GPU Logical Architecture

## CUDA Kernel

A **kernel** is the unit of GPU code programmers write — analogous to a function in CPU-targeted languages. When launched, a kernel spawns a **Grid**, the highest logical compute unit.

## CUDA Thread

The lowest logical execution unit, defining each computation step. Threads are scheduled and dispatched by the **Warp Scheduler**, which determines execution order and hardware mapping.

## SIMT Execution Model

GPUs adopt the **Single Instruction, Multiple Threads (SIMT)** model — a single instruction per warp is applied simultaneously to 32 threads, each operating on different data elements.

## Execution Hierarchy

```
Kernel → Grid → Blocks → Warps (32 threads each) → Threads
```

### Grid → Blocks

A Grid divides computation into **Blocks**. Each block represents an independent computation portion — blocks execute without dependencies on each other.

### Blocks → Warps

Each block is further divided into **Warps**. The number of warps per block depends on workload distribution. Warps within the same block can synchronize; warps across blocks cannot.

### Warps → Threads

Each warp contains exactly **32 threads**. After computation, threads synchronize within the warp — Thread 0 typically acts as the leader collecting results.

## Synchronization and Memory

1. Threads within a warp execute in lockstep (SIMT)
2. Warps within a block synchronize via **shared memory** and barriers
3. Blocks write results to **DRAM (global memory)** independently
4. Block execution order is **non-deterministic** — Block 0 may finish before Block 1

## Logical ↔ Physical Mapping

| Logical | Physical |
|---------|----------|
| Thread | CUDA Core (1 thread per core per cycle) |
| Warp | Warp Scheduler unit |
| Block | Streaming Multiprocessor (SM) |
| Grid | Entire GPU |

Key constraints:

- Each SM handles **one block at a time**
- Block scheduling order is determined by hardware
- Multiple SMs work in parallel, reading/writing DRAM simultaneously

## Core Principles

The GPU logical execution model is built on two ideas:

1. **Parallelism** — divide and conquer across the hierarchy
2. **Communication** — collective operations for synchronization and result aggregation
