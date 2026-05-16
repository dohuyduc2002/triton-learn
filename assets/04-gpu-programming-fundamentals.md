# Section 4: GPU Programming Fundamentals

## Why CUDA Uses 3D Coordinates

CUDA originated from computer graphics where every vertex has (x, y, z) coordinates. This 3D coordinate system carries into GPU programming — `gridDim`, `blockDim`, `blockIdx`, and `threadIdx` all use 3D indexing to identify which data element each thread processes.

## CUDA Programming — SAXPY Example

**SAXPY** (Single-precision A·X Plus Y) is the canonical GPU kernel example.

### Key CUDA Concepts

| Concept | Purpose |
|---------|---------|
| `__global__` | Declares a kernel visible to the host |
| `const` qualifier | Marks inputs as read-only |
| `threadIdx.x` | Thread's position within its block |
| `blockIdx.x` | Block's position within the grid |
| `blockDim.x` | Number of threads per block |

### Global Thread Index

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

- Blocks advance along the grid dimension
- Threads advance along the block dimension
- Each thread maps to exactly one data element

### Bounds Masking

```cpp
if (idx < n) {
    z[idx] = a * x[idx] + y[idx];
}
```

This mask prevents out-of-bounds memory access when thread count exceeds data size. CUDA launches enough threads to cover the grid; the conditional ensures invalid threads perform no work.

## The CUDA Pain Point

CUDA inherits C++ complexity — manual memory management, type-checking, verbose boilerplate. This reduces productivity for researchers who care about algorithm design, not systems programming.

**Solution**: OpenAI created **Triton** — GPU programming in pure Python.

## Triton

### Design Principles

1. No memory management or type-checking boilerplate
2. **Hardware-agnostic**: supports NVIDIA CUDA, AMD ROCm, Google TPU (XLA), AWS Trainium
3. Operates at the **block level** — threads are abstracted away entirely

### SAXPY in Triton

```python
@triton.jit
def saxpy_kernel(x_ptr, y_ptr, z_ptr, a, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = a * x + y
    tl.store(z_ptr + offsets, z, mask=mask)
```

### Key Triton Concepts

| Concept | Description |
|---------|-------------|
| `@triton.jit` | JIT compiles Python to GPU IR |
| `program_id` | Block index (called "program" in Triton) |
| `BLOCK_SIZE` | Number of elements per block (compile-time constant) |
| `offsets` | Element indices within the block |
| `mask` | Prevents overflow memory access |

### Pointer, Mask, and Offset

- **Pointer**: Memory address of the tensor in DRAM
- **Offset**: Indices of tensor elements within a block
- **Mask**: Guards against out-of-bounds access when tensor size isn't divisible by block size

If `num_elements > offset_range` → overflow addresses are masked in the current block; remaining elements go to the next block.

## Striding — 2D Tensor Access

Memory is physically 1D. To access a 2D tensor, we need **strides** to compute element addresses:

```python
# For a matrix with shape (M, N) stored in row-major:
# element[i, j] is at address: base_ptr + i * stride_row + j * stride_col
row_offsets = tl.arange(0, BLOCK_M)[:, None] * stride_row
col_offsets = tl.arange(0, BLOCK_N)[None, :] * stride_col
ptrs = base_ptr + row_offsets + col_offsets
```

Two distinct pointers (row and column) with their respective offsets allow loading arbitrary 2D sub-blocks from linear memory.

## Matrix Multiplication

### Naive Approach

For C = A(M×K) · B(K×N):
- Select row `i` from A, column `j` from B
- Compute dot product → C[i, j]
- Repeat for all (i, j)

**Problem**: Row-major storage forces loading entire matrix B for each row of C — extremely inefficient.

### Row-Major vs Column-Major

| Order | Layout | Default in |
|-------|--------|-----------|
| Row-major | Rows contiguous in memory | C++, Python/NumPy |
| Column-major | Columns contiguous in memory | Fortran, MATLAB, Julia |

### GEMM — General Matrix Multiplication

The solution: **Group ordering** using tiles. Load only the exact sub-blocks needed from both A and B to compute a tile of C.

### Tiling Strategy

Divide the computation into tiles:

- **Tile M**: Subset of rows from A
- **Tile N**: Subset of columns from B
- **Tile K**: Reduction dimension, iterated over

Each output tile of C is computed by accumulating partial dot products across K tiles:

```
for k in range(0, K, BLOCK_SIZE_K):
    a_tile = load A[m_range, k:k+BLOCK_SIZE_K]
    b_tile = load B[k:k+BLOCK_SIZE_K, n_range]
    accumulator += dot(a_tile, b_tile)
```

### Triton MatMul Implementation

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc += tl.dot(a, b)

    # Store result
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)
```

### Numerical Precision

GPU matmul results are **approximate**. On Ada Lovelace (RTX 6000), results match PyTorch within tolerance but may fail strict equality checks (e.g., atol < 1e-4). This is expected due to floating-point accumulation order differences across CUDA, CUTLASS, cuDNN, and cuBLAS backends.
