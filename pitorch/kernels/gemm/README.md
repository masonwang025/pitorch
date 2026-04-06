# GEMM Kernels

Float32 matrix multiply: **C[M][N] = A[M][K] * B[K][N]**, row-major.

These are the QPU-side programs. The host-side API that sets up memory, packs uniforms, and launches them lives in `ops/gemm/`.

## Hardware context

The VC4 has **12 QPUs**, each a 16-wide SIMD processor. One QPU instruction operates on 16 floats simultaneously. Data moves between main memory and QPU registers through two paths:

- **VPM** (Vertex Pipe Memory) — 4 KB shared scratchpad, bulk DMA read/write
- **TMU** (Texture Memory Unit) — per-QPU cached memory lookups, each SIMD lane can fetch an independent address

## Kernels

### `gemm_vpm.qasm` — single QPU, VPM

One QPU computes the entire output matrix, one row at a time. A values streamed via uniform FIFO; B rows loaded via VPM DMA.

**Constraints:** M = K = N = DIM ≤ 16

**Uniform stream:** `B_addr, C_addr, DIM, A[0][0], A[0][1], ..., A[M-1][K-1]`

### `gemm_multi.qasm` — 12 QPUs, VPM

Same algorithm as `gemm_vpm.qasm` but each QPU owns disjoint output rows (stride = NUM_QPUs). Per-QPU VPM rows avoid conflicts.

**Constraints:** M = K = N = DIM ≤ 16

### `gemm_tmu.qasm` — single QPU, TMU

One QPU computes one 16-column tile of the output. Both A and B fetched via TMU0 with pipelined prefetch (next iteration's TMU requests overlap current iteration's processing). C written via VPM DMA.

**Data flow for each output row i, column tile ct:**

```
TMU0 FIFO ──→ A[i][k]  (broadcast: all 16 lanes submit same addr)
TMU0 FIFO ──→ B[k][ct+lane]  (per-lane: each lane fetches its column)
                  │
               fmul + fadd ──→ accumulator[0..15]
                  │
               (repeat for k = 0 .. K-1, pipelined)
                  │
VPM DMA  ←──── write C[i][ct..ct+15]
```

**Constraints:** K ≥ 1; N must be a multiple of 16. Host tiles over N in 16-column chunks.

**Uniform stream:** `A_base, B_tile, C_tile, M, K, stride`

The key TMU advantage: no VPM read conflicts, no uniform-stream size limits on A. Supports arbitrary M and K.

### `gemm_tmu_multi.qasm` — 12 QPUs, TMU

Same algorithm as `gemm_tmu.qasm` but row-striped across QPUs. QPU q handles rows q, q+NUM_QPUs, q+2*NUM_QPUs, ... Per-QPU VPM rows for DMA write, same as `gemm_multi.qasm`.

**Uniform stream:** `A_base, B_tile, C_tile, M, K, stride, NUM_QPUs, QPU_NUM, NUM_ROWS`

## VPM vs TMU comparison

| | VPM kernels | TMU kernels |
|---|---|---|
| **A input** | uniform stream (limited by FIFO) | TMU lookup (unlimited) |
| **B input** | VPM DMA (shared, needs per-QPU rows) | TMU lookup (per-QPU, cached) |
| **Max size** | 16×16 | arbitrary M, K; N multiple of 16 |
| **Multi-QPU** | per-QPU VPM rows for read & write | TMU is independent; VPM only for write |
| **Caching** | none | TMU L2 cache (B reuse across QPUs) |
