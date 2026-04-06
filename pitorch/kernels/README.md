# kernels/

QPU assembly programs (`.qasm`) for the VideoCore IV GPU. Assembled with [vc4asm](https://maazl.de/project/vc4asm/doc/index.html) into C arrays that get linked into the ARM binary and copied to GPU memory at runtime.

## Layout

```
gemm/            GEMM (matrix multiply) kernels — the first real throughput primitive
examples/        Standalone demos used during bringup (deadbeef, index, parallel_add)
include/         Shared vc4asm macros (vc4.qinc: VPM/DMA helpers, constants)
```

## How a kernel gets from .qasm to running on hardware

```
.qasm source
    │  vc4asm assembles
    ▼
shader.c / shader.h    (uint32_t array of QPU instructions)
    │  ARM cross-compiler links into .bin
    ▼
ARM binary on Pi
    │  memcpy into gpu_alloc'd memory
    ▼
qpu_launch() writes code address to V3D_SRQPC
    │  V3D scheduler dispatches to QPU(s)
    ▼
QPU executes: reads uniforms, DMA via VPM, computes, writes results
```

Each kernel directory has its own README explaining the algorithm.
