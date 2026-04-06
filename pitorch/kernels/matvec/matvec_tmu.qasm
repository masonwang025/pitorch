# matvec_tmu.qasm — Multi-QPU float matvec: x via uniforms, W via TMU
#
# Computes 16 output elements of y = W @ x.
# Lane l computes y[row_base + l] = sum_k W[row_base + l][k] * x[k].
# W fetched per-lane via TMU, x broadcast to all lanes via uniform FIFO.
#
# Uniform stream (per-QPU):
#   [0]  W_row_base   bus addr of W[row_start][0] for this QPU's 16 rows
#   [1]  y_row_base   bus addr of y[row_start]
#   [2]  IN_DIM       inner dimension (K)
#   [3]  QPU_NUM      for per-QPU VPM row (avoids DMA write conflicts)
#   [4..] x[0..K-1]   input vector values (broadcast)
#
# TMU addressing: each lane l fetches from
#   W_row_base + l * IN_DIM * 4 + k * 4
# which is W[row_start + l][k] in row-major layout.

# ---- read uniforms ----
mov rb0, unif           # W_row_base
mov rb1, unif           # y_row_base
mov ra3, unif           # IN_DIM
mov ra5, unif           # QPU_NUM

# ---- per-lane TMU start address ----
# tmu_addr[lane] = W_row_base + lane * IN_DIM * 4
mov r0, elem_num
mul24 r0, r0, ra3       # lane * IN_DIM  (ra3 written 2 instrs ago, 1 gap OK)
shl r0, r0, 2           # * 4 bytes
add ra10, rb0, r0       # per-lane TMU base

# ---- per-QPU VPM/DMA setup (avoid write conflicts between QPUs) ----
nop
mov r0, ra5
shl r0, r0, 7           # QPU_NUM << 7 for VDW Y field
mov r1, vdw_setup_0(1, 16, dma_h32(0, 0))
add ra8, r0, r1

mov r1, vpm_setup(1, 1, h32(0))
add ra7, ra5, r1

# ---- accumulator + pipelined TMU prefetch ----
mov ra20, 0             # acc = 0.0

# Prefetch: issue TMU for k=0, read first x value
mov tmu0_s, ra10
add ra10, ra10, 4
mov r3, unif            # x[0]

# Loop counter: IN_DIM - 1 iterations (k=0 handled by prefetch + epilogue).
# Use accumulator to avoid regfile read-after-write hazard on ra31.
mov r1, ra3             # r1 = IN_DIM
sub.setf r1, r1, 1     # r1 = IN_DIM - 1, set Z if IN_DIM==1
mov ra31, r1            # ra31 = IN_DIM - 1 (safe: not read until loop body, 10+ instrs away)
brr.allz -, :epilogue   # if IN_DIM == 1, skip loop
nop
nop
nop

# ---- inner loop: pipelined TMU fetch ----
# Each iteration: prefetch W for k+1, read TMU result for k, accumulate.
# TMU for k was issued ~11 instructions ago → result ready, no stall at ldtmu0.
:k_loop
    mov tmu0_s, ra10        # prefetch W[lane][k+1]
    add ra10, ra10, 4
    ldtmu0                  # r4 = W[lane][k] from previous request

    fmul r0, r3, r4         # x[k] * W[lane][k]
    fadd ra20, ra20, r0

    mov r3, unif            # x[k+1]
    sub.setf ra31, ra31, 1
    brr.anynz -, :k_loop
    nop
    nop
    nop

# ---- epilogue: process final element (TMU issued in last iter or prefetch) ----
:epilogue
ldtmu0
fmul r0, r3, r4
fadd ra20, ra20, r0

# ---- write 16 results via VPM DMA ----
mov vw_setup, ra7
mov vpm, ra20
mov -, vw_wait

mov vw_setup, ra8
mov vw_addr, rb1
mov -, vw_wait

thrend
mov interrupt, 1
nop
