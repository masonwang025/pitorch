# gemm_tmu.qasm — Multi-QPU float GEMM: A via uniforms, B via TMU
#
# C[M][16] = A[M][K] * B[K][16], row-major.
# Works for any num_qpus (1 through 12).
# QPU q handles rows q, q+NUM_QPUs, q+2*NUM_QPUs, ...
# Per-QPU VPM rows avoid DMA write conflicts.
#
# Uniform stream (per-QPU):
#   [0]  B bus address
#   [1]  C bus address
#   [2]  DIM  (M = K; N = 16 per tile)
#   [3]  NUM_QPUs
#   [4]  QPU_NUM
#   [5]  NUM_ROWS  (rows assigned to this QPU)
#   [6..]  A values for assigned rows, row-major (float bits)

# ---- read uniforms ----
mov rb0, unif           # B_base
mov rb1, unif           # C_base
mov ra3, unif           # DIM
nop
mov r0, ra3
shl rb6, r0, 2          # DIM * 4 (row stride bytes)

mov ra4, unif           # NUM_QPUs
mov ra5, unif           # QPU_NUM
mov ra30, unif          # NUM_ROWS (remaining row counter)

# ---- per-QPU VPM row offsets (same pattern as gemm_multi.qasm) ----

# VDW (DMA write from VPM): Y at bits [13:7]
nop
mov r0, ra5
shl r0, r0, 7
mov r1, vdw_setup_0(1, 16, dma_h32(0, 0))
add ra8, r0, r1

# VPM write setup: Y at bits [5:0]
mov r1, vpm_setup(1, 1, h32(0))
add ra7, ra5, r1

# ---- C starting offset and row stride ----

# C_offset = QPU_NUM * DIM * 4
mov r0, ra5
mov r1, rb6
nop
mul24 ra13, r0, r1

# C_stride = NUM_QPUs * DIM * 4
mov r0, ra4
mov r1, rb6
nop
mul24 rb7, r0, r1

# ---- guard: skip if NUM_ROWS == 0 ----
mov.setf -, ra30
brr.allz -, :end
nop
nop
nop

# ---- outer loop: over assigned rows ----
:row_loop

    mov ra20, 0             # acc = 0.0
    mov ra12, 0             # B row byte offset
    mov ra31, ra3           # remaining k = DIM

    :k_loop

        # A[i][k] from uniform (broadcast)
        mov r3, unif

        # B[k][lane] via TMU (per-lane address)
        mov r0, elem_num
        shl r0, r0, 2
        add r0, ra12, r0        # B_row_offset + lane*4
        add r0, rb0, r0         # B_base + above
        mov tmu0_s, r0
        nop
        nop
        nop
        nop
        ldtmu0

        # acc += A * B
        fmul r0, r3, r4
        fadd ra20, ra20, r0

        # advance B row offset
        add ra12, ra12, rb6

        sub.setf ra31, ra31, 1
        brr.anynz -, :k_loop
        nop
        nop
        nop

    # ---- write C row via per-QPU VPM row ----
    mov vw_setup, ra7
    mov vpm, ra20
    mov -, vw_wait

    mov vw_setup, ra8
    add vw_addr, ra13, rb1
    mov -, vw_wait

    # advance to next assigned row
    add ra13, ra13, rb7

    sub.setf ra30, ra30, 1
    brr.anynz -, :row_loop
    nop
    nop
    nop

:end
thrend
mov interrupt, 1
nop
