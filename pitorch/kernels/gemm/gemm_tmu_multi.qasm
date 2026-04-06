# gemm_tmu_multi.qasm — Multi-QPU float GEMM via TMU
#
# Same algorithm as gemm_tmu.qasm, row-striped across QPUs.
# QPU q handles rows q, q+NUM_QPUs, q+2*NUM_QPUs, ...
# Per-QPU VPM rows avoid DMA write conflicts.
#
# Uniform stream:
#   [0] A_base     (bus address)
#   [1] B_tile     (bus address of B[0][col_tile])
#   [2] C_tile     (bus address of C[0][col_tile])
#   [3] M          (unused in kernel, kept for debug)
#   [4] K          (inner dimension, must be >= 1)
#   [5] stride     (N * 4)
#   [6] NUM_QPUs
#   [7] QPU_NUM
#   [8] NUM_ROWS   (rows assigned to this QPU)
#
# Register map:
#   ra0  = A_base           ra1  = C_tile
#   ra3  = M (debug)        ra4  = K
#   ra5  = stride           ra6  = NUM_QPUs
#   ra7  = QPU_NUM          ra8  = elem_num * 4
#   ra10 = a_ptr            ra14 = vpm_setup (per-QPU)
#   ra15 = vdw_setup (per-QPU)
#   ra20 = accumulator      ra30 = remaining rows
#   ra31 = k loop counter
#   rb0  = B_tile           rb1  = (scratch)
#   rb3  = C row offset     rb6  = K * 4
#   rb8  = A skip = (NUM_QPUs - 1) * K * 4
#   rb9  = C skip = NUM_QPUs * stride
#   rb10 = b_row_ptr

# ---- read uniforms ----
mov ra0, unif           # A_base
mov rb0, unif           # B_tile
mov ra1, unif           # C_tile
mov ra3, unif           # M
mov ra4, unif           # K
mov ra5, unif           # stride
mov ra6, unif           # NUM_QPUs
mov ra7, unif           # QPU_NUM
mov ra30, unif          # NUM_ROWS → doubles as remaining-rows counter

# ---- derived constants ----
nop
mov r0, ra4
shl rb6, r0, 2          # K * 4

mov r0, elem_num
shl ra8, r0, 2          # lane * 4

# A skip between assigned rows: (NUM_QPUs - 1) * K * 4
nop
mov r0, ra6
sub r0, r0, 1
mov r1, rb6
nop
mul24 rb8, r0, r1

# C row stride for this QPU: NUM_QPUs * stride
nop
mov r0, ra6
mov r1, ra5
nop
mul24 rb9, r0, r1

# ---- per-QPU VPM row setup ----
nop
mov r0, ra7             # QPU_NUM

# VPM write setup: vpm_setup base + QPU_NUM
mov r1, vpm_setup(1, 1, h32(0))
add ra14, r0, r1

# VDW setup: dma_h32 Y-field is at bits [13:7], so shift QPU_NUM << 7
shl r0, r0, 7
mov r1, vdw_setup_0(1, 16, dma_h32(0, 0))
add ra15, r0, r1

# ---- initial offsets for this QPU ----
# a_start = A_base + QPU_NUM * K * 4
nop
mov r0, ra7
mov r1, rb6
nop
mul24 r0, r0, r1
add ra10, ra0, r0

# c_start_offset = QPU_NUM * stride
nop
mov r0, ra7
mov r1, ra5
nop
mul24 rb3, r0, r1

# ---- guard: skip if no rows assigned ----
mov.setf -, ra30
brr.allz -, :end
nop
nop
nop

:row_loop

    mov ra20, 0             # acc = 0.0
    mov rb10, rb0           # b_row_ptr = B_tile

    # ---- prolog: submit first TMU requests ----
    mov tmu0_s, ra10
    add r0, rb10, ra8
    mov tmu0_s, r0
    add ra10, ra10, 4
    add rb10, rb10, ra5

    mov r0, ra4
    sub.setf ra31, r0, 1
    brr.allz -, :k_epilog
    nop
    nop
    nop

    :k_loop
        mov tmu0_s, ra10
        add r0, rb10, ra8
        mov tmu0_s, r0

        ldtmu0
        mov r2, r4
        ldtmu0
        fmul r0, r2, r4
        fadd ra20, ra20, r0

        add ra10, ra10, 4
        add rb10, rb10, ra5

        sub.setf ra31, ra31, 1
        brr.anynz -, :k_loop
        nop
        nop
        nop

    :k_epilog
    ldtmu0
    mov r2, r4
    ldtmu0
    fmul r0, r2, r4
    fadd ra20, ra20, r0

    # ---- write C via per-QPU VPM row ----
    mov vw_setup, ra14
    mov vpm, ra20
    mov -, vw_wait

    mov vw_setup, ra15
    add vw_addr, rb3, ra1
    mov -, vw_wait

    # advance a_ptr past the (NUM_QPUs - 1) rows we skip
    add ra10, ra10, rb8

    # advance C offset (rb3 and rb9 both regfile B → use temp)
    mov r0, rb9
    add rb3, rb3, r0

    sub.setf ra30, ra30, 1
    brr.anynz -, :row_loop
    nop
    nop
    nop

:end
thrend
mov interrupt, 1
nop
