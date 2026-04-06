# gemm_multi.qasm — Multi-QPU float GEMM (VPM + uniforms)
#
# Same algorithm as gemm_vpm.qasm but with disjoint row ownership.
# QPU q processes rows q, q+NUM_QPUs, q+2*NUM_QPUs, ...
# Each QPU uses VPM row QPU_NUM to avoid VPM conflicts.
#
# Uniform stream:
#   [0]  B bus address
#   [1]  C bus address
#   [2]  DIM  (M = K; N = 16)
#   [3]  NUM_QPUs
#   [4]  QPU_NUM
#   [5]  NUM_ROWS  (rows assigned to this QPU)
#   [6..]  A values for assigned rows, row-major (float bits)

# ---- read uniforms ----
mov rb0, unif           # B bus address
mov rb1, unif           # C bus address
mov ra3, unif           # DIM
nop
mov r0, ra3
shl rb6, r0, 2          # rb6 = DIM * 4 (row stride in bytes)

mov ra4, unif           # NUM_QPUs
mov ra5, unif           # QPU_NUM
mov ra30, unif          # NUM_ROWS (remaining row counter)

# ---- compute VPM row offsets for this QPU ----

# VDR (DMA read into VPM): Y field at bits [9:4]
nop
mov r0, ra5
shl r0, r0, 4
mov r1, vdr_setup_0(1, 16, 1, vdr_h32(1, 0, 0))
add ra6, r0, r1

# VPM read/write setup: Y field at bits [5:0]
mov r1, vpm_setup(1, 1, h32(0))
add ra7, ra5, r1

# VDW (DMA write from VPM): Y field at bits [13:7]
mov r0, ra5
shl r0, r0, 7
mov r1, vdw_setup_0(1, 16, dma_h32(0, 0))
add ra8, r0, r1

# ---- compute C starting offset and row stride ----

# C_offset = QPU_NUM * row_stride
mov r0, ra5
mov r1, rb6
nop
mul24 ra13, r0, r1

# qpu_c_stride = NUM_QPUs * row_stride
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

# ---- outer loop: over this QPU's assigned rows ----
:row_loop

    mov ra20, 0             # acc = 0.0
    mov ra12, 0             # B row byte-offset
    mov ra31, ra3           # remaining k = DIM

    :k_loop

        mov r3, unif

        mov vr_setup, ra6
        add vr_addr, ra12, rb0
        mov -, vr_wait
        nop

        mov vr_setup, ra7
        nop
        mov r1, vpm

        fmul r0, r3, r1
        nop
        fadd ra20, ra20, r0

        add ra12, ra12, rb6
        sub.setf ra31, ra31, 1

        brr.anynz -, :k_loop
        nop
        nop
        nop

    # ---- write C row ----
    mov vw_setup, ra7
    mov vpm, ra20
    mov -, vw_wait

    mov vw_setup, ra8
    add vw_addr, ra13, rb1
    mov -, vw_wait

    # ---- advance to next assigned row ----
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
