# gemm_vpm.qasm — Single-QPU float GEMM (VPM + uniforms)
#
# C[M][16] = A[M][K] * B[K][16],  float32,  M = K = DIM.
# SIMD lane j computes C[i][j], outer loop over rows i.
#
# Uniform stream:
#   [0]  B bus address
#   [1]  C bus address
#   [2]  DIM  (M = K; N is always 16)
#   [3 .. 3+DIM*DIM-1]  A values row-major (float bits)
#
# VPM row 0 is the scratch row for both B reads and C writes.

# ---- read uniforms (addresses in regfile B to avoid read-port conflicts) ----
mov rb0, unif           # B bus address
mov rb1, unif           # C bus address
mov ra3, unif           # DIM

# ---- precompute constants ----
nop                     # 1-cycle gap: ra3 write-to-read hazard
mov r0, ra3
shl rb6, r0, 2          # rb6 = DIM * 4 (row stride in bytes)

# ---- outer loop: rows i = 0 .. DIM-1 ----
mov ra13, 0             # C row byte-offset  (i * DIM * 4)
mov ra30, ra3           # remaining rows = DIM

:row_loop

    # acc = 0.0 in all 16 lanes  (IEEE-754 zero = integer zero)
    mov ra20, 0
    mov ra12, 0             # B row byte-offset  (k * DIM * 4)
    mov ra31, ra3           # remaining k iterations = DIM

    :k_loop

        # --- A[i][k]: next uniform (broadcast to all lanes) ---
        mov r3, unif

        # --- VPM DMA read: B[k][0..15] into VPM row 0 ---
        mov vr_setup, vdr_setup_0(1, 16, 1, vdr_h32(1, 0, 0))
        add vr_addr, ra12, rb0
        mov -, vr_wait
        nop

        # --- read B row from VPM ---
        mov vr_setup, vpm_setup(1, 1, h32(0))
        nop
        mov r1, vpm

        # --- acc += A[i][k] * B[k][lane] ---
        fmul r0, r3, r1
        nop
        fadd ra20, ra20, r0

        # --- advance B offset, decrement remaining ---
        add ra12, ra12, rb6
        sub.setf ra31, ra31, 1

        # --- loop while remaining > 0 ---
        brr.anynz -, :k_loop
        nop
        nop
        nop

    # ---- write C row i ----
    mov vw_setup, vpm_setup(1, 1, h32(0))
    mov vpm, ra20
    mov -, vw_wait

    mov vw_setup, vdw_setup_0(1, 16, dma_h32(0, 0))
    add vw_addr, ra13, rb1
    mov -, vw_wait

    # ---- advance C offset, decrement remaining rows ----
    add ra13, ra13, rb6
    sub.setf ra30, ra30, 1

    # ---- loop while remaining rows > 0 ----
    brr.anynz -, :row_loop
    nop
    nop
    nop

:end
thrend
mov interrupt, 1
nop
