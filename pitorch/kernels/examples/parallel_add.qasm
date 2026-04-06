mov   ra0, unif # A address
mov   ra1, unif # B address
mov   ra2, unif # C address
mov   ra3, unif # N
mov   ra4, unif # NUM_QPUS
mov   ra5, unif # QPU_NUM

mov r0, 4
shl ra11, r0, 4

mov r0, ra5
shl r0, r0, 1

shl r1, r0, 4
mov r2, vdr_setup_0(1, 16, 1, vdr_h32(1, 0, 0))
add ra20, r1, r2

mov r2, vdr_setup_0(1, 16, 1, vdr_h32(1, 1, 0))
add ra21, r1, r2

mov r1, vpm_setup(1, 1, h32(0))
add ra22, r0, r1

add r2, r0, 1
mov r1, vpm_setup(1, 1, h32(0))
add ra23, r2, r1

mov r1, vpm_setup(1, 1, h32(0))
add ra24, r0, r1

shl r1, r0, 7
mov r2, vdw_setup_0(1, 16, dma_h32(0,0))
add ra25, r1, r2

mov r0, ra4
mov r1, ra11
mul24 ra12, r0, r1

mov r0, ra5
mov r1, ra11
mul24 ra10, r0, r1

nop
nop
nop
nop

:loop

    mov vr_setup, ra20
    mov r0, ra0
    add vr_addr, r0, ra10
    mov -, vr_wait

    mov vr_setup, ra21
    mov r0, ra1
    add vr_addr, r0, ra10
    mov -, vr_wait

    mov vr_setup, ra22
    mov r1, vpm

    mov vr_setup, ra23
    mov r2, vpm

    add r0, r1, r2

    mov vw_setup, ra24
    mov vpm, r0
    mov -, vw_wait

    mov vw_setup, ra25
    mov r0, ra2
    add vw_addr, r0, ra10
    mov -, vw_wait

    mov r0, ra12
    add ra10, ra10, r0

    mov r0, ra3
    shl r0, r0, 2
    nop
    sub.setf r0, ra10, r0
    brr.anyc -, :loop

    nop
    nop
    nop

:end
thrend
mov interrupt, 1
nop
