mov   ra0, unif # HEIGHT
mov   ra1, unif # WIDTH
mov   ra2, unif # NUM_QPUS
mov   ra3, unif # QPU_NUM
mov   ra4, unif # ADDRESS

mov ra10, ra3

:row_loop

mov ra11, 0

shl r1, ra1, 2
mul24 ra12, r1, ra10

:column_loop

    mov r0, ra11
    add r0, r0, elem_num

    mov r1, ra10
    mov r2, ra1

    mul24 r1, r1, r2
    add r1, r1, r0

    mov r2, vpm_setup(1, 1, h32(0))
    add vw_setup, ra3, r2

    mov vpm, r1
    mov -, vw_wait

    shl r1, ra3, 7
    mov r2, vdw_setup_0(1, 16, dma_h32(0,0))
    add vw_setup, r1, r2

    mov r1, ra11
    shl r1, r1, 2
    add r1, ra12, r1
    add vw_addr, ra4, r1
    mov -, vw_wait

    add ra11, ra11, 16

    mov r1, ra1
    sub.setf r1, ra11, r1
    brr.anyc -, :column_loop

    nop
    nop
    nop

    mov r1, ra2
    mov r2, ra10
    add ra10, r1, r2
    mov r1, ra0

    sub.setf r1, ra10, r1
    brr.anyc -, :row_loop

    nop
    nop
    nop

:end
thrend
mov interrupt, 1
nop
