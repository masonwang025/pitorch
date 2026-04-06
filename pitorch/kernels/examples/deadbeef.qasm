mov vw_setup, vpm_setup(4, 1, h32(0))

ldi vpm, 0xdeadbeef
mov -, vw_wait

ldi vpm, 0xbeefdead
mov -, vw_wait

ldi vpm, 0xfaded070
mov -, vw_wait

ldi vpm, 0xfeedface
mov -, vw_wait

ldi vw_setup, vdw_setup_0(4, 16, dma_h32(0,0))
mov r0, unif
mov vw_addr, r0
mov -, vw_wait
nop;  thrend
nop
nop
