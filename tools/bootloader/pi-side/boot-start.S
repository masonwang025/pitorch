@ Bootloader entry point.
@ Places a 2MB gap after _start so received code at 0x8000
@ won't collide with bootloader code.

.section ".text.boot"

.globl _start
_start:
    b skip
.space 0x200000-0x8004,0
skip:
    mov sp,#0x01000000
    bl _cstart
hang: b rpi_reboot

.globl put32
.globl PUT32
put32:
PUT32:
    str r1,[r0]
    bx lr

.globl get32
.globl GET32
get32:
GET32:
    ldr r0,[r0]
    bx lr

.globl BRANCHTO
BRANCHTO:
    bx r0

.globl put8
.globl PUT8
put8:
PUT8:
    strb r1,[r0]
    bx lr
