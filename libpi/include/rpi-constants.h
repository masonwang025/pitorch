#ifndef __RPI_CONSTANTS__
#define __RPI_CONSTANTS__
/*
 * we put all the various magic constants here.   otherwise they are buried in different
 * places and it's easy to get conflicts.
 *
 * 
 * should add all the machine constants.
 */

/*
 * Stacks placed near top of ARM RAM (384 MB with gpu_mem=128) so they
 * don't collide with model weights loaded at 0x02000000 via initramfs.
 * Values must be valid ARM immediates (0xNN000000 pattern).
 * Each stack gets 16 MB of headroom (grows downward).
 *
 * gpu_mem=128 → GPU boundary at 0x18000000 (384 MB).
 * Heaviest config: 110M 4-Pi R3 uses ~295 MB, leaving ~21 MB
 * before INT_STACK_ADDR2.
 */
#ifndef STACK_ADDR
#   define STACK_ADDR          0x18000000
#   define STACK_ADDR2         0x17000000
#endif

#ifndef INT_STACK_ADDR
#   define INT_STACK_ADDR      0x16000000
#   define INT_STACK_ADDR2     0x15000000
#endif

#define HIGHEST_USED_ADDR STACK_ADDR

#define MK_FN(fn_name)     \
.globl fn_name;             \
fn_name:

#define CYC_PER_USEC 700
#define PI_MHz  (700*1000*1000UL)

/* from A2-2 */
#define USER_MODE       0b10000
#define FIQ_MODE        0b10001
#define IRQ_MODE        0b10010
#define SUPER_MODE      0b10011
#define ABORT_MODE      0b10111
#define UNDEF_MODE      0b11011
#define SYS_MODE        0b11111


// 1.4285714285714286 nanosecond per cycle: we don't have fp so
// use 142857 and then divide by 100000
// 
#define cycles_to_nanosec(c) (((c) * 142857UL) / 100000UL)

// if we overclock, will have to change this stuff.
#define usec_to_cycles(usec) ((usec) * CYC_PER_USEC)

#endif
