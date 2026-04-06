#include "rpi.h"
#include "mmu.h"

/*
 * ARM1176JZF-S first-level section descriptor (1 MB pages):
 *
 *   [31:20] section base address
 *   [19:15] SBZ
 *   [14:12] TEX (type extension)
 *   [11:10] AP (access permission)
 *   [9]     implementation defined (0)
 *   [8:5]   domain
 *   [4]     XN (execute never) — set 0
 *   [3]     C (cacheable)
 *   [2]     B (bufferable)
 *   [1:0]   0b10 = section descriptor
 *
 *   TEX=001, C=1, B=1 → write-back, write-allocate (best for computation)
 *   TEX=000, C=0, B=1 → shared device (peripherals)
 *   TEX=000, C=0, B=0 → strongly ordered
 *   AP=0b11 → full access (read/write, no permission faults)
 */

#define SECTION_DESC       0x2        /* bits [1:0] = 0b10 */
#define AP_FULL            (0x3 << 10) /* AP = 0b11 */
#define DOMAIN_0           (0x0 << 5)

#define TEX_NORMAL_WB_WA   (0x1 << 12) /* TEX=001 */
#define CACHEABLE          (1 << 3)
#define BUFFERABLE         (1 << 2)

#define TEX_DEVICE         (0x0 << 12) /* TEX=000 */

#define MB (1024 * 1024)

/* 16 KB page table, 16-byte aligned (ARM1176 requires 16 KB alignment) */
static unsigned page_table[4096] __attribute__((aligned(16384)));

void mmu_init_and_enable(void) {
    /*
     * Fill page table. Each entry maps 1 MB.
     *   [0x000, 0x200) = 512 MB SDRAM: cacheable, bufferable (write-back)
     *   [0x200, 0x210) = 16 MB peripherals: device memory
     *   [0x210, 0xFFF) = unmapped (fault)
     */
    for (int i = 0; i < 4096; i++)
        page_table[i] = 0;

    /* SDRAM: 0x00000000 - 0x1FFFFFFF (512 entries) */
    for (int i = 0; i < 512; i++) {
        page_table[i] = (i << 20)
                       | SECTION_DESC
                       | AP_FULL
                       | DOMAIN_0
                       | TEX_NORMAL_WB_WA
                       | CACHEABLE
                       | BUFFERABLE;
    }

    /* Peripherals: 0x20000000 - 0x20FFFFFF (16 entries)
     * Strongly ordered (C=0, B=0): no write buffering.
     * Required for GPIO handshake — buffered writes cause stale reads
     * during direction switches (send→recv mode transitions). */
    for (int i = 0x200; i < 0x210; i++) {
        page_table[i] = (i << 20)
                       | SECTION_DESC
                       | AP_FULL
                       | DOMAIN_0
                       | TEX_DEVICE;    /* C=0, B=0 → strongly ordered */
    }

    /* Invalidate caches and TLB before enabling MMU */
    unsigned zero = 0;
    asm volatile("mcr p15, 0, %0, c7, c7, 0" :: "r"(zero));  /* invalidate I+D cache */
    asm volatile("mcr p15, 0, %0, c8, c7, 0" :: "r"(zero));  /* invalidate TLB */
    asm volatile("mcr p15, 0, %0, c7, c10, 4" :: "r"(zero)); /* DSB */

    /* Set TTBR0 (translation table base register 0) */
    unsigned ttbr = (unsigned)page_table;
    asm volatile("mcr p15, 0, %0, c2, c0, 0" :: "r"(ttbr));

    /* Domain access control: domain 0 = client (use AP bits) */
    unsigned dacr = 0x1; /* domain 0 = client */
    asm volatile("mcr p15, 0, %0, c3, c0, 0" :: "r"(dacr));

    /* Enable MMU + D-cache + I-cache + branch prediction */
    unsigned ctrl;
    asm volatile("mrc p15, 0, %0, c1, c0, 0" : "=r"(ctrl));
    ctrl |= (1 << 0);   /* M: MMU enable */
    ctrl |= (1 << 2);   /* C: D-cache enable */
    ctrl |= (1 << 11);  /* Z: branch prediction */
    ctrl |= (1 << 12);  /* I: I-cache enable */
    asm volatile("mcr p15, 0, %0, c1, c0, 0" :: "r"(ctrl));

    /* Barrier to ensure MMU is active before proceeding */
    asm volatile("mcr p15, 0, %0, c7, c5, 4" :: "r"(zero));  /* prefetch flush */
    asm volatile("mcr p15, 0, %0, c7, c10, 4" :: "r"(zero)); /* DSB */
    asm volatile("mcr p15, 0, %0, c7, c10, 5" :: "r"(zero)); /* DMB */
}
