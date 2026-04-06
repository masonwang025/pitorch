#ifndef PITORCH_MMU_H
#define PITORCH_MMU_H

/*
 * Minimal flat-mapped MMU for ARM1176JZF-S (BCM2835 / Pi Zero).
 *
 * Creates an identity page table (VA == PA) with:
 *   0x00000000 - 0x1FFFFFFF : SDRAM, write-back cacheable + bufferable
 *   0x20000000 - 0x20FFFFFF : ARM peripherals, device (uncacheable)
 *   0x40000000+             : unmapped
 *
 * Enables MMU + L1 D-cache + L1 I-cache + branch prediction.
 * After this call, all SDRAM accesses go through the 16 KB L1 data cache.
 *
 * GPU coherency: call cache_flush_all() before GPU dispatch and after
 * GPU completion to ensure ARM↔GPU data consistency.
 */
void mmu_init_and_enable(void);

#endif
