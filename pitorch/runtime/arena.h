#ifndef PITORCH_ARENA_H
#define PITORCH_ARENA_H

#include <stdint.h>

/* One-shot GPU arena allocator.
 * Sidesteps the 2-alloc firmware bug: calls gpu_alloc() exactly once,
 * then bump-allocates within that block. Reset between layers. */

void  gpu_arena_init(uint32_t total_bytes);
void *gpu_arena_alloc(uint32_t size);   /* 16-byte aligned, returns CPU ptr */
void  gpu_arena_reset(void);            /* rewind bump pointer to base */
void  gpu_arena_free(void);             /* release the underlying GPU block */

uint32_t gpu_arena_used(void);
uint32_t gpu_arena_total(void);

#endif
