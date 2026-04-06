#include "rpi.h"
#include "gpu.h"
#include "arena.h"

static gpu_mem_t mem;
static uint32_t total;
static uint32_t used;
static int inited;

void gpu_arena_init(uint32_t total_bytes) {
    demand(!inited, "gpu_arena_init called twice");
    mem = gpu_alloc(total_bytes);
    total = total_bytes;
    used = 0;
    inited = 1;
}

void *gpu_arena_alloc(uint32_t size) {
    demand(inited, "gpu_arena not initialized");
    size = (size + 15u) & ~15u;
    if (used + size > total)
        panic("gpu_arena_alloc: OOM (used=%d req=%d total=%d)", used, size, total);
    void *ptr = (uint8_t *)mem.cpu_ptr + used;
    used += size;
    return ptr;
}

void gpu_arena_reset(void) {
    demand(inited, "gpu_arena not initialized");
    used = 0;
}

void gpu_arena_free(void) {
    demand(inited, "gpu_arena not initialized");
    gpu_free(&mem);
    inited = 0;
    used = 0;
    total = 0;
}

uint32_t gpu_arena_used(void)  { return used; }
uint32_t gpu_arena_total(void) { return total; }
