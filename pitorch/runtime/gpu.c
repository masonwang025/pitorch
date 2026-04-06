#include <stdint.h>
#include "rpi.h"
#include "mailbox.h"
#include "gpu.h"

/* libpi assembly: clean+invalidate D-cache, invalidate I-cache+TLB+BTB */
void cache_flush_all(void);

/* ---- V3D register addresses ---- */

#define V3D_BASE    0x20C00000
#define V3D_L2CACTL (V3D_BASE + 0x020)
#define V3D_SLCACTL (V3D_BASE + 0x024)
#define V3D_SRQSC  (V3D_BASE + 0x418)
#define V3D_SRQPC  (V3D_BASE + 0x430)
#define V3D_SRQUA  (V3D_BASE + 0x434)
#define V3D_SRQCS  (V3D_BASE + 0x43c)
#define V3D_DBCFG  (V3D_BASE + 0xe00)
#define V3D_DBQITE (V3D_BASE + 0xe2c)
#define V3D_DBQITC (V3D_BASE + 0xe30)

/* ---- GPU memory management (mailbox property calls) ---- */

uint32_t gpu_mem_alloc(uint32_t size, uint32_t align, uint32_t flags) {
    uint32_t p[9] __attribute__((aligned(16))) = {
        9 * sizeof(uint32_t),
        0x00000000,
        0x3000c,
        3 * sizeof(uint32_t),
        3 * sizeof(uint32_t),
        size, align, flags,
        0
    };
    assert(mbox_property(p));
    return p[5];
}

uint32_t gpu_mem_free(uint32_t handle) {
    uint32_t p[7] __attribute__((aligned(16))) = {
        7 * sizeof(uint32_t),
        0x00000000,
        0x3000f,
        1 * sizeof(uint32_t),
        1 * sizeof(uint32_t),
        handle,
        0
    };
    assert(mbox_property(p));
    return p[5];
}

uint32_t gpu_mem_lock(uint32_t handle) {
    uint32_t p[7] __attribute__((aligned(16))) = {
        7 * sizeof(uint32_t),
        0x00000000,
        0x3000d,
        1 * sizeof(uint32_t),
        1 * sizeof(uint32_t),
        handle,
        0
    };
    assert(mbox_property(p));
    return p[5];
}

uint32_t gpu_mem_unlock(uint32_t handle) {
    uint32_t p[7] __attribute__((aligned(16))) = {
        7 * sizeof(uint32_t),
        0x00000000,
        0x3000e,
        1 * sizeof(uint32_t),
        1 * sizeof(uint32_t),
        handle,
        0
    };
    assert(mbox_property(p));
    return p[5];
}

uint32_t gpu_qpu_enable(uint32_t enable) {
    uint32_t p[7] __attribute__((aligned(16))) = {
        7 * sizeof(uint32_t),
        0x00000000,
        0x30012,
        1 * sizeof(uint32_t),
        1 * sizeof(uint32_t),
        enable,
        0
    };
    assert(mbox_property(p));
    return p[5];
}

/* ---- Runtime API ---- */

void qpu_enable(void) {
    if (gpu_qpu_enable(1))
        panic("qpu_enable failed");
}

void qpu_disable(void) {
    gpu_qpu_enable(0);
}

gpu_mem_t gpu_alloc(uint32_t size) {
    uint32_t handle = gpu_mem_alloc(size, 4096, GPU_MEM_FLG);
    if (!handle)
        panic("gpu_alloc failed (size=%d)", size);
    uint32_t bus = gpu_mem_lock(handle);
    return (gpu_mem_t){
        .cpu_ptr  = (volatile void *)GPU_TO_CPU(bus),
        .bus_addr = bus,
        .handle   = handle
    };
}

void gpu_free(gpu_mem_t *mem) {
    gpu_mem_unlock(mem->handle);
    gpu_mem_free(mem->handle);
    mem->cpu_ptr = NULL;
}

/* QPU dispatch via V3D registers (see Broadcom docs p. 89-91) */

void qpu_launch(uint32_t code, uint32_t uniforms[], int num_qpus) {
    /*
     * ARM D-cache coherency: clean+invalidate so GPU sees ARM-written data
     * (uniforms, shader code, weight matrices updated by SGD).
     * No-op if D-cache is disabled (flat memory, no dirty lines).
     */
    cache_flush_all();

    PUT32(V3D_DBCFG, 0);
    PUT32(V3D_DBQITE, 0);
    PUT32(V3D_DBQITC, -1);

    PUT32(V3D_L2CACTL, 1 << 2);
    PUT32(V3D_SLCACTL, -1);

    PUT32(V3D_SRQCS, (1 << 7) | (1 << 8) | (1 << 16));

    for (unsigned q = 0; q < num_qpus; q++) {
        PUT32(V3D_SRQUA, uniforms[q]);
        PUT32(V3D_SRQPC, code);
    }

    while (((GET32(V3D_SRQCS) >> 16) & 0xff) != num_qpus)
        ;
}
