#ifndef PITORCH_GPU_H
#define PITORCH_GPU_H

#include <stdint.h>

#define GPU_MEM_FLG  0xC
#define GPU_BASE     0x40000000

#define CPU_TO_GPU(addr) (GPU_BASE + (uint32_t)(addr))
#define GPU_TO_CPU(addr) ((uint32_t)(addr) - GPU_BASE)

/* Low-level mailbox property calls (thin wrappers over VideoCore firmware) */
uint32_t gpu_mem_alloc(uint32_t size, uint32_t align, uint32_t flags);
uint32_t gpu_mem_free(uint32_t handle);
uint32_t gpu_mem_lock(uint32_t handle);
uint32_t gpu_mem_unlock(uint32_t handle);
uint32_t gpu_qpu_enable(uint32_t enable);

/* ---- Runtime API ---- */

typedef struct {
    volatile void *cpu_ptr;
    uint32_t bus_addr;
    uint32_t handle;
} gpu_mem_t;

void qpu_enable(void);
void qpu_disable(void);

/* Allocate GPU-visible memory. Returns CPU pointer + bus address. Panics on failure. */
gpu_mem_t gpu_alloc(uint32_t size);
void gpu_free(gpu_mem_t *mem);

/*
 * Dispatch shader code to QPUs via V3D scheduler registers.
 * code:     GPU bus address of the instruction array.
 * uniforms: array of per-QPU GPU uniform-base addresses.
 * num_qpus: number of QPUs to launch.
 */
void qpu_launch(uint32_t code, uint32_t uniforms[], int num_qpus);

#endif
