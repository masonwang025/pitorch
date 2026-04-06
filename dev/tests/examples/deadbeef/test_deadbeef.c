#include "rpi.h"
#include <string.h>
#include "gpu.h"
#include "deadbeef_shader.h"

struct deadbeef_data {
    uint32_t output[64];
    uint32_t code[sizeof(deadbeef_shader) / sizeof(uint32_t)];
    uint32_t unif[1];
    uint32_t unif_ptr[1];
};

void notmain(void) {
    printk("Testing GPU DMA writes...\n");

    qpu_enable();
    gpu_mem_t mem = gpu_alloc(sizeof(struct deadbeef_data));
    volatile struct deadbeef_data *g = mem.cpu_ptr;

    memcpy((void *)g->code, deadbeef_shader, sizeof(g->code));
    g->unif[0] = CPU_TO_GPU(&g->output);
    g->unif_ptr[0] = CPU_TO_GPU(&g->unif);

    memset((void *)g->output, 0xff, sizeof(g->output));

    printk("Memory before running code: %x %x %x %x\n",
           g->output[0], g->output[16], g->output[32], g->output[48]);

    qpu_launch(CPU_TO_GPU(&g->code), (uint32_t *)g->unif_ptr, 1);

    printk("Memory after running code:  %x %x %x %x\n",
           g->output[0], g->output[16], g->output[32], g->output[48]);

    gpu_free(&mem);
    qpu_disable();
}
