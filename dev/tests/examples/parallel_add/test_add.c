#include "rpi.h"
#include <string.h>
#include "gpu.h"
#include "add_shader.h"

#define N         1048576
#define NUM_UNIFS 6
#define NUM_QPUS  8

struct add_data {
    uint32_t A[N];
    uint32_t B[N];
    uint32_t C[N];
    uint32_t code[sizeof(add_shader) / sizeof(uint32_t)];
    uint32_t unif[NUM_QPUS][NUM_UNIFS];
    uint32_t unif_ptr[NUM_QPUS];
};

void notmain(void) {
    printk("Testing addition on GPU...\n");

    qpu_enable();
    gpu_mem_t mem = gpu_alloc(sizeof(struct add_data));
    volatile struct add_data *g = mem.cpu_ptr;

    memcpy((void *)g->code, add_shader, sizeof(g->code));

    for (int i = 0; i < NUM_QPUS; i++) {
        g->unif[i][0] = CPU_TO_GPU(&g->A);
        g->unif[i][1] = CPU_TO_GPU(&g->B);
        g->unif[i][2] = CPU_TO_GPU(&g->C);
        g->unif[i][3] = N;
        g->unif[i][4] = NUM_QPUS;
        g->unif[i][5] = i;
        g->unif_ptr[i] = CPU_TO_GPU(&g->unif[i]);
    }

    for (int i = 0; i < N; i++) {
        g->A[i] = 32 + i;
        g->B[i] = 64 + i;
        g->C[i] = 0;
    }

    printk("\nMemory before addition: %x %x %x %x\n",
           g->C[0], g->C[1], g->C[2], g->C[3]);

    int start_time = timer_get_usec();
    qpu_launch(CPU_TO_GPU(&g->code), (uint32_t *)g->unif_ptr, NUM_QPUS);
    int gpu_add_time = timer_get_usec() - start_time;

    printk("Memory after addition:  %d %d %d %d\n",
           g->C[0], g->C[1], g->C[2], g->C[3]);

    for (int i = 0; i < N; i++) {
        if (g->C[i] != (unsigned)((32 + i) + (64 + i))) {
            panic("Add Iteration %d: %d + %d = %d. INCORRECT",
                  i, g->A[i], g->B[i], g->C[i]);
        } else if (i * 16 % N == 0) {
            printk("Add Iteration %d: %d + %d = %d. CORRECT\n",
                  i, g->A[i], g->B[i], g->C[i]);
        }
    }

    start_time = timer_get_usec();
    for (int i = 0; i < N; i++)
        g->C[i] = g->A[i] + g->B[i];
    int cpu_add_time = timer_get_usec() - start_time;

    printk("CPU Addition Time: %d us\n", cpu_add_time);
    printk("GPU Addition Time: %d us\n", gpu_add_time);
    printk("Speedup: %dx\n", cpu_add_time / gpu_add_time);

    gpu_free(&mem);
    qpu_disable();
}
