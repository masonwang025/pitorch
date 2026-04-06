#include "rpi.h"
#include <string.h>
#include "gpu.h"
#include "index_shader.h"

#define WIDTH    64
#define HEIGHT   32
#define NUM_QPUS 16
#define NUM_UNIFS 5

struct index_data {
    uint32_t output[HEIGHT][WIDTH];
    uint32_t code[sizeof(index_shader) / sizeof(uint32_t)];
    uint32_t unif[NUM_QPUS][NUM_UNIFS];
    uint32_t unif_ptr[NUM_QPUS];
};

void notmain(void) {
    qpu_enable();
    gpu_mem_t mem = gpu_alloc(sizeof(struct index_data));
    volatile struct index_data *g = mem.cpu_ptr;

    memcpy((void *)g->code, index_shader, sizeof(g->code));

    for (int i = 0; i < NUM_QPUS; i++) {
        g->unif[i][0] = HEIGHT;
        g->unif[i][1] = WIDTH;
        g->unif[i][2] = NUM_QPUS;
        g->unif[i][3] = i;
        g->unif[i][4] = CPU_TO_GPU(&g->output);
        g->unif_ptr[i] = CPU_TO_GPU(&g->unif[i]);
    }
    for (int i = 0; i < HEIGHT; i++)
        for (int j = 0; j < WIDTH; j++)
            g->output[i][j] = 0;

    printk("Running code on GPU...\n");

    int start_time = timer_get_usec();
    qpu_launch(CPU_TO_GPU(&g->code), (uint32_t *)g->unif_ptr, NUM_QPUS);
    int end_time = timer_get_usec();

    printk("DONE!\n");
    printk("Time taken on GPU: %d us\n", end_time - start_time);

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            if (g->output[i][j] != (unsigned)(i * WIDTH + j)) {
                printk("ERROR: gpu->output[%d][%d] = %d\n", i, j, g->output[i][j]);
            } else if (i * 4 % HEIGHT == 0 && j * 4 % WIDTH == 0) {
                printk("CORRECT: gpu->output[%d][%d] = %d (%d * %d + %d)\n",
                       i, j, g->output[i][j], i, WIDTH, j);
            }
        }
    }

    gpu_free(&mem);
    qpu_disable();
}
