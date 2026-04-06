#include <stdint.h>
#include "rpi.h"
#include "profiler.h"

/* V3D performance-counter registers (same V3D_BASE as runtime/gpu.c) */
#define V3D_BASE     0x20C00000
#define V3D_PCTRC    (V3D_BASE + 0x670)
#define V3D_PCTRE    (V3D_BASE + 0x674)
#define V3D_PCTR(n)  (V3D_BASE + 0x680 + 8*(n))
#define V3D_PCTRS(n) (V3D_BASE + 0x684 + 8*(n))

/* Counter source IDs (Broadcom V3D arch-ref, table 35) */
#define SRC_QPU_EXEC   19   /* clock cycles executing instructions  */
#define SRC_QPU_IDLE   13   /* clock cycles idle                    */
#define SRC_QPU_TMU    16   /* clock cycles stalled on TMU          */
#define SRC_QPU_VPM    18   /* clock cycles stalled on VPM          */

static uint32_t start_us;

void perf_init(void) {
    PUT32(V3D_PCTRS(0), SRC_QPU_EXEC);
    PUT32(V3D_PCTRS(1), SRC_QPU_IDLE);
    PUT32(V3D_PCTRS(2), SRC_QPU_TMU);
    PUT32(V3D_PCTRS(3), SRC_QPU_VPM);
}

void perf_start(void) {
    PUT32(V3D_PCTRC, 0xF);
    PUT32(V3D_PCTRE, 0xF);
    start_us = timer_get_usec();
}

perf_t perf_stop(void) {
    uint32_t end_us = timer_get_usec();
    PUT32(V3D_PCTRE, 0);
    return (perf_t){
        .wall_us       = end_us - start_us,
        .qpu_exec_cyc  = GET32(V3D_PCTR(0)),
        .qpu_idle_cyc  = GET32(V3D_PCTR(1)),
        .qpu_tmu_stall = GET32(V3D_PCTR(2)),
        .qpu_vpm_stall = GET32(V3D_PCTR(3)),
    };
}

void perf_print(const char *label, unsigned M, unsigned K, unsigned N,
                const perf_t *p) {
    printk("[%s] %dx%dx%d  %d us", label, M, K, N, p->wall_us);

    if (p->qpu_exec_cyc | p->qpu_idle_cyc |
        p->qpu_tmu_stall | p->qpu_vpm_stall) {
        printk("  |  exec %d  idle %d  tmu %d  vpm %d",
               p->qpu_exec_cyc, p->qpu_idle_cyc,
               p->qpu_tmu_stall, p->qpu_vpm_stall);
    }
    printk("\n");
}
