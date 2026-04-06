#ifndef PITORCH_PROFILER_H
#define PITORCH_PROFILER_H

#include <stdint.h>

typedef struct {
    uint32_t wall_us;
    uint32_t qpu_exec_cyc;
    uint32_t qpu_idle_cyc;
    uint32_t qpu_tmu_stall;
    uint32_t qpu_vpm_stall;
} perf_t;

/*
 * Configure V3D performance counter sources.
 * Call once after qpu_enable().
 */
void perf_init(void);

/* Clear and enable counters, start wall-clock timer. */
void perf_start(void);

/* Stop counters, return snapshot. */
perf_t perf_stop(void);

/* Pretty-print a perf_t over UART.  label is e.g. "cpu_ref" or "gpu_vpm". */
void perf_print(const char *label, unsigned M, unsigned K, unsigned N,
                const perf_t *p);

#endif
