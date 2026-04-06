#ifndef PITORCH_TRACE_H
#define PITORCH_TRACE_H

/*
 * Trace recording for profiling forward/backward passes.
 * Records per-op timing as Chrome Tracing JSON (viewable in Perfetto UI).
 *
 * Usage:
 *   pt_trace_t trace;
 *   pt_trace_init(&trace);
 *   pt_trace_begin(&trace, "embedding", "fwd", -1);
 *   // ... do work ...
 *   pt_trace_end(&trace);
 *   pt_trace_write_json(&trace, "trace.json");  // Mac
 *   pt_trace_emit_uart(&trace);                  // Pi
 *
 * All functions are no-ops when t is NULL or t->enabled is 0.
 */

#include <stdint.h>

#define PT_TRACE_MAX_EVENTS  4096
#define PT_TRACE_MAX_DEPTH   32

typedef struct {
    const char *name;
    const char *cat;       /* "fwd", "bwd", "sgd", "train" */
    uint32_t    ts_us;     /* start timestamp (microseconds) */
    uint32_t    dur_us;    /* duration (microseconds) */
    int8_t      layer;     /* transformer layer index, -1 = N/A */
    int8_t      depth;     /* nesting depth */
    uint32_t    hw_cycles; /* V3D QPU exec cycles (0 on Mac) */
    uint32_t    hw_stall;  /* V3D TMU stall cycles (0 on Mac) */
} pt_trace_event_t;

typedef struct {
    pt_trace_event_t events[PT_TRACE_MAX_EVENTS];
    int              count;
    int              enabled;

    struct {
        const char *name;
        const char *cat;
        uint32_t    ts_us;
        int8_t      layer;
    } stack[PT_TRACE_MAX_DEPTH];
    int              depth;
} pt_trace_t;

/* Initialize trace (zeroes everything, sets enabled=1). */
void pt_trace_init(pt_trace_t *t);

/* Push a named span. layer=-1 for non-layer events. */
void pt_trace_begin(pt_trace_t *t, const char *name, const char *cat, int layer);

/* Pop current span, record the event with measured duration. */
void pt_trace_end(pt_trace_t *t);

/* Discard all recorded events but keep enabled state. */
void pt_trace_reset(pt_trace_t *t);

/* Portable microsecond timestamp. */
uint32_t pt_time_us(void);

#ifndef __RPI__

/* Write Chrome Tracing JSON to a file (Mac only). */
void pt_trace_write_json(const pt_trace_t *t, const char *path);

#else

/*
 * Emit trace as Chrome Tracing JSON over UART, wrapped in sentinels:
 *   ---TRACE_BEGIN---
 *   [...json...]
 *   ---TRACE_END---
 */
void pt_trace_emit_uart(const pt_trace_t *t);

#endif

/*
 * Write meta.json alongside a trace.
 * On Mac: writes to file at path.
 * On Pi: emits over UART between ---META_BEGIN--- / ---META_END---.
 */
#ifndef __RPI__
void pt_trace_write_meta(const char *path, const char *name,
                         const char *model, int dim, int layers,
                         int vocab, const char *device, int num_qpus,
                         int steps, float lr, float final_loss,
                         float total_time_s, const char *notes);
#else
void pt_trace_emit_meta(const char *name, const char *model,
                        int dim, int layers, int vocab,
                        int num_qpus, int steps,
                        unsigned total_time_us,
                        const char *notes);
#endif

#endif
