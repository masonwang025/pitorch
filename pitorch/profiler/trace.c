#include "trace.h"
#include <string.h>

#ifdef __RPI__
#include "rpi.h"
#else
#include <stdio.h>
#include <sys/time.h>
#endif

/* ── portable timer ─────────────────────────────────────────── */

uint32_t pt_time_us(void) {
#ifdef __RPI__
    return timer_get_usec();
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint32_t)(tv.tv_sec * 1000000 + tv.tv_usec);
#endif
}

/* ── core API ───────────────────────────────────────────────── */

void pt_trace_init(pt_trace_t *t) {
    if (!t) return;
    memset(t, 0, sizeof(*t));
    t->enabled = 1;
}

void pt_trace_begin(pt_trace_t *t, const char *name, const char *cat, int layer) {
    if (!t || !t->enabled) return;
    if (t->depth >= PT_TRACE_MAX_DEPTH) return;
    int d = t->depth;
    t->stack[d].name  = name;
    t->stack[d].cat   = cat;
    t->stack[d].ts_us = pt_time_us();
    t->stack[d].layer = (int8_t)layer;
    t->depth++;
}

void pt_trace_end(pt_trace_t *t) {
    if (!t || !t->enabled) return;
    if (t->depth <= 0) return;
    t->depth--;
    uint32_t now = pt_time_us();
    if (t->count >= PT_TRACE_MAX_EVENTS) return;
    pt_trace_event_t *e = &t->events[t->count++];
    int d = t->depth;
    e->name      = t->stack[d].name;
    e->cat       = t->stack[d].cat;
    e->ts_us     = t->stack[d].ts_us;
    e->dur_us    = now - e->ts_us;
    e->layer     = t->stack[d].layer;
    e->depth     = (int8_t)d;
    e->hw_cycles = 0;
    e->hw_stall  = 0;
}

void pt_trace_reset(pt_trace_t *t) {
    if (!t) return;
    t->count = 0;
    t->depth = 0;
}

/* ── JSON output helpers ────────────────────────────────────── */

/*
 * Emit one Chrome Tracing event in {"ph":"X"} (complete) format.
 * If layer >= 0, name becomes "L{layer}_{name}".
 */

#ifndef __RPI__

static void write_event(FILE *f, const pt_trace_event_t *e, int first) {
    if (!first) fprintf(f, ",\n");
    if (e->layer >= 0)
        fprintf(f, "{\"name\":\"L%d_%s\"", e->layer, e->name);
    else
        fprintf(f, "{\"name\":\"%s\"", e->name);
    fprintf(f, ",\"cat\":\"%s\",\"ph\":\"X\",\"ts\":%u,\"dur\":%u,\"pid\":0,\"tid\":0",
            e->cat, e->ts_us, e->dur_us);
    if (e->hw_cycles || e->hw_stall)
        fprintf(f, ",\"args\":{\"qpu_exec\":%u,\"tmu_stall\":%u}",
                e->hw_cycles, e->hw_stall);
    fprintf(f, "}");
}

void pt_trace_write_json(const pt_trace_t *t, const char *path) {
    if (!t || t->count == 0) return;
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "[\n");
    for (int i = 0; i < t->count; i++)
        write_event(f, &t->events[i], i == 0);
    fprintf(f, "\n]\n");
    fclose(f);
}

void pt_trace_write_meta(const char *path, const char *name,
                         const char *model, int dim, int layers,
                         int vocab, const char *device, int num_qpus,
                         int steps, float lr, float final_loss,
                         float total_time_s, const char *notes) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "{\n");
    fprintf(f, "  \"name\": \"%s\",\n", name);
    fprintf(f, "  \"model\": \"%s\",\n", model);
    fprintf(f, "  \"dim\": %d, \"layers\": %d, \"vocab\": %d,\n", dim, layers, vocab);
    fprintf(f, "  \"device\": \"%s\",\n", device);
    fprintf(f, "  \"num_qpus\": %d,\n", num_qpus);
    fprintf(f, "  \"steps\": %d,\n", steps);
    fprintf(f, "  \"lr\": %.6f,\n", lr);
    fprintf(f, "  \"final_loss\": %.6f,\n", final_loss);
    fprintf(f, "  \"total_time_s\": %.1f,\n", total_time_s);
    fprintf(f, "  \"notes\": \"%s\"\n", notes ? notes : "");
    fprintf(f, "}\n");
    fclose(f);
}

#else /* __RPI__ */

static void emit_event_uart(const pt_trace_event_t *e, int first) {
    if (!first) printk(",\n");
    if (e->layer >= 0)
        printk("{\"name\":\"L%d_%s\"", e->layer, e->name);
    else
        printk("{\"name\":\"%s\"", e->name);
    printk(",\"cat\":\"%s\",\"ph\":\"X\",\"ts\":%d,\"dur\":%d,\"pid\":0,\"tid\":0",
           e->cat, e->ts_us, e->dur_us);
    if (e->hw_cycles || e->hw_stall)
        printk(",\"args\":{\"qpu_exec\":%d,\"tmu_stall\":%d}",
               e->hw_cycles, e->hw_stall);
    printk("}");
}

void pt_trace_emit_uart(const pt_trace_t *t) {
    if (!t || t->count == 0) return;
    printk("\n---TRACE_BEGIN---\n[\n");
    for (int i = 0; i < t->count; i++)
        emit_event_uart(&t->events[i], i == 0);
    printk("\n]\n---TRACE_END---\n");
}

void pt_trace_emit_meta(const char *name, const char *model,
                        int dim, int layers, int vocab,
                        int num_qpus, int steps,
                        unsigned total_time_us,
                        const char *notes) {
    printk("\n---META_BEGIN---\n");
    printk("{\"name\":\"%s\",\"model\":\"%s\",", name, model);
    printk("\"dim\":%d,\"layers\":%d,\"vocab\":%d,", dim, layers, vocab);
    printk("\"device\":\"pi_zero\",\"num_qpus\":%d,", num_qpus);
    printk("\"steps\":%d,", steps);
    printk("\"total_time_s\":%d,", total_time_us / 1000000);
    printk("\"notes\":\"%s\"", notes ? notes : "");
    printk("}\n---META_END---\n");
}

#endif
