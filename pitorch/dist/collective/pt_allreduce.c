#include <string.h>
#ifdef __RPI__
#include "rpi.h"
#else
#include <stdio.h>
#define printk printf
#endif
#include "pt_allreduce.h"

/*
 * 2-rank allreduce: reduce-scatter + allgather.
 *
 * Split the buffer in half. Each rank "owns" one half:
 *   - Rank 0 owns the first half, rank 1 owns the second half.
 *
 * Phase 1 (reduce-scatter): each rank sends its non-owned half to the peer.
 *   The peer adds the received values into its owned half, then divides by 2.
 *
 * Phase 2 (allgather): each rank sends its owned (now-averaged) half to the peer.
 *   The peer replaces its non-owned half with the received values.
 *
 * Result: both ranks hold identical averaged buffers.
 *
 * All transfers are raw bytes (no proto framing) to minimize overhead.
 * The caller must ensure both ranks are synchronized before calling.
 */

/* Send a chunk of floats as raw bytes */
static void send_floats(pt_transport_t *t, const float *data, int n) {
    t->send_raw(t, data, n * sizeof(float));
}

/* Receive a chunk of floats as raw bytes */
static int recv_floats(pt_transport_t *t, float *data, int n) {
    return t->recv_raw(t, data, n * sizeof(float));
}

/* Receive floats into a temp buffer, add into dst, divide by 2 */
static int recv_and_reduce(pt_transport_t *t, float *dst, float *tmp, int n) {
    if (recv_floats(t, tmp, n) < 0)
        return -1;
    for (int i = 0; i < n; i++)
        dst[i] = (dst[i] + tmp[i]) * 0.5f;
    return 0;
}

/* Progress: print every PROGRESS_INTERVAL floats */
#define PROGRESS_INTERVAL  (1024 * 1024)  /* every 1M floats = 4MB */

int pt_allreduce_avg(int rank, float *buf, int n_floats,
                     pt_transport_t *transport) {
    int half = n_floats / 2;
    int second_half = n_floats - half;  /* handles odd n_floats */

    float *first  = buf;
    float *second = buf + half;

    /* Temp buffer for receiving chunks — on stack, chunked */
    int chunk = PT_ALLREDUCE_CHUNK / sizeof(float);  /* floats per chunk */
    float tmp[PT_ALLREDUCE_CHUNK / sizeof(float)];
    int progress = 0;
    int total_mb = n_floats * 4 / (1024 * 1024);

    printk("[ar:p1");
    /*
     * Phase 1: reduce-scatter.
     * Rank 0 sends its second half → rank 1 reduces into its second half.
     * Rank 1 sends its first half  → rank 0 reduces into its first half.
     */
    if (rank == 0) {
        /* Send my second half to rank 1 — single call, no chunking */
        send_floats(transport, second, second_half);
        printk("S|");
        /* Receive rank 1's first half, reduce into my first half */
        progress = 0;
        for (int off = 0; off < half; off += chunk) {
            int n = (off + chunk <= half) ? chunk : half - off;
            if (recv_and_reduce(transport, first + off, tmp, n) < 0)
                return -1;
            progress += n;
            if (progress >= PROGRESS_INTERVAL) {
                printk(".");
                progress -= PROGRESS_INTERVAL;
            }
        }
    } else {
        /* Receive rank 0's second half, reduce into my second half */
        progress = 0;
        for (int off = 0; off < second_half; off += chunk) {
            int n = (off + chunk <= second_half) ? chunk : second_half - off;
            if (recv_and_reduce(transport, second + off, tmp, n) < 0)
                return -1;
            progress += n;
            if (progress >= PROGRESS_INTERVAL) {
                printk(".");
                progress -= PROGRESS_INTERVAL;
            }
        }
        printk("R|");
        /* Send my first half to rank 0 — single call */
        send_floats(transport, first, half);
    }

    printk("]");
    printk("[ar:p2");
    /*
     * Phase 2: allgather.
     * Rank 0 sends its averaged first half → rank 1 replaces.
     * Rank 1 sends its averaged second half → rank 0 replaces.
     */
    if (rank == 0) {
        /* Send my averaged first half to rank 1 */
        progress = 0;
        for (int off = 0; off < half; off += chunk) {
            int n = (off + chunk <= half) ? chunk : half - off;
            send_floats(transport, first + off, n);
            progress += n;
            if (progress >= PROGRESS_INTERVAL) {
                printk(".");
                progress -= PROGRESS_INTERVAL;
            }
        }
        printk("|");
        /* Receive rank 1's averaged second half, replace mine */
        progress = 0;
        for (int off = 0; off < second_half; off += chunk) {
            int n = (off + chunk <= second_half) ? chunk : second_half - off;
            if (recv_floats(transport, second + off, n) < 0)
                return -1;
            progress += n;
            if (progress >= PROGRESS_INTERVAL) {
                printk(".");
                progress -= PROGRESS_INTERVAL;
            }
        }
    } else {
        /* Receive rank 0's averaged first half, replace mine */
        progress = 0;
        for (int off = 0; off < half; off += chunk) {
            int n = (off + chunk <= half) ? chunk : half - off;
            if (recv_floats(transport, first + off, n) < 0)
                return -1;
            progress += n;
            if (progress >= PROGRESS_INTERVAL) {
                printk(".");
                progress -= PROGRESS_INTERVAL;
            }
        }
        printk("|");
        /* Send my averaged second half to rank 0 */
        progress = 0;
        for (int off = 0; off < second_half; off += chunk) {
            int n = (off + chunk <= second_half) ? chunk : second_half - off;
            send_floats(transport, second + off, n);
            progress += n;
            if (progress >= PROGRESS_INTERVAL) {
                printk(".");
                progress -= PROGRESS_INTERVAL;
            }
        }
    }

    printk("]\n");
    return 0;
}
