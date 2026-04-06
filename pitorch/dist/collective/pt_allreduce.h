#ifndef PT_ALLREDUCE_H
#define PT_ALLREDUCE_H

#include "pt_transport.h"

/*
 * 2-rank allreduce for gradient averaging.
 *
 * After calling, both ranks hold the element-wise average:
 *   buf[i] = (rank0_buf[i] + rank1_buf[i]) / 2.0
 *
 * Protocol (half-duplex, rank 0 initiates):
 *   1. Rank 0 sends first half   → Rank 1 adds into its first half
 *   2. Rank 1 sends second half  → Rank 0 adds into its second half
 *   3. Rank 0 sends its averaged second half → Rank 1 replaces
 *   4. Rank 1 sends its averaged first half  → Rank 0 replaces
 *
 * Chunked: transfers are broken into CHUNK_SIZE pieces to allow
 * the GPIO bus to pipeline. Each chunk is sent as raw bytes
 * (no proto framing — we're inside a synchronized phase).
 */

#define PT_ALLREDUCE_CHUNK  (32 * 1024)  /* 32 KB per chunk */

/*
 * In-place allreduce over a float buffer.
 *
 * rank: 0 or 1 (determines who sends first)
 * buf: float array of n_floats elements
 * n_floats: number of floats to reduce
 * transport: GPIO link (or any transport)
 *
 * Returns 0 on success, -1 on timeout.
 */
int pt_allreduce_avg(int rank, float *buf, int n_floats,
                     pt_transport_t *transport);

#endif
