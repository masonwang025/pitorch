#ifndef PITORCH_OPS_H
#define PITORCH_OPS_H

/*
 * CPU reference ops for LLM inference.
 * All operate on flat float arrays. No allocations.
 */

void rmsnorm(float *o, const float *x, const float *w, int dim);
void softmax(float *x, int size);
void silu(float *x, int size);
void rope(float *q, float *k, int dim, int head_dim, int pos);

void vec_add(float *o, const float *a, const float *b, int size);
void vec_mul(float *o, const float *a, const float *b, int size);

void embedding_lookup(float *o, const float *table, int dim, int token);
int  argmax(const float *x, int n);

/* CPU reference matvec: y = W @ x.  W is [out_dim x in_dim], row-major. */
void smatvec_cpu(const float *W, const float *x, float *y,
                 int out_dim, int in_dim);

/*
 * Print a float over UART/stdout.  Single-precision only — no double-precision
 * soft-float dependencies, safe for bare-metal Pi.
 */
void pt_pf(float f, int decimals);

#endif
