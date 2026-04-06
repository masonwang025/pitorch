#ifndef PITORCH_BACKWARD_OPS_H
#define PITORCH_BACKWARD_OPS_H

/*
 * Backward ops for Llama2 training.
 * Each mirrors a forward op. All use += semantics for output gradients
 * (caller zeros gradient buffers before the backward pass).
 *
 * Portable C — no Pi-specific dependencies.
 */

/*
 * RMSNorm backward.
 * Forward: o[i] = (x[i] / rms) * w[i],  rms = sqrt(mean(x^2) + eps)
 * Needs saved input x and weight w.
 */
void rmsnorm_backward(float *d_x, float *d_w,
                      const float *d_o, const float *x, const float *w,
                      int dim);

/*
 * Linear (matvec) input gradient:  d_x += W^T @ d_y
 * W is [out_dim, in_dim] row-major.
 */
void matmul_backward_input(float *d_x,
                           const float *W, const float *d_y,
                           int out_dim, int in_dim);

/*
 * Linear (matvec) weight gradient:  d_W += d_y ⊗ x^T  (rank-1 outer product)
 * d_W is [out_dim, in_dim] row-major.
 */
void matmul_backward_weight(float *d_W,
                            const float *d_y, const float *x,
                            int out_dim, int in_dim);

/*
 * RoPE backward.
 * RoPE is an orthogonal rotation, so backward is the inverse rotation
 * (transpose of rotation matrix = rotation by -angle).
 * d_q, d_k are pre-RoPE gradients; d_q_out, d_k_out are post-RoPE gradients.
 */
void rope_backward(float *d_q, float *d_k,
                   const float *d_q_out, const float *d_k_out,
                   int dim, int head_dim, int pos);

/*
 * Softmax backward.
 * Given softmax output y and upstream gradient d_y:
 *   d_x[i] += y[i] * (d_y[i] - sum_j(d_y[j] * y[j]))
 */
void softmax_backward(float *d_x,
                      const float *d_y, const float *y,
                      int size);

/*
 * SiLU backward.
 * Forward: y = x * sigmoid(x)
 * x is the pre-activation (before SiLU was applied).
 */
void silu_backward(float *d_x,
                   const float *d_y, const float *x,
                   int size);

/*
 * Embedding backward: scatter-add upstream gradient into the table.
 *   d_table[token * dim + i] += d_o[i]
 */
void embedding_backward(float *d_table,
                        const float *d_o,
                        int dim, int token);

#endif
