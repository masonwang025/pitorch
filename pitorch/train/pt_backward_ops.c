#include "pt_backward_ops.h"
#include "pt_math.h"

void rmsnorm_backward(float *d_x, float *d_w,
                      const float *d_o, const float *x, const float *w,
                      int dim) {
    /*
     * Forward: ss = sum(x^2)/dim + eps,  inv = 1/sqrt(ss),  o[i] = x[i]*inv*w[i]
     *
     * d_w[i] += d_o[i] * x[i] * inv
     * d_x[j] += inv * d_o[j] * w[j]  -  (inv^3 / dim) * dot * x[j]
     *   where dot = sum_i( d_o[i] * w[i] * x[i] )
     */
    float ss = 0.0f;
    for (int i = 0; i < dim; i++)
        ss += x[i] * x[i];
    ss = ss / dim + 1e-5f;
    float inv = 1.0f / pt_sqrtf(ss);

    for (int i = 0; i < dim; i++)
        d_w[i] += d_o[i] * x[i] * inv;

    float dot = 0.0f;
    for (int i = 0; i < dim; i++)
        dot += d_o[i] * w[i] * x[i];

    float coeff = inv * inv * inv / (float)dim;
    for (int i = 0; i < dim; i++)
        d_x[i] += inv * d_o[i] * w[i] - coeff * dot * x[i];
}

void matmul_backward_input(float *d_x,
                           const float *W, const float *d_y,
                           int out_dim, int in_dim) {
    for (int j = 0; j < in_dim; j++) {
        float sum = 0.0f;
        for (int i = 0; i < out_dim; i++)
            sum += W[i * in_dim + j] * d_y[i];
        d_x[j] += sum;
    }
}

void matmul_backward_weight(float *d_W,
                            const float *d_y, const float *x,
                            int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; i++)
        for (int j = 0; j < in_dim; j++)
            d_W[i * in_dim + j] += d_y[i] * x[j];
}


void rope_backward(float *d_q, float *d_k,
                   const float *d_q_out, const float *d_k_out,
                   int dim, int head_dim, int pos) {
    for (int i = 0; i < dim; i += 2) {
        int head_i = i % head_dim;
        float freq = pt_expf(-9.210340372f * (float)head_i / (float)head_dim);
        float val  = (float)pos * freq;
        float c    = pt_cosf(val);
        float s    = pt_sinf(val);

        /* R = [c -s; s c],  R^T = [c s; -s c] */
        d_q[i]     +=  d_q_out[i] * c + d_q_out[i + 1] * s;
        d_q[i + 1] += -d_q_out[i] * s + d_q_out[i + 1] * c;

        d_k[i]     +=  d_k_out[i] * c + d_k_out[i + 1] * s;
        d_k[i + 1] += -d_k_out[i] * s + d_k_out[i + 1] * c;
    }
}

void softmax_backward(float *d_x,
                      const float *d_y, const float *y,
                      int size) {
    float dot = 0.0f;
    for (int i = 0; i < size; i++)
        dot += d_y[i] * y[i];
    for (int i = 0; i < size; i++)
        d_x[i] += y[i] * (d_y[i] - dot);
}

void silu_backward(float *d_x,
                   const float *d_y, const float *x,
                   int size) {
    for (int i = 0; i < size; i++) {
        float sig = 1.0f / (1.0f + pt_expf(-x[i]));
        d_x[i] += d_y[i] * sig * (1.0f + x[i] * (1.0f - sig));
    }
}

void embedding_backward(float *d_table,
                        const float *d_o,
                        int dim, int token) {
    for (int i = 0; i < dim; i++)
        d_table[token * dim + i] += d_o[i];
}
