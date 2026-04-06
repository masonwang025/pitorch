#include <string.h>
#include "pt_math.h"
#include "pt_ops.h"

void rmsnorm(float *o, const float *x, const float *w, int dim) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++)
        ss += x[i] * x[i];
    ss = 1.0f / pt_sqrtf(ss / dim + 1e-5f);
    for (int i = 0; i < dim; i++)
        o[i] = x[i] * ss * w[i];
}

void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = pt_expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < size; i++)
        x[i] *= inv;
}

void silu(float *x, int size) {
    for (int i = 0; i < size; i++)
        x[i] = x[i] / (1.0f + pt_expf(-x[i]));
}

void rope(float *q, float *k, int dim, int head_dim, int pos) {
    for (int i = 0; i < dim; i += 2) {
        int head_i = i % head_dim;
        float freq = pt_expf(-9.210340372f * (float)head_i / (float)head_dim);
        float val = (float)pos * freq;
        float cos_val = pt_cosf(val);
        float sin_val = pt_sinf(val);

        float q0 = q[i], q1 = q[i + 1];
        q[i]     = q0 * cos_val - q1 * sin_val;
        q[i + 1] = q0 * sin_val + q1 * cos_val;

        float k0 = k[i], k1 = k[i + 1];
        k[i]     = k0 * cos_val - k1 * sin_val;
        k[i + 1] = k0 * sin_val + k1 * cos_val;
    }
}

void vec_add(float *o, const float *a, const float *b, int size) {
    for (int i = 0; i < size; i++)
        o[i] = a[i] + b[i];
}

void vec_mul(float *o, const float *a, const float *b, int size) {
    for (int i = 0; i < size; i++)
        o[i] = a[i] * b[i];
}

void embedding_lookup(float *o, const float *table, int dim, int token) {
    memcpy(o, table + token * dim, dim * sizeof(float));
}

int argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

void smatvec_cpu(const float *W, const float *x, float *y,
                 int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; i++) {
        float acc = 0.0f;
        for (int j = 0; j < in_dim; j++)
            acc += W[i * in_dim + j] * x[j];
        y[i] = acc;
    }
}

#ifdef __RPI__
#include "rpi.h"
#define PT_PF_PUT(c) printk("%c", (c))
#else
#include <stdio.h>
#define PT_PF_PUT(c) putchar(c)
#endif

void pt_pf(float f, int decimals) {
    if (f < 0.0f) { PT_PF_PUT('-'); f = -f; }
    unsigned whole = (unsigned)f;
    float frac = f - (float)whole;
    /* emit integer part via a small stack buffer */
    char buf[12];
    int n = 0;
    if (whole == 0) { buf[n++] = '0'; }
    else { while (whole) { buf[n++] = '0' + (whole % 10); whole /= 10; } }
    while (n > 0) PT_PF_PUT(buf[--n]);
    PT_PF_PUT('.');
    for (int i = 0; i < decimals; i++) {
        frac *= 10.0f;
        int d = (int)frac;
        if (d > 9) d = 9;
        PT_PF_PUT('0' + d);
        frac -= (float)d;
    }
}
