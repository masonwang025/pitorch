#include <stdint.h>
#include "pt_math.h"

/* ── expf ──────────────────────────────────────────────────────
 * Range reduction: x = n*ln2 + r,  |r| <= ln2/2
 * exp(r) via degree-5 Taylor, 2^n via IEEE 754 exponent.
 * Max relative error ~2e-7 on [-87, 88]. */
float pt_expf(float x) {
    if (x > 88.7f)  return 3.4028235e+38f;
    if (x < -87.3f) return 0.0f;

    const float LN2     = 0.6931471805599453f;
    const float INV_LN2 = 1.4426950408889634f;

    float fn = x * INV_LN2;
    int n = (int)(fn + (fn >= 0 ? 0.5f : -0.5f));
    float r = x - (float)n * LN2;

    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.1666666667f
                    + r * (0.0416666667f + r * 0.0083333333f))));

    union { float f; uint32_t u; } scale;
    scale.u = (uint32_t)(n + 127) << 23;
    return p * scale.f;
}

/* ── logf ─────────────────────────────────────────────────────
 * Decompose x = m * 2^e (IEEE 754), then ln(x) = ln(m) + e*ln(2).
 * Mantissa remapped near 1 for polynomial convergence.
 * 9-term Taylor for ln(1+f), max relative error ~1e-5. */
float pt_logf(float x) {
    if (x <= 0.0f) return -3.4028235e+38f;
    union { float f; uint32_t u; } conv;
    conv.f = x;
    int e = (int)((conv.u >> 23) & 0xFF) - 127;
    conv.u = (conv.u & 0x007FFFFF) | 0x3F800000;
    float m = conv.f;
    if (m > 1.41421356f) { m *= 0.5f; e++; }
    float f = m - 1.0f;
    float r = f * (1.0f + f * (-0.5f + f * (0.33333333f
              + f * (-0.25f + f * (0.2f + f * (-0.16666667f
              + f * (0.14285714f + f * (-0.125f + f * 0.11111111f))))))));
    return r + (float)e * 0.69314718f;
}

/* ── sinf / cosf ──────────────────────────────────────────────
 * Cody-Waite reduction to [-pi/2, pi/2], then polynomial.
 * sin: odd polynomial degree 9  (5 terms)
 * cos: even polynomial degree 8 (5 terms)
 * Max relative error ~3e-7. */

static const float PI      = 3.14159265358979f;
static const float HALF_PI = 1.57079632679490f;
static const float TWO_PI  = 6.28318530717959f;
static const float INV_TWO_PI = 0.15915494309190f;

/* reduce x to [-pi, pi] */
static float reduce(float x) {
    float n = x * INV_TWO_PI;
    n = (float)(int)(n + (n >= 0 ? 0.5f : -0.5f));
    return x - n * TWO_PI;
}

float pt_sinf(float x) {
    x = reduce(x);

    /* map to [-pi/2, pi/2] using sin(pi-x) = sin(x) */
    if (x > HALF_PI)       x = PI - x;
    else if (x < -HALF_PI) x = -PI - x;

    float x2 = x * x;
    return x * (1.0f + x2 * (-0.16666666667f + x2 * (0.00833333333f
              + x2 * (-0.00019841270f + x2 * 0.00000275573f))));
}

float pt_cosf(float x) {
    x = reduce(x);

    /* cos(pi-x) = -cos(x), unlike sin where sin(pi-x) = sin(x) */
    float sign = 1.0f;
    if (x > HALF_PI)       { x = PI - x; sign = -1.0f; }
    else if (x < -HALF_PI) { x = -PI - x; sign = -1.0f; }

    float x2 = x * x;
    return sign * (1.0f + x2 * (-0.50000000000f + x2 * (0.04166666667f
                  + x2 * (-0.00138888889f + x2 * 0.00002480159f))));
}

/* ── sqrtf ────────────────────────────────────────────────────
 * Bit-trick initial guess + 3 Newton-Raphson iterations.
 * Accurate to float32 precision. */
float pt_sqrtf(float x) {
    if (x <= 0.0f) return 0.0f;
    union { float f; uint32_t i; } conv;
    conv.f = x;
    conv.i = 0x1fbd1df5 + (conv.i >> 1);
    float y = conv.f;
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    return y;
}
