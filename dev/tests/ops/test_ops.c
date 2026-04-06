#include "rpi.h"
#include <string.h>
#include "gpu.h"
#include "arena.h"
#include "pt_math.h"
#include "pt_ops.h"

/* ── helpers ─────────────────────────────────────────────── */

static int pass_count, fail_count;

static inline uint32_t f2u(float f)
{
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return u;
}

static float fabsf_(float x) { return x < 0 ? -x : x; }

static int approx_eq(float a, float b, float atol, float rtol)
{
    float diff = fabsf_(a - b);
    float scale = fabsf_(b);
    if (scale < 1e-6f)
        return diff < atol;
    return diff < atol || (diff / scale) < rtol;
}

static void result(const char *name, int ok)
{
    printk("%s: %s\n", name, ok ? "PASS" : "FAIL");
    if (ok)
        pass_count++;
    else
        fail_count++;
}

/* ── arena test ──────────────────────────────────────────── */

static void test_arena(void)
{
    qpu_enable();
    gpu_arena_init(4096);

    volatile uint32_t *a = gpu_arena_alloc(64);
    volatile uint32_t *b = gpu_arena_alloc(64);
    volatile uint32_t *c = gpu_arena_alloc(64);

    for (int i = 0; i < 16; i++)
    {
        a[i] = 0xAAAA0000 + i;
    }
    for (int i = 0; i < 16; i++)
    {
        b[i] = 0xBBBB0000 + i;
    }
    for (int i = 0; i < 16; i++)
    {
        c[i] = 0xCCCC0000 + i;
    }

    int ok = 1;
    for (int i = 0; i < 16; i++)
    {
        if (a[i] != 0xAAAA0000 + (unsigned)i)
        {
            ok = 0;
            printk("  arena: a[%d]=%x\n", i, a[i]);
        }
        if (b[i] != 0xBBBB0000 + (unsigned)i)
        {
            ok = 0;
            printk("  arena: b[%d]=%x\n", i, b[i]);
        }
        if (c[i] != 0xCCCC0000 + (unsigned)i)
        {
            ok = 0;
            printk("  arena: c[%d]=%x\n", i, c[i]);
        }
    }

    gpu_arena_reset();
    volatile uint32_t *d = gpu_arena_alloc(64);
    /* after reset, d should start at same base as a */
    ok = ok && ((uint32_t)d == (uint32_t)a);
    if ((uint32_t)d != (uint32_t)a)
        printk("  arena: reset base mismatch d=%x a=%x\n", (uint32_t)d, (uint32_t)a);

    d[0] = 0xDEADBEEF;
    ok = ok && (d[0] == 0xDEADBEEF);

    gpu_arena_free();
    qpu_disable();
    result("gpu_arena", ok);
}

/* ── math tests ──────────────────────────────────────────── */

static void test_expf(void)
{
    float in[] = {-10.0f, -5.0f, -2.0f, -1.0f, -0.5f, -0.1f, 0.0f,
                  0.1f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f};
    float ref[] = {0.00004540f, 0.00673795f, 0.13533528f, 0.36787945f,
                   0.60653067f, 0.90483743f, 1.00000000f, 1.10517097f,
                   1.64872122f, 2.71828175f, 7.38905621f, 148.41316223f,
                   22026.46484375f};
    int n = 13, ok = 1;
    for (int i = 0; i < n; i++)
    {
        float got = pt_expf(in[i]);
        if (!approx_eq(got, ref[i], 1e-4f, 1e-4f))
        {
            printk("  expf(%d): got %x ref %x\n", (int)in[i], f2u(got), f2u(ref[i]));
            ok = 0;
        }
    }
    result("expf", ok);
}

static void test_sinf(void)
{
    float in[] = {0.0f, 0.1f, 0.5f, 1.0f, 0.78539819f, 1.57079637f,
                  3.14159274f, 4.71238899f, 6.28318548f, -0.5f, -1.0f,
                  -1.57079637f, -3.14159274f};
    float ref[] = {0.0f, 0.09983342f, 0.47942555f, 0.84147102f, 0.70710677f,
                   1.0f, -0.00000009f, -1.0f, 0.00000017f, -0.47942555f,
                   -0.84147102f, -1.0f, 0.00000009f};
    int n = 13, ok = 1;
    for (int i = 0; i < n; i++)
    {
        float got = pt_sinf(in[i]);
        if (!approx_eq(got, ref[i], 5e-4f, 5e-4f))
        {
            printk("  sinf[%d]: got %x ref %x\n", i, f2u(got), f2u(ref[i]));
            ok = 0;
        }
    }
    result("sinf", ok);
}

static void test_cosf(void)
{
    float in[] = {0.0f, 0.1f, 0.5f, 1.0f, 0.78539819f, 1.57079637f,
                  3.14159274f, 4.71238899f, 6.28318548f, -0.5f, -1.0f,
                  -1.57079637f, -3.14159274f};
    float ref[] = {1.0f, 0.99500418f, 0.87758255f, 0.54030228f, 0.70710677f,
                   -0.00000004f, -1.0f, 0.00000001f, 1.0f, 0.87758255f,
                   0.54030228f, -0.00000004f, -1.0f};
    int n = 13, ok = 1;
    for (int i = 0; i < n; i++)
    {
        float got = pt_cosf(in[i]);
        if (!approx_eq(got, ref[i], 5e-4f, 5e-4f))
        {
            printk("  cosf[%d]: got %x ref %x\n", i, f2u(got), f2u(ref[i]));
            ok = 0;
        }
    }
    result("cosf", ok);
}

/* ── op tests ────────────────────────────────────────────── */

static void test_rmsnorm(void)
{
    float x[] = {1.0f, 2.0f, -1.0f, 0.5f, -0.5f, 3.0f};
    float w[] = {0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f};
    float ref[] = {0.31108490f, 1.24433959f, -0.31108490f, 0.31108490f,
                   -0.15554245f, 1.86650944f};
    float out[6];
    rmsnorm(out, x, w, 6);
    int ok = 1;
    for (int i = 0; i < 6; i++)
    {
        if (!approx_eq(out[i], ref[i], 1e-4f, 1e-4f))
        {
            printk("  rmsnorm[%d]: got %x ref %x\n", i, f2u(out[i]), f2u(ref[i]));
            ok = 0;
        }
    }
    result("rmsnorm", ok);
}

static void test_softmax(void)
{
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f};
    float ref[] = {0.02864414f, 0.07786284f, 0.21165316f, 0.57533288f,
                   0.02864414f, 0.07786284f};
    softmax(x, 6);
    int ok = 1;
    for (int i = 0; i < 6; i++)
    {
        if (!approx_eq(x[i], ref[i], 1e-4f, 1e-4f))
        {
            printk("  softmax[%d]: got %x ref %x\n", i, f2u(x[i]), f2u(ref[i]));
            ok = 0;
        }
    }
    result("softmax", ok);
}

static void test_silu(void)
{
    float x[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    float ref[] = {-0.23840584f, -0.26894143f, 0.0f, 0.73105860f,
                   1.76159406f, 2.85772228f};
    silu(x, 6);
    int ok = 1;
    for (int i = 0; i < 6; i++)
    {
        if (!approx_eq(x[i], ref[i], 1e-4f, 1e-4f))
        {
            printk("  silu[%d]: got %x ref %x\n", i, f2u(x[i]), f2u(ref[i]));
            ok = 0;
        }
    }
    result("silu", ok);
}

static void test_rope(void)
{
    float q[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float k[] = {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    float q_ref[] = {-1.27223253f, -1.83886504f, 2.41576958f, 4.37767696f,
                     4.96111584f, 6.03219128f, -8.05890751f, -6.93209982f,
                     7.52490473f, 11.15238953f, 10.92221069f, 12.07084560f};
    float k_ref[] = {-13.43223000f, -9.19647789f, 8.65402412f, 10.30086708f,
                     7.95459032f, 7.05155993f, -6.64555454f, -4.10324287f,
                     3.54488850f, 3.52615452f, 1.99349499f, 1.01290560f};
    rope(q, k, 12, 6, 3);
    int ok = 1;
    for (int i = 0; i < 12; i++)
    {
        if (!approx_eq(q[i], q_ref[i], 5e-3f, 5e-3f))
        {
            printk("  rope q[%d]: got %x ref %x\n", i, f2u(q[i]), f2u(q_ref[i]));
            ok = 0;
        }
        if (!approx_eq(k[i], k_ref[i], 5e-3f, 5e-3f))
        {
            printk("  rope k[%d]: got %x ref %x\n", i, f2u(k[i]), f2u(k_ref[i]));
            ok = 0;
        }
    }
    result("rope", ok);
}

static void test_vec_add(void)
{
    float a[] = {1, 2, 3, 4, 5, 6};
    float b[] = {6, 5, 4, 3, 2, 1};
    float ref[] = {7, 7, 7, 7, 7, 7};
    float out[6];
    vec_add(out, a, b, 6);
    int ok = 1;
    for (int i = 0; i < 6; i++)
        if (out[i] != ref[i])
        {
            ok = 0;
            printk("  vec_add[%d]: got %x\n", i, f2u(out[i]));
        }
    result("vec_add", ok);
}

static void test_vec_mul(void)
{
    float a[] = {1, 2, 3, 4, 5, 6};
    float b[] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
    float ref[] = {0.5f, 3.0f, 7.5f, 14.0f, 22.5f, 33.0f};
    float out[6];
    vec_mul(out, a, b, 6);
    int ok = 1;
    for (int i = 0; i < 6; i++)
    {
        if (!approx_eq(out[i], ref[i], 1e-5f, 1e-5f))
        {
            printk("  vec_mul[%d]: got %x ref %x\n", i, f2u(out[i]), f2u(ref[i]));
            ok = 0;
        }
    }
    result("vec_mul", ok);
}

static void test_embedding_lookup(void)
{
    float table[] = {
        1.0f, 0.5f, -0.5f, 0.1f, -0.1f, 0.3f,
        0.2f, -0.3f, 0.8f, -0.6f, 0.4f, 0.1f,
        0.9f, 0.7f, -0.2f, 0.5f, -0.8f, 0.6f,
        -0.4f, 0.3f, 0.1f, -0.9f, 0.2f, -0.5f};
    float ref[] = {0.9f, 0.7f, -0.2f, 0.5f, -0.8f, 0.6f};
    float out[6];
    embedding_lookup(out, table, 6, 2);
    int ok = 1;
    for (int i = 0; i < 6; i++)
    {
        if (!approx_eq(out[i], ref[i], 1e-6f, 1e-6f))
        {
            printk("  emb[%d]: got %x ref %x\n", i, f2u(out[i]), f2u(ref[i]));
            ok = 0;
        }
    }
    result("embedding_lookup", ok);
}

static void test_argmax(void)
{
    float x[] = {0.1f, 0.5f, 0.3f, 0.9f, 0.2f, 0.4f};
    int got = argmax(x, 6);
    int ok = (got == 3);
    if (!ok)
        printk("  argmax: got %d ref 3\n", got);
    result("argmax", ok);
}

static void test_smatvec_cpu(void)
{
    float W[] = {
        1.0f, 0.5f, -0.3f, 0.2f, 0.1f, -0.4f,
        -0.2f, 0.8f, 0.1f, -0.5f, 0.3f, 0.6f,
        0.4f, -0.1f, 0.7f, 0.3f, -0.6f, 0.2f,
        0.1f, 0.3f, -0.2f, 0.9f, 0.4f, -0.1f};
    float x[] = {1.0f, 2.0f, 3.0f, -1.0f, 0.5f, -0.5f};
    float ref[] = {1.14999998f, 2.04999995f, 1.60000002f, -0.54999995f};
    float y[4];
    smatvec_cpu(W, x, y, 4, 6);
    int ok = 1;
    for (int i = 0; i < 4; i++)
    {
        if (!approx_eq(y[i], ref[i], 1e-4f, 1e-4f))
        {
            printk("  smatvec[%d]: got %x ref %x\n", i, f2u(y[i]), f2u(ref[i]));
            ok = 0;
        }
    }
    result("smatvec_cpu", ok);
}

/* ── main ────────────────────────────────────────────────── */

void notmain(void)
{
    printk("\n=== ops tests ===\n\n");

    test_arena();
    test_expf();
    test_sinf();
    test_cosf();
    test_rmsnorm();
    test_softmax();
    test_silu();
    test_rope();
    test_vec_add();
    test_vec_mul();
    test_embedding_lookup();
    test_argmax();
    test_smatvec_cpu();

    printk("\n=== %d passed, %d failed ===\n", pass_count, fail_count);
    if (fail_count == 0)
        printk("ALL PASS\n");
}
