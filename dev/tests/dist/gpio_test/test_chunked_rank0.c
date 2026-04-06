/*
 * Chunked GPIO transfer test: rank 0.
 *
 * Tests the pattern used by allreduce:
 *   Phase 1: rank 0 sends 30MB in ONE big send_raw
 *   Phase 2: rank 0 receives 30MB in CHUNKED recv_raw (32KB chunks)
 *            with computation between chunks (simulating reduce)
 *
 * This isolates whether chunked recv after a big send causes issues.
 */
#include "rpi.h"
#include "pt_link_gpio.h"
#include "mmu.h"

#define D_BASE  16
#define CLK_PIN 24
#define ACK_PIN 25

#define SEND_BUF  ((uint8_t *)0x02000000u)
#define RECV_BUF  ((uint8_t *)0x03000000u)
#define CHUNK     (32 * 1024)
#define TEST_SIZE (30 * 1024 * 1024)

static void fill_pattern(uint8_t *buf, unsigned len, uint8_t seed) {
    for (unsigned i = 0; i < len; i++)
        buf[i] = (uint8_t)(seed + i * 7 + (i >> 8));
}

void notmain(void) {
    mmu_init_and_enable();
    printk("=== Chunked transfer test: rank 0 (MMU ON) ===\n");

    pt_link_gpio_t gpio = pt_link_gpio_init(D_BASE, CLK_PIN, ACK_PIN);

    /* Sync */
    delay_ms(2000);
    uint8_t sync = 0x42;
    gpio.base.send_raw(&gpio.base, &sync, 1);
    if (gpio.base.recv_raw(&gpio.base, &sync, 1) < 0) {
        printk("FAIL: sync\n");
        clean_reboot();
    }
    printk("sync OK\n");

    /* Fill send buffer */
    fill_pattern(SEND_BUF, TEST_SIZE, 0xA0);

    /* Phase 1: send 30MB in one call */
    printk("sending %d MB...\n", TEST_SIZE / (1024*1024));
    uint32_t t0 = timer_get_usec();
    gpio.base.send_raw(&gpio.base, SEND_BUF, TEST_SIZE);
    uint32_t t1 = timer_get_usec();
    printk("send done (%d ms)\n", (t1 - t0) / 1000);

    /* Phase 2: recv 30MB in 32KB chunks (like allreduce recv+reduce) */
    printk("chunked recv (%d chunks of %d KB)...\n",
           TEST_SIZE / CHUNK, CHUNK / 1024);

    unsigned progress = 0;
    for (unsigned off = 0; off < TEST_SIZE; off += CHUNK) {
        unsigned n = (off + CHUNK <= TEST_SIZE) ? CHUNK : TEST_SIZE - off;
        int rc = gpio.base.recv_raw(&gpio.base, RECV_BUF + off, n);
        if (rc < 0) {
            printk("FAIL: recv timeout at chunk %d (off=%d)\n",
                   off / CHUNK, off);
            clean_reboot();
        }

        /* Simulate reduction work (touch the received data) */
        volatile uint32_t dummy = 0;
        for (unsigned i = 0; i < n; i += 64)
            dummy += RECV_BUF[off + i];

        progress += n;
        if (progress >= 4 * 1024 * 1024) {
            printk(".");
            progress -= 4 * 1024 * 1024;
        }
    }

    uint32_t t2 = timer_get_usec();
    printk("\nrecv done (%d ms)\n", (t2 - t1) / 1000);

    /* Verify */
    int ok = 1;
    for (unsigned i = 0; i < 100; i++) {
        unsigned idx = i * (TEST_SIZE / 100);
        uint8_t expected = (uint8_t)(0xB1 + idx * 7 + (idx >> 8));
        if (RECV_BUF[idx] != expected) {
            printk("MISMATCH at %d: got 0x%x expected 0x%x\n",
                   idx, RECV_BUF[idx], expected);
            ok = 0;
            break;
        }
    }

    if (ok) printk("PASS: data verified\n");
    printk("=== rank 0 DONE ===\n");
    clean_reboot();
}
