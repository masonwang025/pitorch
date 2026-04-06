/*
 * Chunked GPIO transfer test: rank 1.
 *
 * Mirror of rank 0:
 *   Phase 1: rank 1 receives 30MB in CHUNKED recv_raw (with compute)
 *   Phase 2: rank 1 sends 30MB in ONE big send_raw
 */
#include "rpi.h"
#include "pt_link_gpio.h"
#include "mmu.h"

#define D_BASE  4
#define CLK_PIN 12
#define ACK_PIN 13

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
    printk("=== Chunked transfer test: rank 1 (MMU ON) ===\n");

    pt_link_gpio_t gpio = pt_link_gpio_init(D_BASE, CLK_PIN, ACK_PIN);

    /* Sync */
    uint8_t sync = 0;
    if (gpio.base.recv_raw(&gpio.base, &sync, 1) < 0) {
        printk("FAIL: sync\n");
        clean_reboot();
    }
    sync = 0x43;
    gpio.base.send_raw(&gpio.base, &sync, 1);
    printk("sync OK\n");

    /* Fill send buffer */
    fill_pattern(SEND_BUF, TEST_SIZE, 0xB1);

    /* Phase 1: recv 30MB in 32KB chunks (like allreduce recv+reduce) */
    printk("chunked recv (%d chunks of %d KB)...\n",
           TEST_SIZE / CHUNK, CHUNK / 1024);

    uint32_t t0 = timer_get_usec();
    unsigned progress = 0;
    for (unsigned off = 0; off < TEST_SIZE; off += CHUNK) {
        unsigned n = (off + CHUNK <= TEST_SIZE) ? CHUNK : TEST_SIZE - off;
        int rc = gpio.base.recv_raw(&gpio.base, RECV_BUF + off, n);
        if (rc < 0) {
            printk("FAIL: recv timeout at chunk %d (off=%d)\n",
                   off / CHUNK, off);
            clean_reboot();
        }

        /* Simulate reduction work */
        volatile uint32_t dummy = 0;
        for (unsigned i = 0; i < n; i += 64)
            dummy += RECV_BUF[off + i];

        progress += n;
        if (progress >= 4 * 1024 * 1024) {
            printk(".");
            progress -= 4 * 1024 * 1024;
        }
    }

    uint32_t t1 = timer_get_usec();
    printk("\nrecv done (%d ms)\n", (t1 - t0) / 1000);

    /* Phase 2: send 30MB in one call */
    printk("sending %d MB...\n", TEST_SIZE / (1024*1024));
    gpio.base.send_raw(&gpio.base, SEND_BUF, TEST_SIZE);
    uint32_t t2 = timer_get_usec();
    printk("send done (%d ms)\n", (t2 - t1) / 1000);

    printk("=== rank 1 DONE ===\n");
    clean_reboot();
}
