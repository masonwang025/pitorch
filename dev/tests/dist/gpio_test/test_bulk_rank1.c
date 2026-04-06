/*
 * Bulk GPIO transfer test: rank 1.
 *
 * Mirror of rank 0: for each size, rank 1 recvs first, then sends.
 */
#include "rpi.h"
#include "pt_link_gpio.h"

#define D_BASE  4
#define CLK_PIN 12
#define ACK_PIN 13

/* Use high memory for buffers */
#define SEND_BUF  ((uint8_t *)0x02000000u)
#define RECV_BUF  ((uint8_t *)0x03000000u)

static void fill_pattern(uint8_t *buf, unsigned len, uint8_t seed) {
    for (unsigned i = 0; i < len; i++)
        buf[i] = (uint8_t)(seed + i * 7 + (i >> 8));
}

static int verify_pattern(uint8_t *buf, unsigned len, uint8_t seed) {
    for (unsigned i = 0; i < len; i++) {
        uint8_t expected = (uint8_t)(seed + i * 7 + (i >> 8));
        if (buf[i] != expected) {
            printk("  MISMATCH at byte %d: got 0x%x expected 0x%x\n",
                   i, buf[i], expected);
            return -1;
        }
    }
    return 0;
}

static int test_bulk(pt_link_gpio_t *gpio, unsigned size) {
    printk("--- test %d bytes (%d KB) ---\n", size, size / 1024);

    /* Fill send buffer with rank 1's pattern */
    fill_pattern(SEND_BUF, size, 0xB1);

    /* Clear recv buffer */
    for (unsigned i = 0; i < size; i++) RECV_BUF[i] = 0;

    uint32_t t0 = timer_get_usec();

    /* Phase 1: rank 1 recvs from rank 0 */
    printk("  receiving %d bytes...\n", size);
    int rc = gpio->base.recv_raw(&gpio->base, RECV_BUF, size);
    if (rc < 0) {
        printk("  FAIL: recv timeout\n");
        return -1;
    }

    uint32_t t1 = timer_get_usec();
    printk("  recv done (%d us). sending...\n", t1 - t0);

    /* Phase 2: rank 1 sends to rank 0 (direction swap!) */
    gpio->base.send_raw(&gpio->base, SEND_BUF, size);

    uint32_t t2 = timer_get_usec();
    printk("  send done (%d us).\n", t2 - t1);

    /* Verify: recv buf should have rank 0's pattern (seed 0xA0) */
    if (verify_pattern(RECV_BUF, size, 0xA0) < 0) {
        printk("  FAIL: data integrity check\n");
        return -1;
    }

    unsigned total_bytes = size * 2;
    unsigned total_us = t2 - t0;
    unsigned kbps = total_bytes * 1000 / total_us;
    printk("  PASS: %d bytes round-trip in %d us (%d KB/s)\n",
           total_bytes, total_us, kbps);
    return 0;
}

void notmain(void) {
    uart_init();
    printk("=== Bulk GPIO test: rank 1 ===\n");

    pt_link_gpio_t gpio = pt_link_gpio_init(D_BASE, CLK_PIN, ACK_PIN);
    printk("gpio ready\n");

    /* Handshake: recv from rank 0, send back */
    printk("sync recv...\n");
    uint8_t sync = 0;
    if (gpio.base.recv_raw(&gpio.base, &sync, 1) < 0) {
        printk("FAIL: sync recv timeout\n");
        clean_reboot();
    }
    printk("sync OK (got 0x%x). sending back...\n", sync);
    sync = 0x43;
    gpio.base.send_raw(&gpio.base, &sync, 1);
    printk("sync done\n");

    /* Test increasing sizes */
    unsigned sizes[] = { 1024, 32*1024, 1024*1024, 4*1024*1024, 30*1024*1024 };
    int n_tests = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < n_tests; i++) {
        if (test_bulk(&gpio, sizes[i]) < 0) {
            printk("STOPPED at size %d\n", sizes[i]);
            clean_reboot();
        }
        delay_ms(100);
    }

    printk("=== ALL PASS ===\n");
    clean_reboot();
}
