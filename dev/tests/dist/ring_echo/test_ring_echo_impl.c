/*
 * Ring echo test: send data around the full 4-Pi ring and verify integrity.
 *
 *   rank 0 --high--> rank 1 --high--> rank 2 --high--> rank 3 --high--> rank 0
 *            (low)            (low)            (low)            (low)
 *
 * Rank 0: send on downstream, recv on upstream, verify.
 * Ranks 1-3: recv on upstream, forward on downstream.
 *
 * PERMANENT DIAGNOSTIC -- keep this test for verifying ring hardware.
 * Compile with -DRANK=0/1/2/3.
 */
#include "rpi.h"
#include "pt_link_gpio.h"
#include "pt_proto.h"

#ifndef RANK
#error "RANK must be defined (0-3)"
#endif

#define WORLD_SIZE 4
#define N_ROUNDS   10
#define BUF_SIZE   1152

static void fill_pattern(uint8_t *buf, int round) {
    for (int i = 0; i < BUF_SIZE; i++)
        buf[i] = (uint8_t)((i * 7 + round * 13 + 42) & 0xFF);
}

void notmain(void) {
    uart_init();
    printk("=== ring_echo rank %d/%d ===\n", RANK, WORLD_SIZE);

    pt_link_gpio_t downstream = pt_link_gpio_init(16, 24, 25);
    pt_link_gpio_t upstream   = pt_link_gpio_init(4, 12, 13);
    printk("gpio ready\n");

    if (RANK == 0) delay_ms(3000);

    uint8_t buf[BUF_SIZE];
    int pass = 0, fail = 0;

    for (int round = 0; round < N_ROUNDS; round++) {
        if (RANK == 0) {
            /* Send pattern downstream */
            fill_pattern(buf, round);
            pt_proto_send(&downstream.base, PT_OP_DATA, buf, BUF_SIZE);

            /* Recv from upstream (came around the ring) */
            uint32_t op, len;
            if (pt_proto_recv(&upstream.base, &op, buf, BUF_SIZE, &len) < 0) {
                printk("round %d: FAIL recv timeout\n", round);
                fail++;
                continue;
            }

            /* Verify */
            uint8_t expected[BUF_SIZE];
            fill_pattern(expected, round);
            int errors = 0;
            for (int i = 0; i < BUF_SIZE; i++) {
                if (buf[i] != expected[i]) errors++;
            }
            if (errors) {
                printk("round %d: FAIL %d/%d byte errors\n", round, errors, BUF_SIZE);
                fail++;
            } else {
                printk("round %d: PASS\n", round);
                pass++;
            }
        } else {
            /* Recv from upstream, forward downstream */
            uint32_t op, len;
            if (pt_proto_recv(&upstream.base, &op, buf, BUF_SIZE, &len) < 0) {
                printk("round %d: FAIL recv timeout\n", round);
                clean_reboot();
            }
            pt_proto_send(&downstream.base, PT_OP_DATA, buf, len);
            printk("round %d: forwarded %d bytes\n", round, len);
        }
    }

    if (RANK == 0)
        printk("\n%d/%d PASS, %d FAIL\n", pass, N_ROUNDS, fail);

    printk("=== ring_echo rank %d DONE ===\n", RANK);
    clean_reboot();
}
