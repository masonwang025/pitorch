/*
 * Dual-link relay test: verify one Pi can use both GPIO banks simultaneously.
 *
 * 3 Pis in a chain:  rank 0 --high--> rank 1 --high--> rank 2
 *                              (low bank)      (low bank)
 *
 * Rank 0: send on high bank
 * Rank 1: recv on low bank, relay on high bank (BOTH banks)
 * Rank 2: recv on low bank, verify data
 *
 * 10 rounds, deterministic pattern, no model weights.
 * Compile with -DRANK=0/1/2.
 */
#include "rpi.h"
#include "pt_link_gpio.h"
#include "pt_proto.h"

#ifndef RANK
#error "RANK must be defined (0, 1, or 2)"
#endif

#define N_ROUNDS  10
#define BUF_SIZE  1152   /* same as dim=288 activation vector */

static void fill_pattern(uint8_t *buf, int round) {
    for (int i = 0; i < BUF_SIZE; i++)
        buf[i] = (uint8_t)((i * 7 + round * 13) & 0xFF);
}

void notmain(void) {
    uart_init();
    printk("=== dual_link rank %d ===\n", RANK);

    /* Every rank inits both banks (rank 1 uses both, others use one) */
    pt_link_gpio_t high = pt_link_gpio_init(16, 24, 25);
    pt_link_gpio_t low  = pt_link_gpio_init(4, 12, 13);

    printk("gpio ready: high bank D=16-23 CLK=24 ACK=25\n");
    printk("            low  bank D=4-11  CLK=12 ACK=13\n");

    /* Rank 0 delays to let others boot */
    if (RANK == 0) delay_ms(3000);

    uint8_t buf[BUF_SIZE];

    for (int round = 0; round < N_ROUNDS; round++) {
        if (RANK == 0) {
            /* Send pattern on high bank (downstream) */
            fill_pattern(buf, round);
            pt_proto_send(&high.base, PT_OP_DATA, buf, BUF_SIZE);
            printk("round %d: sent %d bytes\n", round, BUF_SIZE);

        } else if (RANK == 1) {
            /* Recv on low bank, relay on high bank */
            uint32_t op, len;
            if (pt_proto_recv(&low.base, &op, buf, BUF_SIZE, &len) < 0) {
                printk("FAIL: recv timeout round %d\n", round);
                clean_reboot();
            }
            pt_proto_send(&high.base, PT_OP_DATA, buf, len);
            printk("round %d: relayed %d bytes\n", round, len);

        } else {
            /* Rank 2: recv on low bank, verify */
            uint32_t op, len;
            if (pt_proto_recv(&low.base, &op, buf, BUF_SIZE, &len) < 0) {
                printk("FAIL: recv timeout round %d\n", round);
                clean_reboot();
            }

            uint8_t expected[BUF_SIZE];
            fill_pattern(expected, round);
            int errors = 0;
            for (int i = 0; i < BUF_SIZE; i++) {
                if (buf[i] != expected[i]) errors++;
            }

            if (errors)
                printk("round %d: FAIL %d/%d byte errors\n", round, errors, BUF_SIZE);
            else
                printk("round %d: PASS (%d bytes)\n", round, len);
        }
    }

    printk("=== dual_link rank %d DONE ===\n", RANK);
    clean_reboot();
}
