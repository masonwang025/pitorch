/*
 * GPIO parallel bus echo test — rank 0 (initiator).
 *
 * Sends 1152 bytes (one activation vector worth) to rank 1,
 * rank 1 echoes it back. Verify byte-for-byte. Repeat 100 times.
 * Then measure bandwidth with a 1 MB transfer.
 */
#include "rpi.h"
#include "pt_link_gpio.h"
#include "pt_proto.h"

/* Rank 0 uses high bank: data=GPIO 16-23, CLK=24, ACK=25 */
#define D_BASE  16
#define CLK_PIN 24
#define ACK_PIN 25

#define ACTIVATION_SIZE 1152   /* dim=288 * sizeof(float) */
#define ECHO_ROUNDS     100
#define BW_SIZE         (64 * 1024)  /* 64 KB for bandwidth test */

void notmain(void) {
    uart_init();
    printk("=== gpio_echo rank 0 ===\n");

    pt_link_gpio_t gpio = pt_link_gpio_init(D_BASE, CLK_PIN, ACK_PIN);
    printk("gpio link ready: D=%d-%d CLK=%d ACK=%d\n",
           D_BASE, D_BASE + 7, CLK_PIN, ACK_PIN);

    /* Handshake via proto layer */
    delay_ms(1000);
    uint32_t dummy = 0, opcode, plen;
    printk("sending PING...\n");
    pt_proto_send(&gpio.base, PT_OP_PING, &dummy, sizeof(dummy));

    if (pt_proto_recv(&gpio.base, &opcode, &dummy, sizeof(dummy), &plen) < 0 ||
        opcode != PT_OP_PONG) {
        printk("FAIL: no PONG\n");
        clean_reboot();
    }
    printk("rank 1 connected!\n");

    /* --- Echo test: send activation-sized buffer, recv back, compare --- */
    uint8_t sendbuf[ACTIVATION_SIZE];
    uint8_t recvbuf[ACTIVATION_SIZE];

    int errors = 0;
    for (int round = 0; round < ECHO_ROUNDS; round++) {
        /* Fill with deterministic pattern */
        for (int i = 0; i < ACTIVATION_SIZE; i++)
            sendbuf[i] = (uint8_t)((round * 137 + i * 41) & 0xFF);

        pt_proto_send(&gpio.base, PT_OP_DATA, sendbuf, ACTIVATION_SIZE);

        uint32_t recv_op, recv_len;
        if (pt_proto_recv(&gpio.base, &recv_op, recvbuf, ACTIVATION_SIZE, &recv_len) < 0) {
            printk("FAIL: timeout on round %d\n", round);
            clean_reboot();
        }

        /* Compare */
        int mismatch = 0;
        for (int i = 0; i < ACTIVATION_SIZE; i++) {
            if (recvbuf[i] != sendbuf[i]) {
                if (mismatch < 3)
                    printk("  round %d byte %d: sent 0x%x got 0x%x\n",
                           round, i, sendbuf[i], recvbuf[i]);
                mismatch++;
            }
        }
        if (mismatch) {
            printk("round %d: %d/%d bytes wrong\n", round, mismatch, ACTIVATION_SIZE);
            errors++;
        }
    }

    if (errors == 0)
        printk("echo test: %d rounds x %d bytes — all PASS\n",
               ECHO_ROUNDS, ACTIVATION_SIZE);
    else
        printk("echo test: %d/%d rounds had errors\n", errors, ECHO_ROUNDS);

    /* --- Bandwidth test: send 64 KB, measure wall time --- */
    uint8_t bw_buf[BW_SIZE];
    for (int i = 0; i < BW_SIZE; i++)
        bw_buf[i] = (uint8_t)(i & 0xFF);

    printk("bandwidth test: sending %d bytes...\n", BW_SIZE);
    uint32_t t0 = timer_get_usec();
    gpio.base.send_raw(&gpio.base, bw_buf, BW_SIZE);
    uint32_t t1 = timer_get_usec();

    uint32_t send_us = t1 - t0;
    uint32_t throughput_kbps = (uint32_t)((uint64_t)BW_SIZE * 1000 / send_us);

    printk("sent %d bytes in %d us = %d KB/s\n", BW_SIZE, send_us, throughput_kbps);

    /* Wait for echo back (bandwidth measurement for recv) */
    uint32_t t2 = timer_get_usec();
    if (gpio.base.recv_raw(&gpio.base, bw_buf, BW_SIZE) < 0) {
        printk("FAIL: timeout on bandwidth recv\n");
        clean_reboot();
    }
    uint32_t t3 = timer_get_usec();

    uint32_t recv_us = t3 - t2;
    uint32_t recv_kbps = (uint32_t)((uint64_t)BW_SIZE * 1000 / recv_us);
    printk("recv %d bytes in %d us = %d KB/s\n", BW_SIZE, recv_us, recv_kbps);

    if (errors == 0)
        printk("=== PASS ===\n");
    else
        printk("=== FAIL ===\n");

    clean_reboot();
}
