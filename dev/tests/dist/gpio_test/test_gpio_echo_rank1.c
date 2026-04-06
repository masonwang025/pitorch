/*
 * GPIO parallel bus echo test — rank 1 (echoer).
 *
 * Receives data from rank 0, echoes it back verbatim.
 */
#include "rpi.h"
#include "pt_link_gpio.h"
#include "pt_proto.h"

/* Rank 1 uses low bank: data=GPIO 4-11, CLK=12, ACK=13 */
#define D_BASE  4
#define CLK_PIN 12
#define ACK_PIN 13

#define ACTIVATION_SIZE 1152
#define ECHO_ROUNDS     100
#define BW_SIZE         (64 * 1024)

void notmain(void) {
    uart_init();
    printk("=== gpio_echo rank 1 ===\n");

    pt_link_gpio_t gpio = pt_link_gpio_init(D_BASE, CLK_PIN, ACK_PIN);
    printk("gpio link ready: D=%d-%d CLK=%d ACK=%d\n",
           D_BASE, D_BASE + 7, CLK_PIN, ACK_PIN);

    /* Wait for PING from rank 0 */
    uint32_t dummy = 0, opcode, plen;
    printk("waiting for PING...\n");
    if (pt_proto_recv(&gpio.base, &opcode, &dummy, sizeof(dummy), &plen) < 0 ||
        opcode != PT_OP_PING) {
        printk("FAIL: no PING\n");
        clean_reboot();
    }
    printk("got PING, sending PONG...\n");
    pt_proto_send(&gpio.base, PT_OP_PONG, &dummy, sizeof(dummy));
    printk("handshake done!\n");

    /* --- Echo test: recv data, send it back --- */
    uint8_t buf[ACTIVATION_SIZE];

    for (int round = 0; round < ECHO_ROUNDS; round++) {
        uint32_t recv_op, recv_len;
        if (pt_proto_recv(&gpio.base, &recv_op, buf, ACTIVATION_SIZE, &recv_len) < 0) {
            printk("FAIL: timeout on round %d\n", round);
            clean_reboot();
        }

        /* Echo back */
        pt_proto_send(&gpio.base, PT_OP_DATA, buf, recv_len);
    }

    printk("echo: %d rounds complete\n", ECHO_ROUNDS);

    /* --- Bandwidth test: recv 64 KB, echo back --- */
    uint8_t bw_buf[BW_SIZE];

    printk("bandwidth: receiving %d bytes...\n", BW_SIZE);
    if (gpio.base.recv_raw(&gpio.base, bw_buf, BW_SIZE) < 0) {
        printk("FAIL: timeout on bandwidth recv\n");
        clean_reboot();
    }
    printk("received, echoing back...\n");
    gpio.base.send_raw(&gpio.base, bw_buf, BW_SIZE);

    printk("=== DONE ===\n");
    clean_reboot();
}
