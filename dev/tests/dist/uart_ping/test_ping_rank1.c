#include "rpi.h"
#include "pt_link.h"
#include "pt_proto.h"

#define TX_PIN 17
#define RX_PIN 27
#define BAUD   57600

void notmain(void) {
    uart_init();
    printk("=== uart_ping rank 1 ===\n");

    pt_link_t link = pt_link_init(TX_PIN, RX_PIN, BAUD);
    printk("link ready: TX=GPIO%d RX=GPIO%d baud=%d\n", TX_PIN, RX_PIN, BAUD);

    /* Wait for PING */
    uint32_t opcode, plen;
    uint32_t val;
    printk("waiting for PING...\n");
    if (pt_proto_recv(&link.base, &opcode, &val, sizeof(val), &plen) < 0) {
        printk("FAIL: no PING received\n");
        clean_reboot();
    }

    if (opcode != PT_OP_PING) {
        printk("FAIL: expected PING (opcode %d), got %d\n", PT_OP_PING, opcode);
        clean_reboot();
    }

    printk("got PING: 0x%x (len=%d)\n", val, plen);

    /* Send PONG with same payload */
    printk("sending PONG...\n");
    pt_proto_send(&link.base, PT_OP_PONG, &val, sizeof(val));

    printk("=== DONE ===\n");
    clean_reboot();
}
