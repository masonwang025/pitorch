#include "rpi.h"
#include "pt_link.h"
#include "pt_proto.h"

#define TX_PIN 17
#define RX_PIN 27
#define BAUD   57600

void notmain(void) {
    uart_init();
    printk("=== uart_ping rank 0 ===\n");

    pt_link_t link = pt_link_init(TX_PIN, RX_PIN, BAUD);
    printk("link ready: TX=GPIO%d RX=GPIO%d baud=%d\n", TX_PIN, RX_PIN, BAUD);

    /* Send PING with 4-byte payload */
    uint32_t ping_val = 0xCAFEBABE;
    printk("sending PING (0x%x)...\n", ping_val);
    pt_proto_send(&link.base, PT_OP_PING, &ping_val, sizeof(ping_val));

    /* Wait for PONG */
    uint32_t opcode, plen;
    uint32_t pong_val;
    if (pt_proto_recv(&link.base, &opcode, &pong_val, sizeof(pong_val), &plen) < 0) {
        printk("FAIL: no PONG received\n");
        clean_reboot();
    }

    if (opcode != PT_OP_PONG) {
        printk("FAIL: expected PONG (opcode %d), got %d\n", PT_OP_PONG, opcode);
        clean_reboot();
    }

    printk("got PONG: 0x%x (len=%d)\n", pong_val, plen);

    if (pong_val == ping_val)
        printk("=== PASS ===\n");
    else
        printk("FAIL: PONG value mismatch\n");

    clean_reboot();
}
