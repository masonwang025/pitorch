#include "rpi.h"
#include "pt_link.h"

/* Receive bytes over our bit-bang UART on GPIO 27 */
void notmain(void) {
    uart_init();
    printk("=== byte RX test ===\n");

    pt_link_t link = pt_link_init(17, 27, 115200);
    printk("link init done, usec_per_bit_x8=%d\n", link.usec_per_bit_x8);

    printk("waiting for bytes...\n");

    /* Try to receive 4 bytes */
    for (int i = 0; i < 4; i++) {
        uint8_t b;
        int rc = pt_link_recv_raw(&link.base, &b, 1);
        if (rc < 0) {
            printk("byte %d: TIMEOUT\n", i);
        } else {
            printk("byte %d: 0x%x ('%c')\n", i, b, (b >= 32 && b < 127) ? b : '.');
        }
    }

    printk("=== DONE ===\n");
    clean_reboot();
}
