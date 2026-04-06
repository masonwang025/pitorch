#include "rpi.h"
#include "pt_link.h"

/* Send a single known byte over our bit-bang UART on GPIO 17 */
void notmain(void) {
    uart_init();
    printk("=== byte TX test ===\n");

    pt_link_t link = pt_link_init(17, 27, 115200);
    printk("link init done, usec_per_bit_x8=%d\n", link.usec_per_bit_x8);

    /* Wait a moment for receiver to be ready */
    delay_ms(2000);

    /* Send 0xAA (alternating bits: 01010101 on wire since LSB first) */
    printk("sending 0xAA...\n");
    pt_link_send_raw(&link.base, "\xAA", 1);
    delay_ms(100);

    /* Send 0x55 */
    printk("sending 0x55...\n");
    pt_link_send_raw(&link.base, "\x55", 1);
    delay_ms(100);

    /* Send 0xFF */
    printk("sending 0xFF...\n");
    pt_link_send_raw(&link.base, "\xFF", 1);
    delay_ms(100);

    /* Send "Hi" */
    printk("sending 'Hi'...\n");
    pt_link_send_raw(&link.base, "Hi", 2);
    delay_ms(100);

    printk("=== DONE ===\n");
    clean_reboot();
}
