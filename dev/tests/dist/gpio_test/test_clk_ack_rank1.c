/*
 * Quick test: read CLK(12) and ACK(13) wires from rank 0.
 */
#include "rpi.h"

void notmain(void) {
    uart_init();
    printk("=== CLK/ACK test rank 1 ===\n");

    gpio_set_input(12);
    gpio_set_input(13);

    for (int s = 0; s < 60; s++) {
        int clk = gpio_read(12);
        int ack = gpio_read(13);
        printk("t=%d: CLK=%d ACK=%d\n", s * 100, clk, ack);
        delay_ms(100);
    }

    printk("=== DONE ===\n");
    clean_reboot();
}
