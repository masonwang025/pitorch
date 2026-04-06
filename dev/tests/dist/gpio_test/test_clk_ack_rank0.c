/*
 * Quick test: verify CLK(24) and ACK(25) wires only.
 */
#include "rpi.h"

void notmain(void) {
    uart_init();
    printk("=== CLK/ACK test rank 0 ===\n");

    gpio_set_output(24);
    gpio_set_output(25);
    gpio_set_off(24);
    gpio_set_off(25);

    delay_ms(2000);

    printk("CLK(24) HIGH\n");
    gpio_set_on(24);
    delay_ms(500);
    printk("CLK(24) LOW\n");
    gpio_set_off(24);
    delay_ms(500);

    printk("ACK(25) HIGH\n");
    gpio_set_on(25);
    delay_ms(500);
    printk("ACK(25) LOW\n");
    gpio_set_off(25);
    delay_ms(500);

    printk("BOTH HIGH\n");
    gpio_set_on(24);
    gpio_set_on(25);
    delay_ms(500);
    printk("BOTH LOW\n");
    gpio_set_off(24);
    gpio_set_off(25);
    delay_ms(500);

    printk("=== DONE ===\n");
    clean_reboot();
}
