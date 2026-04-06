#include "rpi.h"

/* Pi that drives GPIO 17 — connect to the other Pi's GPIO 27 */
void notmain(void) {
    uart_init();
    printk("=== GPIO TX test ===\n");

    gpio_set_output(17);

    /* Toggle GPIO 17: HIGH for 1s, LOW for 1s, repeat */
    for (int i = 0; i < 5; i++) {
        printk("GPIO 17 = HIGH\n");
        gpio_set_on(17);
        delay_ms(1000);

        printk("GPIO 17 = LOW\n");
        gpio_set_off(17);
        delay_ms(1000);
    }

    printk("=== DONE ===\n");
    clean_reboot();
}
