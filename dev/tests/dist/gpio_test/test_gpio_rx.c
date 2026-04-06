#include "rpi.h"

/* Pi that reads GPIO 27 — connected to the other Pi's GPIO 17 */
void notmain(void) {
    uart_init();
    printk("=== GPIO RX test ===\n");

    gpio_set_input(27);

    /* Sample GPIO 27 every 200ms for 10s */
    for (int i = 0; i < 50; i++) {
        int val = gpio_read(27);
        printk("t=%d: GPIO 27 = %d\n", i * 200, val);
        delay_ms(200);
    }

    printk("=== DONE ===\n");
    clean_reboot();
}
