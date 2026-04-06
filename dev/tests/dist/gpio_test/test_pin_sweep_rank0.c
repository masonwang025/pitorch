/*
 * Pin sweep test — rank 0 (driver).
 *
 * Tests each of the 10 GPIO wires individually:
 *   Data: rank0 GPIO 16-23 → rank1 GPIO 4-11
 *   CLK:  rank0 GPIO 24    → rank1 GPIO 12
 *   ACK:  rank0 GPIO 25    → rank1 GPIO 13
 *
 * For each pin: set HIGH, wait 200ms, set LOW, wait 200ms.
 * Rank 1 reads and reports what it sees.
 */
#include "rpi.h"

static int pins[] = { 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 };
static const char *names[] = {
    "D0(16)", "D1(17)", "D2(18)", "D3(19)",
    "D4(20)", "D5(21)", "D6(22)", "D7(23)",
    "CLK(24)", "ACK(25)"
};
#define N_PINS 10

void notmain(void) {
    uart_init();
    printk("=== pin sweep rank 0 (driver) ===\n");

    /* Set all 10 pins as output, drive LOW initially */
    for (int i = 0; i < N_PINS; i++) {
        gpio_set_output(pins[i]);
        gpio_set_off(pins[i]);
    }

    /* Wait for rank 1 to be ready */
    delay_ms(2000);

    /* Sweep each pin */
    for (int i = 0; i < N_PINS; i++) {
        printk("testing %s: HIGH\n", names[i]);
        gpio_set_on(pins[i]);
        delay_ms(500);

        printk("testing %s: LOW\n", names[i]);
        gpio_set_off(pins[i]);
        delay_ms(500);
    }

    /* All-on test: drive all 10 HIGH at once */
    printk("ALL HIGH\n");
    for (int i = 0; i < N_PINS; i++)
        gpio_set_on(pins[i]);
    delay_ms(500);

    printk("ALL LOW\n");
    for (int i = 0; i < N_PINS; i++)
        gpio_set_off(pins[i]);
    delay_ms(500);

    printk("=== rank 0 DONE ===\n");
    clean_reboot();
}
