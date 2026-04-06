/*
 * Pin sweep test — rank 1 (reader).
 *
 * Reads the 10 GPIO pins that rank 0 is driving:
 *   Data: rank1 GPIO 4-11  (driven by rank0 GPIO 16-23)
 *   CLK:  rank1 GPIO 12    (driven by rank0 GPIO 24)
 *   ACK:  rank1 GPIO 13    (driven by rank0 GPIO 25)
 *
 * Samples all 10 pins every 100ms and prints a compact status line.
 * Expected: see each pin go HIGH then LOW in sequence.
 */
#include "rpi.h"

static int pins[] = { 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
static const char *names[] = {
    "D0(4)", "D1(5)", "D2(6)", "D3(7)",
    "D4(8)", "D5(9)", "D6(10)", "D7(11)",
    "CLK(12)", "ACK(13)"
};
#define N_PINS 10
#define N_SAMPLES 250   /* 25 seconds at 100ms intervals */

void notmain(void) {
    uart_init();
    printk("=== pin sweep rank 1 (reader) ===\n");

    /* Set all 10 pins as input */
    for (int i = 0; i < N_PINS; i++)
        gpio_set_input(pins[i]);

    printk("reading %d pins every 100ms for %d samples\n", N_PINS, N_SAMPLES);
    printk("pins: ");
    for (int i = 0; i < N_PINS; i++) printk("%s ", names[i]);
    printk("\n");

    int prev_state = 0;  /* bitmask of previous pin states */

    for (int s = 0; s < N_SAMPLES; s++) {
        int state = 0;
        for (int i = 0; i < N_PINS; i++) {
            if (gpio_read(pins[i]))
                state |= (1 << i);
        }

        /* Only print when state changes (reduce spam) */
        if (state != prev_state || s == 0) {
            printk("t=%4d: ", s * 100);
            for (int i = 0; i < N_PINS; i++)
                printk("%d", (state >> i) & 1);
            printk("  (");
            /* Print which pins are HIGH */
            int first = 1;
            for (int i = 0; i < N_PINS; i++) {
                if ((state >> i) & 1) {
                    if (!first) printk(",");
                    printk("%s", names[i]);
                    first = 0;
                }
            }
            if (first) printk("none");
            printk(")\n");
            prev_state = state;
        }

        delay_ms(100);
    }

    printk("=== rank 1 DONE ===\n");
    clean_reboot();
}
