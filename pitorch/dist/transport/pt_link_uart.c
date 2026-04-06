#include "rpi.h"
#include "pt_link.h"

/* Bit-bang UART: 8N1.
 * Uses BCM2835 system timer (1 MHz, timer_get_usec) for accurate bit timing.
 * Fixed-point arithmetic (x8) for sub-microsecond precision at 115200 baud. */

static inline void wait_until(uint32_t target) {
    while ((int32_t)(target - timer_get_usec()) > 0)
        ;
}

static void link_put8(pt_link_t *link, uint8_t b) {
    unsigned tx = link->tx_pin;
    unsigned bit8 = link->usec_per_bit_x8;
    uint32_t t = timer_get_usec();
    uint32_t phase = 0;

    /* start bit */
    gpio_set_off(tx);
    phase += bit8;
    wait_until(t + (phase >> 3));

    /* 8 data bits, LSB first */
    for (int i = 0; i < 8; i++) {
        if (b & 1)
            gpio_set_on(tx);
        else
            gpio_set_off(tx);
        b >>= 1;
        phase += bit8;
        wait_until(t + (phase >> 3));
    }

    /* stop bit */
    gpio_set_on(tx);
    phase += bit8;
    wait_until(t + (phase >> 3));
}

static int link_get8(pt_link_t *link, uint32_t timeout_usec) {
    unsigned rx = link->rx_pin;
    unsigned bit8 = link->usec_per_bit_x8;

    /* First ensure line is idle HIGH (wait for TX to be ready) */
    uint32_t start = timer_get_usec();
    while (gpio_read(rx) == 0) {
        if (timeout_usec > 0 && (timer_get_usec() - start) > timeout_usec)
            return -1;
    }

    /* Wait for start bit (falling edge: line goes LOW) */
    while (gpio_read(rx) != 0) {
        if (timeout_usec > 0 && (timer_get_usec() - start) > timeout_usec)
            return -1;
    }

    /* At beginning of start bit. Advance to middle (half bit). */
    uint32_t t = timer_get_usec();
    uint32_t phase = bit8 / 2;
    wait_until(t + (phase >> 3));

    /* Verify still LOW (valid start bit) */
    if (gpio_read(rx) != 0)
        return -1;

    /* Read 8 data bits, LSB first */
    uint8_t b = 0;
    for (int i = 0; i < 8; i++) {
        phase += bit8;
        wait_until(t + (phase >> 3));
        if (gpio_read(rx))
            b |= (1 << i);
    }

    /* Wait through stop bit */
    phase += bit8;
    wait_until(t + (phase >> 3));

    return b;
}

void pt_link_send_raw(pt_transport_t *t, const void *buf, unsigned len) {
    pt_link_t *link = (pt_link_t *)t;
    const uint8_t *p = (const uint8_t *)buf;
    for (unsigned i = 0; i < len; i++)
        link_put8(link, p[i]);
}

int pt_link_recv_raw(pt_transport_t *t, void *buf, unsigned len) {
    pt_link_t *link = (pt_link_t *)t;
    uint8_t *p = (uint8_t *)buf;
    for (unsigned i = 0; i < len; i++) {
        int c = link_get8(link, 10 * 1000 * 1000);
        if (c < 0) return -1;
        p[i] = (uint8_t)c;
    }
    return 0;
}
