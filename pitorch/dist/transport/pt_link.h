#ifndef PT_LINK_H
#define PT_LINK_H

#include "rpi.h"
#include "pt_transport.h"

typedef struct {
    pt_transport_t base;   /* must be first — castable to pt_transport_t* */
    unsigned tx_pin;
    unsigned rx_pin;
    unsigned usec_per_bit_x8;  /* microseconds per bit * 8, for sub-us precision */
} pt_link_t;

/* Raw send/recv — declared in pt_link_uart.c */
void pt_link_send_raw(pt_transport_t *t, const void *buf, unsigned len);
int  pt_link_recv_raw(pt_transport_t *t, void *buf, unsigned len);

/* Initialize link on GPIO tx_pin/rx_pin at given baud rate. */
static inline pt_link_t pt_link_init(unsigned tx_pin, unsigned rx_pin, unsigned baud) {
    pt_link_t link;
    link.base.send_raw = pt_link_send_raw;
    link.base.recv_raw = pt_link_recv_raw;
    link.tx_pin = tx_pin;
    link.rx_pin = rx_pin;
    /* Fixed-point: 8 * 1000000 / baud.  At 115200: 69 (≈ 8.625 us * 8) */
    link.usec_per_bit_x8 = (8 * 1000 * 1000) / baud;

    gpio_set_output(tx_pin);
    gpio_set_on(tx_pin);   /* idle HIGH */
    gpio_set_input(rx_pin);

    return link;
}

#endif
