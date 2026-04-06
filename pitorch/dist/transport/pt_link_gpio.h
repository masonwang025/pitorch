#ifndef PT_LINK_GPIO_H
#define PT_LINK_GPIO_H

#include "rpi.h"
#include "pt_transport.h"

/*
 * GPIO parallel bus: 8 data pins + CLK + ACK = 10 wires per edge.
 * Half-duplex: same wires, direction flips between send and recv.
 *
 * Each Pi uses one bank of pins per edge:
 *   Pi 0 (rank 0) downstream: high bank — data=GPIO 16-23, CLK=24, ACK=25
 *   Pi 1 (rank 1) upstream:   low bank  — data=GPIO 4-11,  CLK=12, ACK=13
 *
 * Protocol per byte (4-phase handshake):
 *   Sender: write D0-D7, raise CLK, wait ACK HIGH, lower CLK, wait ACK LOW
 *   Receiver: wait CLK HIGH, read D0-D7, raise ACK, wait CLK LOW, lower ACK
 */

/* BCM2835 GPIO register addresses (all pins 0-31 are in bank 0) */
#define PT_GPIO_SET0  0x2020001Cu
#define PT_GPIO_CLR0  0x20200028u
#define PT_GPIO_LEV0  0x20200034u

typedef struct {
    pt_transport_t base;     /* must be first — castable to pt_transport_t* */
    unsigned d_base;         /* first data pin (16 for high bank, 4 for low bank) */
    unsigned clk_pin;        /* CLK pin (24 or 12) */
    unsigned ack_pin;        /* ACK pin (25 or 13) */
    uint32_t data_mask;      /* 0xFF << d_base */
    uint32_t clk_mask;       /* 1 << clk_pin */
    uint32_t ack_mask;       /* 1 << ack_pin */
    int mode;                /* 0=idle, 1=send, 2=recv — avoids redundant mode switches */
} pt_link_gpio_t;

/* Raw send/recv — implemented in pt_link_gpio.c */
void pt_link_gpio_send_raw(pt_transport_t *t, const void *buf, unsigned len);
int  pt_link_gpio_recv_raw(pt_transport_t *t, void *buf, unsigned len);

/* Initialize a GPIO parallel link for a given edge.
 * For rank 0: pt_link_gpio_init(16, 24, 25)  — high bank
 * For rank 1: pt_link_gpio_init(4, 12, 13)   — low bank */
static inline pt_link_gpio_t pt_link_gpio_init(unsigned d_base, unsigned clk_pin, unsigned ack_pin) {
    pt_link_gpio_t g;
    g.base.send_raw = pt_link_gpio_send_raw;
    g.base.recv_raw = pt_link_gpio_recv_raw;
    g.d_base   = d_base;
    g.clk_pin  = clk_pin;
    g.ack_pin  = ack_pin;
    g.data_mask = 0xFFu << d_base;
    g.clk_mask  = 1u << clk_pin;
    g.ack_mask  = 1u << ack_pin;
    g.mode      = 0;  /* idle */

    /* Start with all pins as input (safe idle state) */
    for (int i = 0; i < 8; i++)
        gpio_set_input(d_base + i);
    gpio_set_input(clk_pin);
    gpio_set_input(ack_pin);

    return g;
}

#endif
