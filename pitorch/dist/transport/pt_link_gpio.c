#include "rpi.h"
#include "pt_link_gpio.h"

/*
 * GPIO parallel bus: 8-bit data + CLK/ACK handshake.
 *
 * Uses direct register access (PUT32/GET32) in the hot loop for speed.
 * Direction is configured at the start of each send/recv call.
 *
 * 4-phase handshake per byte:
 *   Sender: put data → raise CLK → wait ACK HIGH → lower CLK → wait ACK LOW
 *   Receiver: wait CLK HIGH → read data → raise ACK → wait CLK LOW → lower ACK
 *
 * Mode tracking: consecutive calls in the same direction skip mode switching
 * to avoid floating CLK/ACK lines between chunks (which can cause phantom
 * handshake signals and desync).
 */

static void set_idle(pt_link_gpio_t *g);

/* Configure pins for sending: data+CLK = output (driven LOW), ACK = input */
static void set_send_mode(pt_link_gpio_t *g) {
    if (g->mode == 1) return;  /* already in send mode */
    if (g->mode == 2) set_idle(g);  /* release recv pins before taking send */
    for (int i = 0; i < 8; i++)
        gpio_set_output(g->d_base + i);
    gpio_set_output(g->clk_pin);
    gpio_set_input(g->ack_pin);

    /* Ensure clean starting state: all outputs LOW */
    PUT32(PT_GPIO_CLR0, g->data_mask | g->clk_mask);
    dev_barrier();
    delay_us(10);
    g->mode = 1;
}

/* Configure pins for receiving: data+CLK = input, ACK = output (driven LOW) */
static void set_recv_mode(pt_link_gpio_t *g) {
    if (g->mode == 2) return;  /* already in recv mode */
    if (g->mode == 1) set_idle(g);  /* release send pins before taking recv */
    for (int i = 0; i < 8; i++)
        gpio_set_input(g->d_base + i);
    gpio_set_input(g->clk_pin);
    gpio_set_output(g->ack_pin);

    /* Ensure ACK starts LOW */
    PUT32(PT_GPIO_CLR0, g->ack_mask);
    dev_barrier();
    delay_us(10);
    g->mode = 2;
}

/* Release all pins back to input (safe idle state) */
static void set_idle(pt_link_gpio_t *g) {
    if (g->mode == 0) return;  /* already idle */
    /* Drive everything LOW before releasing */
    PUT32(PT_GPIO_CLR0, g->data_mask | g->clk_mask | g->ack_mask);
    dev_barrier();
    for (int i = 0; i < 8; i++)
        gpio_set_input(g->d_base + i);
    gpio_set_input(g->clk_pin);
    gpio_set_input(g->ack_pin);
    dev_barrier();
    /* Let GPIO hardware settle after direction change */
    delay_us(10);
    g->mode = 0;
}

void pt_link_gpio_send_raw(pt_transport_t *t, const void *buf, unsigned len) {
    pt_link_gpio_t *g = (pt_link_gpio_t *)t;
    const uint8_t *p = (const uint8_t *)buf;
    unsigned d_base = g->d_base;
    uint32_t clk_mask = g->clk_mask;
    uint32_t ack_mask = g->ack_mask;

    set_send_mode(g);

    /* 300s timeout on first byte's ACK — prevents permanent hang if peer is dead.
     * 110M head_bwd (768→32000 weight gradient) takes ~150s on ARM1176,
     * and downstream ranks must wait for the full backward pipeline. */
    uint32_t t0 = timer_get_usec();
    uint32_t timeout_us = 300 * 1000 * 1000;

    for (unsigned i = 0; i < len; i++) {
        uint32_t byte = p[i];

        /* Write 8 data bits to D0-D7 atomically */
        uint32_t set_bits = byte << d_base;
        uint32_t clr_bits = (~byte & 0xFF) << d_base;
        PUT32(PT_GPIO_SET0, set_bits);
        PUT32(PT_GPIO_CLR0, clr_bits);

        /* Raise CLK — signals "data ready" */
        PUT32(PT_GPIO_SET0, clk_mask);

        /* Wait for receiver to assert ACK HIGH */
        while (!(GET32(PT_GPIO_LEV0) & ack_mask)) {
            if (i == 0 && (timer_get_usec() - t0) > timeout_us) {
                printk("gpio_send: timeout waiting for ACK (byte 0 of %d)\n", len);
                set_idle(g);
                clean_reboot();
            }
        }

        /* Lower CLK — signals "acknowledged" */
        PUT32(PT_GPIO_CLR0, clk_mask);

        /* Wait for receiver to deassert ACK LOW */
        while (GET32(PT_GPIO_LEV0) & ack_mask)
            ;
    }

    /* Don't go idle — stay in send mode for the next call.
     * The caller (e.g. proto layer) will trigger a mode change when
     * switching direction, which forces the proper reconfiguration. */
}

int pt_link_gpio_recv_raw(pt_transport_t *t, void *buf, unsigned len) {
    pt_link_gpio_t *g = (pt_link_gpio_t *)t;
    uint8_t *p = (uint8_t *)buf;
    unsigned d_base = g->d_base;
    uint32_t clk_mask = g->clk_mask;
    uint32_t ack_mask = g->ack_mask;

    set_recv_mode(g);

    /* Timeout: 300 seconds for first byte, then no timeout (sender is committed) */
    uint32_t t0 = timer_get_usec();
    uint32_t timeout_us = 300 * 1000 * 1000;

    for (unsigned i = 0; i < len; i++) {
        /* Wait for CLK HIGH (sender has put data on bus) */
        while (!(GET32(PT_GPIO_LEV0) & clk_mask)) {
            if (i == 0 && (timer_get_usec() - t0) > timeout_us) {
                set_idle(g);
                return -1;
            }
        }

        /* Read 8 data bits from D0-D7 */
        uint32_t lev = GET32(PT_GPIO_LEV0);
        p[i] = (lev >> d_base) & 0xFF;

        /* Raise ACK — signals "data received" */
        PUT32(PT_GPIO_SET0, ack_mask);

        /* Wait for CLK LOW (sender acknowledges our ACK) */
        while (GET32(PT_GPIO_LEV0) & clk_mask)
            ;

        /* Lower ACK — ready for next byte */
        PUT32(PT_GPIO_CLR0, ack_mask);
    }

    /* Don't go idle — stay in recv mode for the next call. */
    return 0;
}
