#include <stdint.h>
#include "rpi.h"
#include "mailbox.h"

void cache_flush_all(void);

#define MAILBOX_BASE   0x2000B880
#define MAILBOX_READ   (*(volatile uint32_t *)(MAILBOX_BASE + 0x0))
#define MAILBOX_STATUS (*(volatile uint32_t *)(MAILBOX_BASE + 0x18))
#define MAILBOX_WRITE  (*(volatile uint32_t *)(MAILBOX_BASE + 0x20))

#define MAILBOX_FULL   0x80000000
#define MAILBOX_EMPTY  0x40000000

void mailbox_write(uint8_t channel, uint32_t data) {
    while (MAILBOX_STATUS & MAILBOX_FULL)
        ;
    MAILBOX_WRITE = (data & ~0xF) | (channel & 0xF);
}

uint32_t mailbox_read(uint8_t channel) {
    uint32_t data;
    while (1) {
        data = MAILBOX_READ;
        if ((data & 0xF) == channel)
            return data & ~0xF;
    }
}

int mbox_property(uint32_t *msg) {
    if ((uint32_t)msg & 0xF)
        return 0;
    cache_flush_all(); /* write back D-cache so GPU sees msg data */
    mailbox_write(8, (uint32_t)msg);
    while (mailbox_read(8) != (uint32_t)msg)
        ;
    cache_flush_all(); /* invalidate D-cache so ARM sees GPU response */
    return (msg[1] == 0x80000000);
}

uint32_t arm_ram_end(void) {
    uint32_t msg[8] __attribute__((aligned(16))) = {
        8 * sizeof(uint32_t),  /* total size */
        0,                     /* request */
        0x00010005,            /* tag: Get ARM memory */
        2 * sizeof(uint32_t),  /* value buffer size */
        0,                     /* request code */
        0, 0,                  /* base, size (filled by firmware) */
        0                      /* end tag */
    };
    if (!mbox_property(msg))
        panic("ARM memory mailbox query failed");
    return msg[5] + msg[6];
}
