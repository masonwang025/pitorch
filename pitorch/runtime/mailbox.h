#ifndef PITORCH_MAILBOX_H
#define PITORCH_MAILBOX_H

#include <stdint.h>

void mailbox_write(uint8_t channel, uint32_t data);
uint32_t mailbox_read(uint8_t channel);

/*
 * Send a property message on channel 8 and wait for the response.
 * msg must be 16-byte aligned.  Returns nonzero on success.
 */
int mbox_property(uint32_t *msg);

/*
 * Query the firmware for ARM-accessible SDRAM range.
 * Returns base + size (i.e. one past the last usable ARM byte).
 * Panics if the mailbox query fails.
 */
uint32_t arm_ram_end(void);

#endif
