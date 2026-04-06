// Pi-side bootloader protocol implementation.
// The includer must provide: boot_get8, boot_put8, boot_has_data.
#ifndef __GETCODE_H__
#define __GETCODE_H__
#include "rpi.h"
#include "boot-crc32.h"
#include "boot-defs.h"

static inline uint32_t boot_get32(void) {
    uint32_t u = boot_get8();
    u |= (uint32_t)boot_get8() << 8;
    u |= (uint32_t)boot_get8() << 16;
    u |= (uint32_t)boot_get8() << 24;
    return u;
}

static inline void boot_put32(uint32_t u) {
    boot_put8((u >> 0) & 0xff);
    boot_put8((u >> 8) & 0xff);
    boot_put8((u >> 16) & 0xff);
    boot_put8((u >> 24) & 0xff);
}

static inline void boot_putk(const char *msg) {
    uint32_t n = strlen(msg);
    if (!n) return;
    boot_put32(PRINT_STRING);
    boot_put32(n);
    for (unsigned i = 0; msg[i]; i++)
        boot_put8(msg[i]);
}

#define boot_todo(msg) \
    boot_err(BOOT_ERROR, __FILE__ ":" LINE_STR() ":TODO:" msg "\n")

static inline void
boot_err(uint32_t error_opcode, const char *msg) {
    boot_putk(msg);
    boot_put32(error_opcode);
    uart_flush_tx();
    rpi_reboot();
}

static unsigned
has_data_timeout(unsigned timeout) {
    unsigned start = timer_get_usec();
    while (1) {
        if (boot_has_data())
            return 1;
        unsigned now = timer_get_usec();
        if ((now - start) >= timeout)
            return 0;
    }
}

static void wait_for_data(unsigned usec_timeout) {
    while (1) {
        boot_put32(GET_PROG_INFO);
        if (has_data_timeout(usec_timeout))
            return;
    }
}

uint32_t get_code(void) {
    wait_for_data(300 * 1000);

    uint32_t addr = 0;

    uint32_t op = boot_get32();
    if (op != PUT_PROG_INFO)
        boot_err(BOOT_ERROR, "expected PUT_PROG_INFO\n");

    addr = boot_get32();
    uint32_t nbytes = boot_get32();
    uint32_t cksum = boot_get32();

    extern uint32_t __prog_end__;
    uint32_t boot_start = (uint32_t)PUT32;
    uint32_t boot_end = (uint32_t)&__prog_end__;
    uint32_t code_start = addr;
    uint32_t code_end = addr + nbytes;

    if ((code_end > boot_start) && (code_start < boot_end))
        boot_err(BOOT_ERROR, "binary would overwrite bootloader!\n");

    boot_put32(GET_CODE);
    boot_put32(cksum);

    op = boot_get32();
    if (op != PUT_CODE)
        boot_err(BOOT_ERROR, "expected PUT_CODE\n");

    for (uint32_t i = 0; i < nbytes; i++) {
        uint8_t b = boot_get8();
        PUT8(addr + i, b);
    }

    boot_putk("bootloader: checksum verified");

    uint32_t computed_crc = crc32((void *)addr, nbytes);
    if (computed_crc != cksum)
        boot_err(BOOT_ERROR, "checksum mismatch!\n");

    boot_putk("pitorch UART bootloader: success!");
    boot_put32(BOOT_SUCCESS);
    uart_flush_tx();

    return addr;
}
#endif
