// Pi-side UART bootloader.
// Receives a binary over UART and jumps to it.
// Based on CS140E lab 6.
#include "rpi.h"

static inline int boot_has_data(void) {
    return uart_has_data();
}

static inline uint8_t boot_get8(void) {
    return uart_get8();
}

static void boot_put8(uint8_t x) {
    uart_put8(x);
}

#include "get-code.h"

void notmain(void) {
    uint32_t addr = get_code();
    if(!addr)
        rpi_reboot();
    BRANCHTO(addr);
    not_reached();
}
