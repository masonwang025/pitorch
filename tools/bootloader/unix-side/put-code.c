// Unix-side bootloader protocol implementation.
// Based on CS140E lab 6 (engler).
#include <string.h>
#include "put-code.h"
#include "boot-crc32.h"
#include "boot-defs.h"

#ifndef __RPI__
void put_uint8(int fd, uint8_t b) { write_exact(fd, &b, 1); }
void put_uint32(int fd, uint32_t u) { write_exact(fd, &u, 4); }

uint8_t get_uint8(int fd)
{
    uint8_t b;
    int res;
    if ((res = read(fd, &b, 1)) < 0)
        die("my-install: tty-USB read() returned error=%d (%s): disconnected?\n", res, strerror(res));
    if (res == 0)
        die("my-install: tty-USB read() returned 0 bytes.  r/pi not responding [reboot it?]\n");
    assert(res == 1);
    return b;
}

uint32_t get_uint32(int fd)
{
    uint32_t u;
    u = get_uint8(fd);
    u |= (uint32_t)get_uint8(fd) << 8;
    u |= (uint32_t)get_uint8(fd) << 16;
    u |= (uint32_t)get_uint8(fd) << 24;
    return u;
}
#endif

#define boot_output(msg...) output("BOOT:" msg)

int trace_p = 0;

static inline uint8_t trace_get8(int fd)
{
    uint8_t v = get_uint8(fd);
    if (trace_p)
        trace("GET8:%x\n", v);
    return v;
}

static inline uint32_t trace_get32(int fd)
{
    uint32_t v = get_uint32(fd);
    if (trace_p)
        trace("GET32:%x [%s]\n", v, boot_op_to_str(v));
    return v;
}

static inline void trace_put8(int fd, uint8_t v)
{
    if (trace_p == TRACE_ALL)
        trace("PUT8:%x\n", v);
    put_uint8(fd, v);
}

static inline void trace_put32(int fd, uint32_t v)
{
    if (trace_p)
        trace("PUT32:%x [%s]\n", v, boot_op_to_str(v));
    put_uint32(fd, v);
}

static inline uint32_t get_op(int fd)
{
    while (1)
    {
        uint32_t op = get_uint32(fd);
        if (op != PRINT_STRING)
        {
            if (trace_p)
                trace("GET32:%x [%s]\n", op, boot_op_to_str(op));
            return op;
        }

        debug_output("PRINT_STRING:");
        unsigned nbytes = get_uint32(fd);
        if (!nbytes)
            panic("sent a PRINT_STRING with zero bytes\n");
        if (nbytes > 1024)
            panic("pi sent a suspiciously long string nbytes=%d\n", nbytes);

        output("pi sent print: <");
        for (unsigned i = 0; i < nbytes - 1; i++)
            output("%c", get_uint8(fd));

        uint8_t c = get_uint8(fd);
        if (c != '\n')
            output("%c", c);
        output(">\n");
    }
}

static void
boot_check(int fd, const char *msg, unsigned exp, unsigned got)
{
    if (exp == got)
        return;

    output("%s: expected %x [%s], got %x [%s]\n", msg,
           exp, boot_op_to_str(exp),
           got, boot_op_to_str(got));

#ifndef __RPI__
    unsigned char b;
    while (fd != TRACE_FD && read(fd, &b, 1) == 1)
        fprintf(stderr, "%c [%d]", b, b);
#endif
    panic("pi-boot failed\n");
}

void simple_boot(int fd, uint32_t boot_addr, const uint8_t *buf, unsigned n)
{
    trace("simple_boot: sending %d bytes, crc32=%x\n", n, crc32(buf, n));
    boot_output("waiting for a start\n");

    uint32_t op;

    while ((op = get_op(fd)) != GET_PROG_INFO)
    {
        output("expected initial GET_PROG_INFO, got <%x>: discarding.\n", op);
        get_uint8(fd);
    }

    trace_put32(fd, PUT_PROG_INFO);
    trace_put32(fd, boot_addr);
    trace_put32(fd, n);
    trace_put32(fd, crc32(buf, n));

    while ((op = get_op(fd)) == GET_PROG_INFO)
    {
    }

    boot_check(fd, "expected GET_CODE", GET_CODE, op);

    uint32_t received_crc = trace_get32(fd);
    boot_check(fd, "expected matching crc", crc32(buf, n), received_crc);

    trace_put32(fd, PUT_CODE);
    for (unsigned i = 0; i < n; i++)
        trace_put8(fd, buf[i]);

    op = get_op(fd);
    boot_check(fd, "expected BOOT_SUCCESS", BOOT_SUCCESS, op);

    boot_output("bootloader: Done.\n");
}
