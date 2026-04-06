// Host-side UART bootloader driver.
// Sends compiled binaries to the Pi over tty-USB.
// Based on CS140E lab 6 (engler).
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <string.h>
#include <sys/stat.h>
#include <termios.h>

#include "libunix.h"
#include "put-code.h"

static char *progname = 0;

static void usage(const char *msg, ...)
{
    va_list args;
    va_start(args, msg);
    vfprintf(stderr, msg, args);
    va_end(args);

    output("\nusage: %s [--trace-all] [--trace-control] [--baud <rate>] [--addr <addr>] ([device] | [--last] | [--first] [--device <device>]) <pi-program>\n", progname);
    output("    pi-program = has a '.bin' suffix\n");
    output("    specify a device using any method:\n");
    output("        <device>: has a '/dev' prefix\n");
    output("       --last: gets the last serial device mounted\n");
    output("        --first: gets the first serial device mounted\n");
    output("        --device <device>: manually specify <device>\n");
    output("    --baud <baud_rate>: manually specify baud_rate\n");
    output("    --addr <addr>: set load/jump address (e.g., 0x8000)\n");
    output("    --trace-all: trace all put/get between rpi and unix side\n");
    output("    --trace-control: trace only control [no data] messages\n");
    exit(1);
}

static unsigned long parse_ul_or_die(const char *s, const char *flag)
{
    errno = 0;
    char *end = 0;
    unsigned long v = strtoul(s, &end, 0);
    if (end == s || *end != '\0')
        usage("%s: invalid number: <%s>\n", flag, s);
    if (errno == ERANGE || v > UINT_MAX)
        usage("%s: number out of range: <%s>\n", flag, s);
    return v;
}

static unsigned parse_baud_or_die(const char *s)
{
    if (s[0] == 'B')
        s++;
    unsigned long rate = parse_ul_or_die(s, "--baud");
    /* macOS cfsetspeed() accepts raw baud rates directly */
    if (rate == 9600 || rate == 115200 || rate == 230400 ||
        rate == 460800 || rate == 576000 || rate == 921600 ||
        rate == 1000000)
        return (unsigned)rate;
    usage("--baud: unsupported rate <%s> (try 115200)\n", s);
    return 115200;
}

int main(int argc, char *argv[])
{
    char *dev_name = 0;
    char *pi_prog = 0;

    unsigned baud_rate = 460800;
    unsigned boot_addr = ARMBASE;

    progname = argv[0];
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--trace-control") == 0)
        {
            trace_p = TRACE_CONTROL_ONLY;
        }
        else if (strcmp(argv[i], "--trace-all") == 0)
        {
            trace_p = TRACE_ALL;
        }
        else if (strcmp(argv[i], "--last") == 0)
        {
            dev_name = find_ttyusb_last();
        }
        else if (strcmp(argv[i], "--first") == 0)
        {
            dev_name = find_ttyusb_first();
        }
        else if (prefix_cmp(argv[i], "/dev/"))
        {
            dev_name = argv[i];
        }
        else if (suffix_cmp(argv[i], ".bin"))
        {
            pi_prog = argv[i];
        }
        else if (strcmp(argv[i], "--baud") == 0)
        {
            i++;
            if (!argv[i])
                usage("missing argument to --baud\n");
            baud_rate = parse_baud_or_die(argv[i]);
        }
        else if (strcmp(argv[i], "--addr") == 0)
        {
            i++;
            if (!argv[i])
                usage("missing argument to --addr\n");
            boot_addr = parse_ul_or_die(argv[i], "--addr");
            if (boot_addr == 0 || boot_addr >= (512u * 1024u * 1024u))
                usage("--addr has invalid address: %x\n", boot_addr);
        }
        else if (strcmp(argv[i], "--device") == 0)
        {
            i++;
            if (!argv[i])
                usage("missing argument to --device\n");
            dev_name = argv[i];
        }
        else
        {
            usage("unexpected argument=<%s>\n", argv[i]);
        }
    }
    if (!pi_prog)
        usage("no pi program\n");

    if (!dev_name)
    {
        dev_name = find_ttyusb_last();
        if (!dev_name)
            panic("didn't find a device\n");
    }
    debug_output("done with options: dev name=<%s>, pi-prog=<%s>, trace=%d\n",
                 dev_name, pi_prog, trace_p);

    int tty = open_tty(dev_name);
    if (tty < 0)
        panic("can't open tty <%s>\n", dev_name);

    // At 921600 baud the UART transmits ~92,160 bytes/sec.
    // Even large binaries transfer in well under 1s.
    double timeout_secs = 2.0;
    int fd = set_tty_to_8n1(tty, baud_rate, timeout_secs);
    if (fd < 0)
        panic("could not set tty: <%s>\n", dev_name);

    unsigned nbytes;
    uint8_t *code = read_file(&nbytes, pi_prog);

    debug_output("%s: tty-usb=<%s> program=<%s>: about to boot\n",
                 progname, dev_name, pi_prog);
    simple_boot(fd, boot_addr, code, nbytes);

    pi_echo(0, fd, dev_name);
    return 0;
}
