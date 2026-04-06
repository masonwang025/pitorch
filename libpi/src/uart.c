// simple mini-uart driver: implement every routine
// with a <todo>.
//
// NOTE:
//  - from broadcom: if you are writing to different
//    devices you MUST use a dev_barrier().
//  - its not always clear when X and Y are different
//    devices.
//  - pay attenton for errata!   there are some serious
//    ones here.  if you have a week free you'd learn
//    alot figuring out what these are (esp hard given
//    the lack of printing) but you'd learn alot, and
//    definitely have new-found respect to the pioneers
//    that worked out the bcm eratta.
//
// historically a problem with writing UART code for
// this class (and for human history) is that when
// things go wrong you can't print since doing so uses
// uart.  thus, debugging is very old school circa
// 1950s, which modern brains arne't built for out of
// the box.   you have two options:
//  1. think hard.  we recommend this.
//  2. use the included bit-banging sw uart routine
//     to print.   this makes things much easier.
//     but if you do make sure you delete it at the
//     end, otherwise your GPIO will be in a bad state.
//
// in either case, in the next part of the lab you'll
// implement bit-banged UART yourself.
#include "rpi.h"

// change "1" to "0" if you want to comment out
// the entire block.
#if 1
//*****************************************************
// We provide a bit-banged version of UART for debugging
// your UART code.  delete when done!
//
// NOTE: if you call <emergency_printk>, it takes
// over the UART GPIO pins (14,15). Thus, your UART
// GPIO initialization will get destroyed.  Do not
// forget!

// header in <libpi/include/sw-uart.h>
#include "sw-uart.h"
static sw_uart_t sw_uart;

// a sw-uart putc implementation.
static int sw_uart_putc(int chr)
{
    sw_uart_put8(&sw_uart, chr);
    return chr;
}

// call this routine to print stuff.
//
// note the function pointer hack: after you call it
// once can call the regular printk etc.
__attribute__((noreturn)) static void emergency_printk(const char *fmt, ...)
{
    // we forcibly initialize in case the
    // GPIO got reset. this will setup
    // gpio 14,15 for sw-uart.
    sw_uart = sw_uart_default();

    // all libpi output is via a <putc>
    // function pointer: this installs ours
    // instead of the default
    rpi_putchar_set(sw_uart_putc);

    // do print
    va_list args;
    va_start(args, fmt);
    vprintk(fmt, args);
    va_end(args);

    // at this point UART is all messed up b/c we took it over
    // so just reboot.   we've set the putchar so this will work
    clean_reboot();
}

#undef todo
#define todo(msg)                                      \
    do                                                 \
    {                                                  \
        emergency_printk("%s:%d:%s\nDONE!!!\n",        \
                         __FUNCTION__, __LINE__, msg); \
    } while (0)

// END of the bit bang code.
#endif

// pg 8-19 (0x7E21xxxx -> 0x2021xxxx)
#define AUX_ENB 0x20215004 // aux enables

#define AUX_MU_IO 0x20215040   // i/o data (rx/tx fifo)
#define AUX_MU_IER 0x20215044  // interr enable
#define AUX_MU_IIR 0x20215048  // interrupt identify (write clears fifo)
#define AUX_MU_LCR 0x2021504c  // line control, note the errata
#define AUX_MU_MCR 0x20215050  // modem control
#define AUX_MU_LSR 0x20215054  // line status
#define AUX_MU_CNTL 0x20215060 // control (enable tx/rx)
#define AUX_MU_STAT 0x20215064 // extra status
#define AUX_MU_BAUD 0x20215068 // baud rate

//*****************************************************
// the rest you should implement.

// called first to setup uart to 8n1 115200  baud,
// no interrupts.
//  - you will need memory barriers, use <dev_barrier()>
//
//  later: should add an init that takes a baud rate.
void uart_init(void)
{
    dev_barrier();

    // GPIO pins FIRST (pg 10 says to do this before UART)
    // set pins to alt5 (pg 102) GPIO 14 = TXD1, GPIO 15 = RXD
    gpio_set_function(GPIO_TX, GPIO_FUNC_ALT5);
    gpio_set_function(GPIO_RX, GPIO_FUNC_ALT5);

    dev_barrier(); // switching from GPIO device to AUX device

    // enable mini uart p.9, (read-modify-write to keep SPI enables)
    unsigned aux_enb = GET32(AUX_ENB);
    aux_enb |= 1; // set bit 0
    PUT32(AUX_ENB, aux_enb);

    dev_barrier(); // now we can access UART registers

    PUT32(AUX_MU_CNTL, 0); // disable both TX and RX while we configure (pg16) to rpevent garbage

    // note errata in the name
    PUT32(AUX_MU_IER, 0);   // pg 12 disable interrutps cause we're polling here
    PUT32(AUX_MU_IIR, 0x6); // clear RX and TX FIFOs (0b110)

    PUT32(AUX_MU_LCR, 0x3); // 8 bit mode (write 0b11 for 8bit huge errata) (pg 14)
    PUT32(AUX_MU_MCR, 0);   // mcr pg 14 we can just set to 0 cause we don't use

    PUT32(AUX_MU_BAUD, 66); // set baudrate to 460800 (pg 19, formula on pg 11)
    PUT32(AUX_MU_CNTL, 0x3); // enable TX and RX (pg 16) - must be LAST to flush buffered writes

    dev_barrier();
}

// disable the uart: make sure all bytes have been
//
void uart_disable(void)
{
    dev_barrier();
    uart_flush_tx();       // wait for all pending transmissions
    PUT32(AUX_MU_CNTL, 0); // disable TX and RX
    dev_barrier();
}

// returns one byte from the RX (input) hardware
// FIFO.  if FIFO is empty, blocks until there is
// at least one byte.
int uart_get8(void)
{
    dev_barrier();
    // STAT register (pg 18)
    while ((GET32(AUX_MU_STAT) & 0x1) == 0) // check bit 0 for RX has data to read
        ;                                   // blocked until RX fifo has data

    int c = GET32(AUX_MU_IO) & 0xFF; // read fro io registeres (0-7)
    // unintuitive, but similar to gpio pins: read and write use two different FIFOs but are represented by the same memory mapped registers
    dev_barrier();
    return c;
}

// returns 1 if the hardware TX (output) FIFO has room
// for at least one byte.  returns 0 otherwise.
int uart_can_put8(void)
{
    // STAT register check for space available (pg 18), looking for bit 1, value 1
    return (GET32(AUX_MU_STAT) & 0x2) != 0;
}

// put one byte on the TX FIFO, if necessary, waits
// until the FIFO has space.
int uart_put8(uint8_t c)
{
    dev_barrier();
    while (!uart_can_put8())
        ; // blocked until tx fifo has space

    PUT32(AUX_MU_IO, c); // pg 11 write to register (only cares about bits 0-7)
    dev_barrier();
    return 1;
}

// returns:
//  - 1 if at least one byte on the hardware RX FIFO.
//  - 0 otherwise
int uart_has_data(void)
{
    // STAT register (p.18): bit 0 = "symbol available" (1 = RX has data)
    return (GET32(AUX_MU_STAT) & 0x1) != 0;
}

// returns:
//  -1 if no data on the RX FIFO.
//  otherwise reads a byte and returns it.
int uart_get8_async(void)
{
    if (!uart_has_data())
        return -1;
    return uart_get8();
}

// returns:
//  - 1 if TX FIFO empty AND idle.
//  - 0 if not empty.
int uart_tx_is_empty(void)
{
    // STAT register (pg 18), bit 8 means tx fifo empty, bit 9 menas transmitter is done
    unsigned stat = GET32(AUX_MU_STAT);
    return (stat & (1 << 9)); // 9 being 1 implies both according to broadcom
}

// return only when the TX FIFO is empty AND the
// TX transmitter is idle.
//
// used when rebooting or turning off the UART to
// make sure that any output has been completely
// transmitted.  otherwise can get truncated
// if reboot happens before all bytes have been
// received.
void uart_flush_tx(void)
{
    while (!uart_tx_is_empty())
        rpi_wait();
}
