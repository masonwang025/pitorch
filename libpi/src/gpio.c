/*
 * Implement the following routines to set GPIO pins to input or
 * output, and to read (input) and write (output) them.
 *  1. DO NOT USE loads and stores directly: only use GET32 and
 *    PUT32 to read and write memory.  See <start.S> for thier
 *    definitions.
 *  2. DO USE the minimum number of such calls.
 * (Both of these matter for the next lab.)
 *
 * See <rpi.h> in this directory for the definitions.
 *  - we use <gpio_panic> to try to catch errors.  For lab 2
 *    it only infinite loops since we don't have <printk>
 */
#include "rpi.h"

// See broadcomm documents for magic addresses and magic values.
//
// If you pass addresses as:
//  - pointers use put32/get32.
//  - integers: use PUT32/GET32.
//  semantics are the same.
enum
{
    // Max gpio pin number.
    GPIO_MAX_PIN = 53,

    GPIO_BASE = 0x20200000,
    gpio_set0 = (GPIO_BASE + 0x1C),
    gpio_clr0 = (GPIO_BASE + 0x28),
    gpio_lev0 = (GPIO_BASE + 0x34)

    // <you will need other values from BCM2835!>
};

// set GPIO function for <pin> (input, output, alt...).
// settings for other pins should be unchanged.
void gpio_set_function(unsigned pin, gpio_func_t func)
{
    if (pin > GPIO_MAX_PIN)
        gpio_panic("illegal pin=%d\n", pin);
    if ((func & 0b111) != func) // 3-bit value
        gpio_panic("illegal func=%x\n", func);

    // using broadcomm p90 to get address
    unsigned index = pin / 10;
    unsigned addr = GPIO_BASE + (index * 4);

    // find the correct 3 bits to clear and update
    unsigned pin_shift = (pin % 10) * 3;
    unsigned mask = 0x7 << pin_shift;

    unsigned value = GET32(addr);
    value &= ~mask;
    value |= (func << pin_shift);
    PUT32(addr, value);
}

//
// Part 1 implement gpio_set_on, gpio_set_off, gpio_set_output
//

// set <pin> to be an output pin.
//
// NOTE: fsel0, fsel1, fsel2 are contiguous in memory, so you
// can (and should) use ptr calculations versus if-statements!
void gpio_set_output(unsigned pin)
{
    if (pin > GPIO_MAX_PIN)
        gpio_panic("illegal pin=%d\n", pin);

    // using broadcomm p90 to get address
    unsigned index = pin / 10;
    unsigned addr = GPIO_BASE + (index * 4);

    // find the correct 3 bits to clear and update, keep everything else in register
    unsigned pin_shift = (pin % 10) * 3;
    unsigned mask = 0x7 << pin_shift;

    unsigned value = GET32(addr);
    value &= ~mask;
    value |= (0x1 << pin_shift); // set to output (001)
    PUT32(addr, value);
}

// Set GPIO <pin> = on.
void gpio_set_on(unsigned pin)
{
    if (pin > GPIO_MAX_PIN)
        gpio_panic("illegal pin=%d\n", pin);

    // broadcomm p90 and p95
    // because SET0 and SET1 are continuous in memory
    unsigned addr = gpio_set0 + ((pin / 32) * 4);
    PUT32(addr, 0x1 << (pin % 32));
}

// Set GPIO <pin> = off
void gpio_set_off(unsigned pin)
{
    if (pin > GPIO_MAX_PIN)
        gpio_panic("illegal pin=%d\n", pin);

    // Implement this.
    // NOTE:
    //  - If you want to be slick, you can exploit the fact that
    //    CLR0/CLR1 are contiguous in memory.

    unsigned addr = gpio_clr0 + ((pin / 32) * 4);
    PUT32(addr, 0x1 << (pin % 32));
}

// Set <pin> to <v> (v \in {0,1})
void gpio_write(unsigned pin, unsigned v)
{
    if (v)
        gpio_set_on(pin);
    else
        gpio_set_off(pin);
}

//
// Part 2: implement gpio_set_input and gpio_read
//

// set <pin> = input.
void gpio_set_input(unsigned pin)
{
    if (pin > GPIO_MAX_PIN)
        gpio_panic("illegal pin=%d\n", pin);

    // using broadcomm p90 to get address
    unsigned index = pin / 10;
    unsigned addr = GPIO_BASE + (index * 4);

    // find the correct 3 bits to clear and update, keep everything else in register
    unsigned pin_shift = (pin % 10) * 3;
    unsigned mask = 0x7 << pin_shift;

    unsigned value = GET32(addr);
    value &= ~mask;
    // by clearing we have 000 in the right spot already
    PUT32(addr, value);
}

// Return 1 if <pin> is on, 0 if not.
int gpio_read(unsigned pin)
{
    unsigned v = 0;

    if (pin > GPIO_MAX_PIN)
        gpio_panic("illegal pin=%d\n", pin);

    // broadcomm p90
    // find the corresponding bit to read from (lev0 and lev1 are continuous)
    unsigned addr = gpio_lev0 + ((pin / 32) * 4);
    unsigned value = GET32(addr);
    // shift to the right and pick off last bit
    v = (value >> (pin % 32)) & 1;

    return v;
}
