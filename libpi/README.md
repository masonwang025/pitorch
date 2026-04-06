# libpi

Bare-metal Raspberry Pi library, originally from Stanford CS140E.

Upstream: [dddrrreee/cs140e-26win/libpi](https://github.com/dddrrreee/cs140e-26win/tree/main/libpi)

## What this is

libpi provides the full bare-metal runtime for the Raspberry Pi Zero: boot code, UART, GPIO, timers, printing, memory allocation, exception handling, and a minimal libc. PiTorch builds on top of it.

## Key files

| File / Directory | Purpose |
|-----------------|---------|
| `staff-start.S` | ARM bootstrap assembly — sets up stack, zeroes BSS, calls `notmain()` |
| `memmap` | Linker script — places `.text` at 0x8000, defines heap start |
| `defs.mk` | Toolchain definitions: `arm-none-eabi-gcc`, CFLAGS, include paths |
| `libpi.a` | The compiled static library (rebuilt by `make`) |
| `Makefile` | Builds libpi.a from all sources below |
| `include/` | Public headers: `rpi.h` (main API), `gpio.h`, interrupts, timers |
| `libc/` | Minimal libc: `printk`, `memcpy`, `memset`, string operations |
| `staff-src/` | Staff-provided source: timers, exception handlers, cache control, reboot |
| `staff-objs/` | Prebuilt objects: `staff-kmalloc.o` (allocator), UART, GPIO helpers |
| `src/` | Student-implemented: `gpio.c`, `uart.c` |
| `mk/` | Makefile templates used by test programs (`Makefile.robust-v2`, etc.) |
| `manifest.mk` | Build manifest — configures sources and includes for libpi itself |

## Building

```bash
make        # builds libpi.a
make clean  # removes build artifacts
```

## API surface used by PiTorch

PiTorch primarily uses: `printk`, `panic`, `assert`, `PUT32`/`GET32` (memory-mapped I/O), `timer_get_usec`, `kmalloc`, `delay_*`, and the UART for serial output. All accessed via `#include "rpi.h"`.
