# Bootloader

UART bootloader for Raspberry Pi Zero, based on the [CS140E bootloader lab](https://github.com/dddrrreee/cs140e-26win/tree/main/labs/6-bootloader). Two pieces:

- **`my-install`** (unix-side) — host program that sends `.bin` files to the Pi over USB-UART.
- **`kernel.img`** (pi-side) — firmware that runs on the Pi, receives code over UART, and jumps to it.

## Build

Requires `arm-none-eabi-gcc` on your PATH (see main README for install).

```bash
# Build the host-side loader
make my-install

# Copy it into bin/ so pitorch Makefiles can find it
make install

# Build the SD-card firmware (only needed if you're re-flashing)
make kernel.img
```

## Configuration

**Timeout:** The read timeout (VTIME) controls how long the host waits for each UART response from the Pi. At 460800 baud, the UART transmits ~46,080 bytes/sec, so the default 2-second timeout handles most binaries. To change it, edit `timeout_secs` in `unix-side/my-install.c`.

**Baud rate:** Default is 460800. Override with `my-install --baud <rate> prog.bin`.

## Structure

```
bootloader/
├── Makefile              # top-level: make my-install / make kernel.img
├── pi-side/              # SD-card firmware (ARM bare-metal)
│   ├── main.c            # receives binary over UART, jumps to it
│   ├── get-code.h        # protocol: wait, receive, verify, boot
│   ├── boot-start.S      # entry point with 2MB gap for received code
│   ├── boot-defs.h       # protocol opcodes (shared with unix-side)
│   └── boot-crc32.h      # CRC32 (shared with unix-side)
└── unix-side/            # host tool (native)
    ├── my-install.c       # CLI driver: parse args, open TTY, send binary
    ├── put-code.c         # protocol: handshake, send code, verify
    ├── put-code.h
    ├── pi-echo.c          # echo Pi UART output to terminal
    ├── set-tty-8n1.c      # configure TTY to 8N1
    └── libunix/           # minimal unix helpers
```
