# Setup

Getting from a fresh clone to running inference on the Pi Zero.

## What you need

- Raspberry Pi Zero (ARM1176 + VideoCore IV)
- USB-UART adapter
- Micro-SD card (FAT32, ≤32 GB)

## Step 1: Install the toolchain

**ARM cross-compiler** — compiles C to bare-metal ARM:
- macOS: install via the [cs107e Homebrew formula](https://web.archive.org/web/20210414133806/http://cs107e.github.io/guides/install/mac/)
- Linux: download from [ARM developer](https://developer.arm.com/-/media/Files/downloads/gnu-rm/10.3-2021.10/gcc-arm-none-eabi-10.3-2021.10-x86_64-linux.tar.bz2) and add to `PATH`

**USB-UART driver:**
- macOS: install the [CP210x driver](https://web.archive.org/web/20210414133806/http://cs107e.github.io/guides/install/mac/). Device appears as `/dev/cu.SLAB_USBtoUART`.
- Linux: `sudo usermod -a -G dialout $USER`. Device appears as `/dev/ttyUSB0`.

**QPU assembler** — assembles `.qasm` GPU kernel sources:
- macOS: `brew install vc4asm`
- Linux: [build from source](https://maazl.de/project/vc4asm/doc/index.html#build)

Verify:

```bash
arm-none-eabi-gcc --version
vc4asm --help
ls /dev/cu.SLAB*        # macOS — or ls /dev/ttyUSB* on Linux
```

## Step 2: Build

```bash
git clone https://github.com/masonjwang/pitorch.git && cd pitorch
make -C libpi
make -C tools/bootloader
```

This builds the bare-metal support library and the UART bootloader:
- `tools/bootloader/pi-side/kernel.img` — runs on the Pi, waits for binaries over UART
- `tools/bin/my-install` — runs on your laptop, sends binaries to the Pi

## Step 3: Prepare the SD card

The Pi boots in four stages:

```
Power on
   │
   ▼
bootcode.bin   GPU ROM reads SD, initializes SDRAM
   │
   ▼
start.elf      GPU firmware — reads config.txt, loads initramfs (weights) into RAM,
   │           starts ARM CPU
   ▼
kernel.img     UART bootloader — waits for your .bin over serial, then jumps to it
   │
   ▼
notmain()      your code — bare metal, no OS
```

Format a micro-SD as FAT32 and copy `tools/sdcard/` to the root:

```bash
cp tools/sdcard/* /Volumes/<YOUR_SD>/
```

The `config.txt` included sets `gpu_mem=64`, leaving 448 MB for ARM code and weights.

## Step 4: Load model weights

The `initramfs` line in `config.txt` tells the GPU firmware to read a file from SD into RAM at a fixed address before the ARM CPU starts. Your code reads from that address directly — no file I/O.

Download weights:

```bash
bash tools/download.sh
```

Plug the SD card into your laptop and run:

```bash
bash tools/scripts/setup-sd.sh
```

This concatenates `stories15M.bin` + `tokenizer.bin` into one file, copies it to the SD card, and adds `initramfs weights/stories15M_full.bin 0x2000000` to `config.txt`. The address `0x2000000` is where your code expects the weights:

```
Physical memory (512 MB)
            ┌──────────────────────────┐
0x00000000  │  code + stack            │  ~1 MB
            ├──────────────────────────┤
0x02000000  │  model weights  (58 MB)  │  ← start.elf loads
            │  activations             │    before ARM boots
            │  KV cache                │
            ├──────────────────────────┤
0x1C000000  │  GPU memory   (64 MB)    │  gpu_mem=64 in config.txt
            └──────────────────────────┘
0x20000000
```

Eject and put the SD card back in the Pi.

## Step 5: Run

```bash
cd examples && ./run.sh generate
```

Type a prompt. Expected output:

```
pitorch (288d 6L 32Kv) | 12 QPUs
> Once upon a time
Once upon a time, there was a little girl named Lily...
[48 tokens | 3.9s | 12 tok/s]
```

To switch models, re-run `setup-sd.sh` with the model name:

```bash
bash tools/scripts/setup-sd.sh 42M
```

For 4-Pi distributed setup, see [docs/hardware.md](hardware.md).
