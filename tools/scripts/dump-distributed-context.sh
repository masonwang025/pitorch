#!/bin/bash
# Dumps repo context + distributed planning prompt to clipboard.
# Usage: bash tools/scripts/dump-distributed-context.sh | pbcopy
# Then paste into ChatGPT / Claude on the Web.

cd "$(dirname "$0")/../.." || exit 1

{
cat <<'PROMPT_START'
# Context: PiTorch — Bare-Metal ML Framework for Raspberry Pi Zero

I'm building PiTorch, a bare-metal machine learning framework that runs LLM inference and training on the Raspberry Pi Zero's GPU (VideoCore IV, 12 QPUs). No OS, no Linux — just ARM code on the metal with UART serial I/O. It currently runs Llama 2 models up to 110M parameters (inference) and 42M parameters (training) on a single Pi Zero.

Below is the full codebase context. After the context, I'll describe what I need from you.

---

PROMPT_START

echo "## Repo Tree"
echo '```'
find . -type f \( -name "*.c" -o -name "*.h" -o -name "*.S" -o -name "*.qasm" -o -name "Makefile*" -o -name "*.mk" -o -name "*.sh" -o -name "*.md" -o -name "*.txt" -o -name "*.py" \) \
  | grep -v '.git/' | grep -v 'weights/' | grep -v '.dSYM' | grep -v 'staff-' | sort
echo '```'
echo ""

echo "## README.md"
echo '```'
cat README.md
echo '```'
echo ""

# Key architecture files
for f in \
  pitorch/pt.h \
  pitorch/pt.c \
  pitorch/model/llama2.h \
  pitorch/model/llama2.c \
  pitorch/train/pt_train.h \
  pitorch/ops/core/pt_ops.h \
  pitorch/ops/matvec/matvec.h \
  pitorch/runtime/gpu.h \
  pitorch/runtime/gpu.c \
  pitorch/runtime/mailbox.h \
  pitorch/runtime/mailbox.c \
  pitorch/runtime/arena.h \
  pitorch/runtime/arena.c \
  pitorch/runtime/mmu.h \
  pitorch/runtime/mmu.c \
  pitorch/text/pt_text.h \
  pitorch/profiler/trace.h \
; do
  echo "## $f"
  echo '```c'
  cat "$f"
  echo '```'
  echo ""
done

# Build system
for f in \
  pitorch.mk \
  Makefile \
  demo/tests/forward/Makefile \
  demo/tests/generate/Makefile \
  demo/tests/train/Makefile \
; do
  echo "## $f"
  echo '```make'
  cat "$f"
  echo '```'
  echo ""
done

# Test files (the actual programs people run)
for f in \
  demo/tests/forward/test_forward.c \
  demo/tests/generate/test_generate.c \
  demo/tests/train/test_train.c \
; do
  echo "## $f"
  echo '```c'
  cat "$f"
  echo '```'
  echo ""
done

# Bootloader and hardware
for f in \
  tools/bootloader/pi-side/boot-start.S \
  tools/bootloader/pi-side/main.c \
  tools/bootloader/unix-side/my-install.c \
  tools/sdcard/config.txt \
  tools/scripts/setup-sd.sh \
; do
  echo "## $f"
  echo '```'
  cat "$f"
  echo '```'
  echo ""
done

# Available hardware abstractions (headers only)
for f in \
  libpi/include/spi.h \
  libpi/include/sw-uart.h \
  libpi/include/gpio.h \
  libpi/include/rpi-constants.h \
  libpi/include/timer-interrupt.h \
; do
  echo "## $f"
  echo '```c'
  cat "$f"
  echo '```'
  echo ""
done

# Technical writeups (the hard-won knowledge)
for f in \
  docs/writeups/baremetal-is-cursed.md \
  docs/writeups/scaling-to-42M.md \
; do
  echo "## $f"
  cat "$f"
  echo ""
done

# QPU kernel (the GPU shader)
echo "## pitorch/kernels/matvec/matvec_tmu.qasm"
echo '```asm'
cat pitorch/kernels/matvec/matvec_tmu.qasm
echo '```'
echo ""

# Training backward pass (understanding what needs to be distributed)
echo "## pitorch/train/pt_train.c"
echo '```c'
cat pitorch/train/pt_train.c
echo '```'
echo ""

echo "## pitorch/train/pt_backward_ops.h"
echo '```c'
cat pitorch/train/pt_backward_ops.h
echo '```'
echo ""

cat <<'PROMPT_END'

---

# What I Need: A Distributed Computing Plan for PiTorch

## Current State (what works today, on a single Pi Zero)

- **Inference**: stories15M at 11.7 tok/s (with D-cache), stories42M at 0.7 tok/s, stories110M works but slow
- **Training**: reverse-mode autodiff with GPU-accelerated backward pass, overfits 15M in 18 steps
- **GPU**: all 12 QPUs dispatched via V3D registers, arena-based GPU memory, pipelined TMU matvec
- **Memory**: 448 MB ARM RAM (gpu_mem=64), weights loaded from SD card via GPU firmware initramfs at 0x2000000
- **Communication**: UART serial to host Mac (115200–460800 baud), mailbox to GPU firmware, V3D registers to QPUs
- **Build**: cross-compiled ARM bare-metal binaries, sent over UART bootloader, no OS/libc/malloc in hot path

## Hardware I Have

- **3 identical Raspberry Pi Zero boards** (ARM1176JZF-S @ 700 MHz, 512 MB RAM each, VideoCore IV with 12 QPUs each)
- **3 SD cards** (each can boot independently with its own config.txt + kernel.img + weights)
- **Jumper cables** (for connecting GPIO pins between Pis)
- **USB-to-UART adapter** (currently used for host communication with one Pi)
- The Pi Zero has: 28 GPIO pins, SPI0 (with 2 chip selects), I2C (BSC0/BSC1), 2x UART (PL011 + mini-UART), PWM

## Available Communication Interfaces (from libpi)

- **SPI**: `spi.h` — SPI0 with CE0/CE1, master mode, configurable clock divider. Header exists, not yet used in pitorch.
- **Software UART**: `sw-uart.h` — bit-banged UART on any GPIO pin pair, configurable baud rate. Can create multiple instances on different pins.
- **GPIO**: `gpio.h` — full pin control, interrupt support (rising/falling edge), pullup/pulldown configuration.
- **Hardware UART (PL011)**: currently used for host communication. Could potentially be repurposed or shared.
- **Timer interrupts**: `timer-interrupt.h` — available for scheduling/synchronization.

## What I Want

I want to extend PiTorch to run distributed inference and training across multiple Pi Zero boards, connected via jumper cables (GPIO/SPI/UART — you tell me which is best).

**Starting point**: 2 Pis working together. **End goal**: generalize to N Pis.

I need you to design a concrete, phased plan with milestones — similar to how the project has been built so far (phase 0: foundation, phase 1: first token, phase 2: GPU acceleration, phase 3: training, etc.). Each phase should have:

1. **Clear success criteria** — what specifically works at the end of this phase
2. **What gets built** — the new code/modules, where they live in the repo
3. **What gets tested** — specific test programs with expected output
4. **Estimated complexity** — relative to what's already been done

## Design Constraints (important)

- **Clean, modular code.** The repo is currently well-layered (runtime → ops → model → text). Any distributed code should follow this pattern. I do NOT want complexity to balloon. Each new module should be self-contained and play nicely with the existing layers. The repo should remain readable and the moving parts should be obvious.
- **Bare-metal constraints.** No OS, no TCP/IP, no threads. Everything is polling-based or interrupt-driven. Memory is manually managed. There's no dynamic linking.
- **D-cache coherency.** Any shared-memory or DMA scheme must account for ARM L1 D-cache (16 KB, write-back). The GPU and external bus masters don't see cached data without explicit flushes.
- **Existing API preservation.** The `pt_context_t` / `pt_forward_step` / `pt_train_step` API should ideally still work, with distribution being transparent or opt-in.
- **Build on what exists.** The SPI, GPIO, sw-uart, and interrupt infrastructure in libpi are available. Use them.
- **Start simple.** The first milestone should be dead simple — like one Pi sending a float array to another Pi over a wire and getting it back. Build confidence in the communication layer before doing anything ML-related.

## What I Want From You

**Do not just give me a high-level overview.** I want a detailed, phased plan that I can execute milestone by milestone. Think deeply about:

1. **Which communication protocol** (SPI vs UART vs GPIO bit-bang vs something else) and why. Consider bandwidth, latency, wiring complexity, and the fact that we need to transfer weight matrices (potentially megabytes). Look at what real distributed bare-metal systems do.

2. **Which parallelism strategy** for inference (tensor parallelism? pipeline parallelism? something else?) and for training (data parallelism? gradient averaging? pipeline?). Consider the Pi Zero's constraints: 448 MB RAM each, 12 QPUs each, ~11.7 tok/s single-device inference. What makes sense at this scale?

3. **The communication layer design** — message format, synchronization, error handling. This is bare-metal, so think about what happens when a wire is noisy, when timing drifts, when one Pi is slower than another.

4. **Memory layout on each Pi** — how do weights get split? Does each Pi have a full copy on its SD card, or partial? How does the initramfs mechanism interact with this?

5. **Concrete milestones** with specific test programs I can write and run. Not "implement distributed training" — more like "Pi A sends a 288-element float vector over SPI to Pi B, Pi B computes RMSNorm, sends the result back, Pi A verifies against local computation."

6. **How this generalizes from 2 to N.** The 2-Pi design should not paint us into a corner. Think about what changes when we add a third Pi.

Please research real-world approaches: how PyTorch distributes models (tensor parallelism in Megatron-LM, pipeline parallelism in GPipe/PipeDream), how embedded/HPC systems handle inter-node communication without TCP/IP, what SPI/UART bandwidth limits are on the BCM2835, and any relevant bare-metal distributed computing projects. I want you to bring external knowledge to bear, not just reason from the codebase alone.

Take your time. Think deeply. The plan should be something I can follow for weeks of work, with each phase building cleanly on the last.
PROMPT_END

} 2>/dev/null
