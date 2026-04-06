#!/usr/bin/env python3
"""
Optional host-side verifier for Pi forward pass test.
Boots the Pi binary, reads output, compares token sequence against Mac reference.

Weights must be pre-loaded on the SD card (config.txt: initramfs weights/<model>.bin 0x2000000).
The standard flow (./run.sh) works without this script — it just runs make which
sends the binary and prints output. This script adds automated verification.

Usage:
    python3 host.py <binary.bin> [--expected expected.txt] [--device /dev/tty...]
"""

import sys, os, struct, glob, binascii, argparse, signal

# ── boot protocol opcodes (from boot-defs.h) ──

GET_PROG_INFO = 0x11112222
PUT_PROG_INFO = 0x33334444
GET_CODE      = 0x55556666
PUT_CODE      = 0x77778888
BOOT_SUCCESS  = 0x9999AAAA
BOOT_ERROR    = 0xBBBBCCCC
PRINT_STRING  = 0xDDDDEEEE
ARMBASE       = 0x8000

OP_NAMES = {
    GET_PROG_INFO: "GET_PROG_INFO", PUT_PROG_INFO: "PUT_PROG_INFO",
    GET_CODE: "GET_CODE",           PUT_CODE: "PUT_CODE",
    BOOT_SUCCESS: "BOOT_SUCCESS",   BOOT_ERROR: "BOOT_ERROR",
    PRINT_STRING: "PRINT_STRING",
}


def find_ttyusb():
    patterns = ["/dev/tty.usbserial-*", "/dev/tty.SLAB*",
                "/dev/ttyUSB*", "/dev/ttyACM*"]
    for pat in patterns:
        devs = sorted(glob.glob(pat))
        if devs:
            return devs[-1]
    return None


class PiUART:
    def __init__(self, device, baud=115200, timeout=5.0):
        import serial
        self.ser = serial.Serial(device, baud, timeout=timeout)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def get8(self):
        b = self.ser.read(1)
        if len(b) == 0:
            raise TimeoutError("UART read timeout (Pi not responding — reboot?)")
        return b[0]

    def put8(self, v):
        self.ser.write(bytes([v & 0xFF]))

    def get32(self):
        d = self.ser.read(4)
        if len(d) < 4:
            raise TimeoutError(f"UART read timeout (got {len(d)}/4 bytes)")
        return struct.unpack('<I', d)[0]

    def put32(self, v):
        self.ser.write(struct.pack('<I', v))

    def write_bytes(self, data):
        self.ser.write(data)

    def flush(self):
        self.ser.flush()

    def read_line(self, timeout=None):
        old_timeout = self.ser.timeout
        if timeout is not None:
            self.ser.timeout = timeout
        try:
            line = self.ser.readline()
            if not line:
                return None
            return line.decode('ascii', errors='replace').rstrip('\r\n')
        finally:
            self.ser.timeout = old_timeout

    def close(self):
        self.ser.close()


def crc32(data):
    return binascii.crc32(data) & 0xFFFFFFFF


def get_op(port):
    while True:
        op = port.get32()
        if op != PRINT_STRING:
            return op
        nbytes = port.get32()
        msg = bytes(port.get8() for _ in range(nbytes)).decode('ascii', errors='replace')
        print(f"  [pi] {msg}", end='' if msg.endswith('\n') else '\n')


def boot(port, code):
    code_crc = crc32(code)
    print(f"boot: {len(code)} bytes, crc32={code_crc:#010x}")
    print("boot: waiting for GET_PROG_INFO...")

    while True:
        op = get_op(port)
        if op == GET_PROG_INFO:
            break

    port.put32(PUT_PROG_INFO)
    port.put32(ARMBASE)
    port.put32(len(code))
    port.put32(code_crc)

    while True:
        op = get_op(port)
        if op != GET_PROG_INFO:
            break

    if op != GET_CODE:
        raise RuntimeError(f"expected GET_CODE, got {OP_NAMES.get(op, hex(op))}")

    echoed_crc = port.get32()
    if echoed_crc != code_crc:
        raise RuntimeError(f"CRC mismatch: sent {code_crc:#010x}, got {echoed_crc:#010x}")

    port.put32(PUT_CODE)
    port.write_bytes(code)
    port.flush()

    op = get_op(port)
    if op != BOOT_SUCCESS:
        raise RuntimeError(f"boot failed: {OP_NAMES.get(op, hex(op))}")
    print("boot: success\n")


def read_pi_output(port, timeout=600):
    lines = []
    port.ser.timeout = timeout
    while True:
        line = port.read_line(timeout=timeout)
        if line is None:
            print("[timeout waiting for Pi output]")
            break
        print(f"[pi] {line}")
        lines.append(line)
        if "=== done ===" in line:
            break
    return lines


def parse_tokens(lines, label):
    tokens = []
    for line in lines:
        if f"{label} step" not in line or "next=" not in line:
            continue
        tok_str = line.split("next=")[1].strip().split()[0]
        tokens.append(int(tok_str))
    return tokens


def load_expected(path):
    tokens = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                tokens.append(int(parts[0]))
    return tokens


def main():
    parser = argparse.ArgumentParser(description="Pi forward pass verifier")
    parser.add_argument("binary", help="Path to Pi .bin file")
    parser.add_argument("--device", help="Serial device (auto-detect if omitted)")
    parser.add_argument("--expected", help="Path to expected.txt")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    with open(args.binary, 'rb') as f:
        code = f.read()

    device = args.device or find_ttyusb()
    if not device:
        print("ERROR: no USB-UART device found", file=sys.stderr)
        sys.exit(1)

    print(f"device: {device}")
    print(f"binary: {args.binary} ({len(code)} bytes)\n")

    port = PiUART(device, baud=args.baud)
    signal.signal(signal.SIGINT, lambda *_: (port.close(), sys.exit(1)))

    try:
        boot(port, code)
        lines = read_pi_output(port, timeout=1200)

        expected_path = args.expected
        if not expected_path:
            candidate = os.path.join(os.path.dirname(__file__), "..", "..", "host", "expected.txt")
            if os.path.exists(candidate):
                expected_path = candidate

        if expected_path and os.path.exists(expected_path):
            expected = load_expected(expected_path)
            for label in ["cpu", "gpu"]:
                pi_tokens = parse_tokens(lines, label)
                if not pi_tokens:
                    continue
                n = min(len(pi_tokens), len(expected))
                ok = all(pi_tokens[i] == expected[i] for i in range(n))
                status = "ALL MATCH" if ok else "MISMATCH"
                print(f"\n{label} vs mac reference: {status}")
                if not ok:
                    for i in range(n):
                        if pi_tokens[i] != expected[i]:
                            print(f"  step {i}: pi={pi_tokens[i]} expected={expected[i]}")
    finally:
        port.close()


if __name__ == '__main__':
    main()
