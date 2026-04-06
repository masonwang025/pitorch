# Hardware — 4-Pi Distributed Setup

## Ring topology

```
        ┌──────────┐          ┌──────────┐
        │  Pi 0    │──GPIO──▶ │  Pi 1    │
        │ layers   │          │ layers   │
        │ [0, L/3) │          │[L/3,2L/3)│
        └──────────┘          └──────────┘
             ▲                      │
             │ GPIO                 │ GPIO
             │                      ▼
        ┌──────────┐          ┌──────────┐
        │  Pi 3    │◀──GPIO── │  Pi 2    │
        │ embed +  │          │ layers   │
        │ head     │          │[2L/3, L) │
        └──────────┘          └──────────┘

Forward:  R3 embed → R0 → R1 → R2 → R3 head → argmax
Backward: R3 head  → R2 → R1 → R0 → R3 embed
```

R3 holds the embedding table and classifier head. R0/R1/R2 hold transformer layers. Each Pi loads only its shard from SD — the 110M model (418 MB total) fits across 4 Pis where it wouldn't fit on one.

## GPIO wiring

Each link uses 10 pins (8 data + CLK + ACK), half-duplex. Every Pi has a downstream bank (sends to next rank) and an upstream bank (receives from previous rank):

| Direction | Bank | Data | CLK | ACK |
|---|---|---|---|---|
| Downstream (→ next rank) | High | GPIO 16–23 | 24 | 25 |
| Upstream (← prev rank) | Low | GPIO 4–11 | 12 | 13 |

Wire Pi N's high bank to Pi N+1's low bank:

```
     Pi N  (sender)                     Pi N+1 (receiver)
     HIGH BANK                          LOW BANK
    ┌─────────────┐                    ┌─────────────┐
    │ GPIO 16 ────┼── D0  ──────────── ┤ GPIO 4      │
    │ GPIO 17 ────┼── D1  ──────────── ┤ GPIO 5      │
    │ GPIO 18 ────┼── D2  ──────────── ┤ GPIO 6      │
    │ GPIO 19 ────┼── D3  ──────────── ┤ GPIO 7      │
    │ GPIO 20 ────┼── D4  ──────────── ┤ GPIO 8      │
    │ GPIO 21 ────┼── D5  ──────────── ┤ GPIO 9      │
    │ GPIO 22 ────┼── D6  ──────────── ┤ GPIO 10     │
    │ GPIO 23 ────┼── D7  ──────────── ┤ GPIO 11     │
    │ GPIO 24 ────┼── CLK ──────────── ┤ GPIO 12     │
    │ GPIO 25 ────┼── ACK ──────────── ┤ GPIO 13     │
    │ GND     ────┼── GND ──────────── ┤ GND         │
    └─────────────┘                    └─────────────┘
```

One byte is transferred per handshake cycle: sender raises CLK when data is on the bus, receiver raises ACK when it has read, sender lowers CLK, receiver lowers ACK. The link is self-clocked — no baud rate, no timing constraints.

## USB serial ports

Each Pi connects to the laptop over UART for bootloading and log output. Edit `devices.conf` with your port suffixes:

```
Pi 0 → /dev/cu.usbserial-<suffix>
Pi 1 → /dev/cu.usbserial-<suffix>
Pi 2 → /dev/cu.usbserial-<suffix>
Pi 3 → /dev/cu.usbserial-<suffix>
```

## Weight sharding

Split a full model into 4 per-rank shard files:

```bash
python3 tools/shard_weights.py weights/stories42M.bin  4 weights/shards/42M/
python3 tools/shard_weights.py weights/stories110M.bin 4 weights/shards/110M/
```

Layer assignment for world_size=4:

| Rank | Role | 42M | 110M |
|---|---|---|---|
| R0 | Compute | layers [0, 3) | layers [0, 4) |
| R1 | Compute | layers [3, 6) | layers [4, 8) |
| R2 | Compute | layers [6, 8) | layers [8, 12) |
| R3 | Coord | embed + head | embed + head |

R3 holds both the embedding table and the classifier head because they share the same weight matrix (weight tying). Keeping both on one rank avoids a cross-Pi gradient reduction during training.

## SD card setup

Each Pi gets its own shard file via `initramfs`:

```bash
bash tools/setup-sd-distributed.sh 0 PIE0 42M   # rank 0
bash tools/setup-sd-distributed.sh 1 PIE1 42M   # rank 1
bash tools/setup-sd-distributed.sh 2 PIE2 42M   # rank 2
bash tools/setup-sd-distributed.sh 3 PIE3 42M   # rank 3
```

## Running

```bash
cd examples
./run.sh generate-distributed    # 4-Pi inference
./run.sh train-distributed       # 4-Pi training
```

Logs stream to `examples/logs/pi{0,1,2,3}.log` in real-time. Console shows only the head rank (R3) output during inference.
