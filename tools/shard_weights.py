#!/usr/bin/env python3
"""
Split a pitorch model .bin into per-rank shard files for pipeline parallelism.

Shard file format (56-byte header):
  Bytes 0-27:  Standard model header (global config, unchanged)
  Bytes 28-55: Shard extension: magic, rank, world_size, l_start, l_end,
               has_embed, has_head

Followed by weights (only what this rank needs):
  [if has_embed] token_embedding  [vocab_size × dim]
  [if n_local>0] per-layer weights (rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3)
  [if has_head]  rms_final_weight [dim]
  freq_cis_real  [seq_len × head_dim/2]   (always — needed for RoPE)
  freq_cis_imag  [seq_len × head_dim/2]   (always)
  (wcls omitted when shared_weights=1, which is the standard llama2.c convention)

Usage:
  python tools/shard_weights.py weights/stories42M.bin 4 weights/shards/42M/
  → writes rank0.bin, rank1.bin, rank2.bin, rank3.bin

Layer assignment (world_size=4):
  R3: embed + head (0 layers) — keeps shared weights (wcls=embed) on one rank
  R0..R2: transformer layers, split as evenly as possible
"""
import struct
import sys
import os

SHARD_MAGIC = 0x53485244  # "SHRD"


def read_header(data):
    fields = struct.unpack_from("7i", data, 0)
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, raw_vocab, seq_len = fields
    vocab_size = abs(raw_vocab)
    shared = raw_vocab > 0
    head_dim = dim // n_heads
    kv_dim = n_kv_heads * head_dim
    return {
        "dim": dim, "hidden_dim": hidden_dim, "n_layers": n_layers,
        "n_heads": n_heads, "n_kv_heads": n_kv_heads,
        "vocab_size": vocab_size, "raw_vocab": raw_vocab,
        "seq_len": seq_len, "shared": shared,
        "head_dim": head_dim, "kv_dim": kv_dim,
    }


def layer_assignment(n_layers, world_size):
    """Assign layers to compute ranks (all ranks except the last).
    Last rank (world_size-1) gets embed + head, no layers.
    Returns list of (l_start, l_end, has_embed, has_head) per rank."""
    compute_ranks = world_size - 1
    base = n_layers // compute_ranks
    extra = n_layers % compute_ranks

    assignments = []
    offset = 0
    for r in range(compute_ranks):
        count = base + (1 if r < extra else 0)
        assignments.append((offset, offset + count, 0, 0))
        offset += count

    # Last rank: embed + head
    assignments.append((n_layers, n_layers, 1, 1))
    return assignments


def extract_weights(data, h):
    """Parse the flat weight blob into named segments."""
    dim, hidden_dim, n_layers = h["dim"], h["hidden_dim"], h["n_layers"]
    kv_dim, vocab_size, seq_len = h["kv_dim"], h["vocab_size"], h["seq_len"]
    head_dim = h["head_dim"]

    off = 7 * 4  # skip header
    segs = {}

    def take(name, count):
        nonlocal off
        nbytes = count * 4
        segs[name] = data[off:off + nbytes]
        off += nbytes

    take("token_embedding", vocab_size * dim)

    # Per-layer weights (stored layer-major in the file)
    per_layer = [
        ("rms_att_weight", dim),
        ("wq", dim * dim),
        ("wk", kv_dim * dim),
        ("wv", kv_dim * dim),
        ("wo", dim * dim),
        ("rms_ffn_weight", dim),
        ("w1", hidden_dim * dim),
        ("w2", dim * hidden_dim),
        ("w3", hidden_dim * dim),
    ]

    # In llama2.c format, each weight type is stored contiguously for all layers.
    # e.g., all rms_att_weight[0..L-1], then all wq[0..L-1], etc.
    for name, per_layer_count in per_layer:
        total = per_layer_count * n_layers
        take(name, total)

    take("rms_final_weight", dim)
    take("freq_cis_real", seq_len * head_dim // 2)
    take("freq_cis_imag", seq_len * head_dim // 2)

    if not h["shared"]:
        take("wcls", vocab_size * dim)

    return segs, per_layer


def write_shard(path, h, rank, world_size, l_start, l_end, has_embed, has_head,
                segs, per_layer_info):
    """Write a single shard file."""
    n_local = l_end - l_start

    # Standard header (28 bytes) — global config, unchanged
    header = struct.pack("7i",
        h["dim"], h["hidden_dim"], h["n_layers"],
        h["n_heads"], h["n_kv_heads"], h["raw_vocab"], h["seq_len"])

    # Shard extension (28 bytes)
    shard_ext = struct.pack("7i",
        SHARD_MAGIC, rank, world_size, l_start, l_end, has_embed, has_head)

    with open(path, "wb") as f:
        f.write(header)
        f.write(shard_ext)

        # Embed weights
        if has_embed:
            f.write(segs["token_embedding"])

        # Per-layer weights (only for owned layers)
        if n_local > 0:
            for name, per_layer_count in per_layer_info:
                full_blob = segs[name]
                stride = per_layer_count * 4
                chunk = full_blob[l_start * stride : l_end * stride]
                f.write(chunk)

        # Head weights
        if has_head:
            f.write(segs["rms_final_weight"])

        # freq_cis (always needed for RoPE)
        f.write(segs["freq_cis_real"])
        f.write(segs["freq_cis_imag"])

        # wcls (only if has_head and not shared)
        if has_head and not h["shared"]:
            f.write(segs["wcls"])

    return os.path.getsize(path)


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <model.bin> <world_size> <output_dir>")
        sys.exit(1)

    model_path = sys.argv[1]
    world_size = int(sys.argv[2])
    out_dir = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    with open(model_path, "rb") as f:
        data = f.read()

    h = read_header(data)
    print(f"Model: {h['dim']}d {h['n_layers']}L {h['vocab_size']//1000}Kv "
          f"hidden={h['hidden_dim']} seq={h['seq_len']} "
          f"shared={h['shared']}")
    print(f"File size: {len(data) / 1024 / 1024:.1f} MB")

    segs, per_layer_info = extract_weights(data, h)

    assignments = layer_assignment(h["n_layers"], world_size)
    print(f"\nLayer assignment ({world_size} ranks):")

    for rank, (l_start, l_end, has_embed, has_head) in enumerate(assignments):
        n_local = l_end - l_start
        parts = []
        if has_embed: parts.append("embed")
        if n_local > 0: parts.append(f"layers [{l_start},{l_end})")
        if has_head: parts.append("head")
        print(f"  R{rank}: {' + '.join(parts)}")

    print()
    for rank, (l_start, l_end, has_embed, has_head) in enumerate(assignments):
        path = os.path.join(out_dir, f"rank{rank}.bin")
        size = write_shard(path, h, rank, world_size, l_start, l_end,
                           has_embed, has_head, segs, per_layer_info)
        print(f"  rank{rank}.bin: {size / 1024 / 1024:.1f} MB")

    print(f"\nDone. Shards written to {out_dir}/")


if __name__ == "__main__":
    main()
