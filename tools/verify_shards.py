#!/usr/bin/env python3
"""
Verify that shard files are byte-for-byte identical to the corresponding
slices of the original model file.

Usage:
  python tools/verify_shards.py [model_path] [shard_dir]

Defaults:
  model_path = weights/stories42M.bin
  shard_dir  = weights/shards/42M/
"""
import struct
import sys
import os
import numpy as np

SHARD_MAGIC = 0x53485244  # "SHRD"
HEADER_BYTES = 28         # 7 × int32
SHARD_HEADER_BYTES = 56   # 14 × int32 (standard + extension)


def read_model_header(data):
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


def read_shard_header(data):
    std = struct.unpack_from("7i", data, 0)
    ext = struct.unpack_from("7i", data, 28)
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, raw_vocab, seq_len = std
    magic, rank, world_size, l_start, l_end, has_embed, has_head = ext
    assert magic == SHARD_MAGIC, f"Bad shard magic: 0x{magic:08X}"
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
        "rank": rank, "world_size": world_size,
        "l_start": l_start, "l_end": l_end,
        "has_embed": has_embed, "has_head": has_head,
    }


def build_original_weight_map(h):
    """Return an ordered list of (name, byte_offset, byte_length) for the original model."""
    dim = h["dim"]
    hidden_dim = h["hidden_dim"]
    n_layers = h["n_layers"]
    kv_dim = h["kv_dim"]
    vocab_size = h["vocab_size"]
    seq_len = h["seq_len"]
    head_dim = h["head_dim"]

    weights = []
    off = HEADER_BYTES

    def add(name, n_floats):
        nonlocal off
        nbytes = n_floats * 4
        weights.append((name, off, nbytes))
        off += nbytes

    add("token_embedding", vocab_size * dim)

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
    for name, per_layer_count in per_layer:
        add(name, per_layer_count * n_layers)

    add("rms_final_weight", dim)
    add("freq_cis_real", seq_len * head_dim // 2)
    add("freq_cis_imag", seq_len * head_dim // 2)

    if not h["shared"]:
        add("wcls", vocab_size * dim)

    return weights, per_layer


def verify_shard(shard_path, orig_data, h_orig, per_layer_info):
    """Verify one shard file against the original model. Returns (n_pass, n_fail, results)."""
    with open(shard_path, "rb") as f:
        shard_data = f.read()

    sh = read_shard_header(shard_data)
    rank = sh["rank"]
    l_start, l_end = sh["l_start"], sh["l_end"]
    n_local = l_end - l_start
    has_embed = sh["has_embed"]
    has_head = sh["has_head"]

    dim = h_orig["dim"]
    hidden_dim = h_orig["hidden_dim"]
    n_layers = h_orig["n_layers"]
    kv_dim = h_orig["kv_dim"]
    vocab_size = h_orig["vocab_size"]
    seq_len = h_orig["seq_len"]
    head_dim = h_orig["head_dim"]

    # Build offset map for the original file
    orig_off = HEADER_BYTES
    orig_offsets = {}

    def orig_add(name, n_floats):
        nonlocal orig_off
        nbytes = n_floats * 4
        orig_offsets[name] = (orig_off, nbytes)
        orig_off += nbytes

    orig_add("token_embedding", vocab_size * dim)
    for name, per_layer_count in per_layer_info:
        orig_add(name, per_layer_count * n_layers)
    orig_add("rms_final_weight", dim)
    orig_add("freq_cis_real", seq_len * head_dim // 2)
    orig_add("freq_cis_imag", seq_len * head_dim // 2)
    if not h_orig["shared"]:
        orig_add("wcls", vocab_size * dim)

    # Walk the shard data and compare each segment
    shard_off = SHARD_HEADER_BYTES
    results = []
    n_pass = 0
    n_fail = 0

    def compare(name, shard_start, nbytes, orig_start):
        nonlocal n_pass, n_fail
        shard_slice = np.frombuffer(shard_data, dtype=np.uint8, count=nbytes, offset=shard_start)
        orig_slice = np.frombuffer(orig_data, dtype=np.uint8, count=nbytes, offset=orig_start)
        match = np.array_equal(shard_slice, orig_slice)
        status = "PASS" if match else "FAIL"
        if match:
            n_pass += 1
        else:
            n_fail += 1
            # Find first mismatch for diagnostics
            diff_indices = np.where(shard_slice != orig_slice)[0]
            first_diff = diff_indices[0]
            results.append((name, nbytes, orig_start, orig_start + nbytes, status,
                            f"first diff at byte +{first_diff}"))
            return
        results.append((name, nbytes, orig_start, orig_start + nbytes, status, ""))

    # token_embedding
    if has_embed:
        o_off, o_len = orig_offsets["token_embedding"]
        compare("token_embedding", shard_off, o_len, o_off)
        shard_off += o_len

    # per-layer weights
    if n_local > 0:
        for name, per_layer_count in per_layer_info:
            stride = per_layer_count * 4
            chunk_bytes = stride * n_local
            full_off, _ = orig_offsets[name]
            slice_orig_off = full_off + l_start * stride
            label = f"{name}[{l_start}:{l_end}]"
            compare(label, shard_off, chunk_bytes, slice_orig_off)
            shard_off += chunk_bytes

    # rms_final_weight
    if has_head:
        o_off, o_len = orig_offsets["rms_final_weight"]
        compare("rms_final_weight", shard_off, o_len, o_off)
        shard_off += o_len

    # freq_cis_real (always present)
    o_off, o_len = orig_offsets["freq_cis_real"]
    compare("freq_cis_real", shard_off, o_len, o_off)
    shard_off += o_len

    # freq_cis_imag (always present)
    o_off, o_len = orig_offsets["freq_cis_imag"]
    compare("freq_cis_imag", shard_off, o_len, o_off)
    shard_off += o_len

    # wcls (if has_head and not shared)
    if has_head and not h_orig["shared"]:
        o_off, o_len = orig_offsets["wcls"]
        compare("wcls", shard_off, o_len, o_off)
        shard_off += o_len

    # Verify we consumed the entire shard
    if shard_off != len(shard_data):
        results.append(("SHARD_SIZE_CHECK", 0, 0, 0, "FAIL",
                        f"consumed {shard_off} bytes but shard is {len(shard_data)} bytes"))
        n_fail += 1
    else:
        results.append(("SHARD_SIZE_CHECK", 0, 0, 0, "PASS",
                        f"all {shard_off} bytes accounted for"))
        n_pass += 1

    return rank, l_start, l_end, has_embed, has_head, n_pass, n_fail, results


def fmt_bytes(n):
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "weights/stories42M.bin"
    shard_dir = sys.argv[2] if len(sys.argv) > 2 else "weights/shards/42M/"

    sep = "=" * 72
    thin = "-" * 72

    print(sep)
    print("  SHARD VERIFICATION")
    print(sep)
    print(f"  Original model : {model_path}")
    print(f"  Shard directory: {shard_dir}")
    print()

    # Read original model
    with open(model_path, "rb") as f:
        orig_data = f.read()
    h = read_model_header(orig_data)

    print(f"  Model config:")
    print(f"    dim={h['dim']}  hidden_dim={h['hidden_dim']}  n_layers={h['n_layers']}")
    print(f"    n_heads={h['n_heads']}  n_kv_heads={h['n_kv_heads']}  vocab_size={h['vocab_size']}")
    print(f"    seq_len={h['seq_len']}  head_dim={h['head_dim']}  kv_dim={h['kv_dim']}")
    print(f"    shared_weights={'yes' if h['shared'] else 'no'}")
    print(f"    file size: {fmt_bytes(len(orig_data))}")
    print()

    _, per_layer_info = build_original_weight_map(h)

    # Discover shard files
    shard_files = sorted([
        os.path.join(shard_dir, f) for f in os.listdir(shard_dir)
        if f.startswith("rank") and f.endswith(".bin")
    ])

    if not shard_files:
        print("  ERROR: No shard files found.")
        sys.exit(1)

    print(f"  Found {len(shard_files)} shard file(s)")
    print()

    total_pass = 0
    total_fail = 0

    for shard_path in shard_files:
        rank, l_start, l_end, has_embed, has_head, n_pass, n_fail, results = \
            verify_shard(shard_path, orig_data, h, per_layer_info)

        n_local = l_end - l_start
        parts = []
        if has_embed:
            parts.append("embed")
        if n_local > 0:
            parts.append(f"layers [{l_start},{l_end})")
        if has_head:
            parts.append("head")
        parts_str = " + ".join(parts) if parts else "(freq_cis only)"

        shard_size = os.path.getsize(shard_path)

        print(sep)
        print(f"  RANK {rank}  |  {os.path.basename(shard_path)}  |  {fmt_bytes(shard_size)}")
        print(f"  Contains: {parts_str}")
        print(thin)
        print(f"  {'Weight':<30s}  {'Size':>10s}  {'Orig range':>26s}  {'Result':>6s}")
        print(thin)

        for name, nbytes, orig_start, orig_end, status, detail in results:
            if name == "SHARD_SIZE_CHECK":
                print(thin)
                print(f"  {'[size check]':<30s}  {'':>10s}  {'':>26s}  {status:>6s}")
                if detail:
                    print(f"    {detail}")
            else:
                range_str = f"0x{orig_start:08X}..0x{orig_end:08X}"
                print(f"  {name:<30s}  {fmt_bytes(nbytes):>10s}  {range_str:>26s}  {status:>6s}")
                if detail:
                    print(f"    {detail}")

        total_pass += n_pass
        total_fail += n_fail
        print()

    # Final summary
    print(sep)
    if total_fail == 0:
        print(f"  ALL CHECKS PASSED  ({total_pass}/{total_pass + total_fail})")
    else:
        print(f"  VERIFICATION FAILED  ({total_fail} failed, {total_pass} passed"
              f" out of {total_pass + total_fail})")
    print(sep)

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
