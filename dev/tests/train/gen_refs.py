#!/usr/bin/env python3
"""
PyTorch reference for pt_forward_train().
Loads stories15M, runs a batched forward pass on a fixed token sequence,
computes per-position logits and cross-entropy loss.
Outputs expected.txt for the C test to validate against.
"""

import struct, sys, math
import torch
import torch.nn.functional as F

T_DEFAULT = 16


def load_checkpoint(path):
    with open(path, 'rb') as f:
        header = struct.unpack('7i', f.read(28))
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, raw_vocab, seq_len = header
    shared_weights = raw_vocab > 0
    vocab_size = abs(raw_vocab)
    head_dim = dim // n_heads
    kv_dim = n_kv_heads * head_dim

    cfg = dict(dim=dim, hidden_dim=hidden_dim, n_layers=n_layers,
               n_heads=n_heads, n_kv_heads=n_kv_heads,
               vocab_size=vocab_size, seq_len=seq_len,
               head_dim=head_dim, kv_dim=kv_dim)

    with open(path, 'rb') as f:
        f.read(28)
        data = torch.frombuffer(bytearray(f.read()), dtype=torch.float32)

    p = [0]
    def take(n):
        t = data[p[0]:p[0]+n].clone()
        p[0] += n
        return t

    w = {}
    w['emb']      = take(vocab_size * dim).view(vocab_size, dim)
    w['rms_att']  = take(n_layers * dim).view(n_layers, dim)
    w['wq']       = take(n_layers * dim * dim).view(n_layers, dim, dim)
    w['wk']       = take(n_layers * kv_dim * dim).view(n_layers, kv_dim, dim)
    w['wv']       = take(n_layers * kv_dim * dim).view(n_layers, kv_dim, dim)
    w['wo']       = take(n_layers * dim * dim).view(n_layers, dim, dim)
    w['rms_ffn']  = take(n_layers * dim).view(n_layers, dim)
    w['w1']       = take(n_layers * hidden_dim * dim).view(n_layers, hidden_dim, dim)
    w['w2']       = take(n_layers * dim * hidden_dim).view(n_layers, dim, hidden_dim)
    w['w3']       = take(n_layers * hidden_dim * dim).view(n_layers, hidden_dim, dim)
    w['rms_final'] = take(dim)
    take(seq_len * head_dim // 2)   # freq_cis_real
    take(seq_len * head_dim // 2)   # freq_cis_imag
    w['wcls'] = w['emb'] if shared_weights else take(vocab_size * dim).view(vocab_size, dim)

    return cfg, w


def rmsnorm(x, weight):
    ss = (x * x).mean(-1, keepdim=True) + 1e-5
    return x * torch.rsqrt(ss) * weight


def rope_single(q, k, dim, head_dim, kv_dim, pos):
    for i in range(0, dim, 2):
        head_i = i % head_dim
        freq = 1.0 / (10000.0 ** (head_i / head_dim))
        val = pos * freq
        cos_val = math.cos(val)
        sin_val = math.sin(val)

        q0, q1 = q[i].item(), q[i+1].item()
        q[i]   = q0 * cos_val - q1 * sin_val
        q[i+1] = q0 * sin_val + q1 * cos_val

        if i < kv_dim:
            k0, k1 = k[i].item(), k[i+1].item()
            k[i]   = k0 * cos_val - k1 * sin_val
            k[i+1] = k0 * sin_val + k1 * cos_val


def forward_batched(cfg, w, tokens):
    """Run the full transformer on T tokens at once, returning logits[T, V]."""
    dim        = cfg['dim']
    hidden_dim = cfg['hidden_dim']
    n_heads    = cfg['n_heads']
    head_dim   = cfg['head_dim']
    kv_dim     = cfg['kv_dim']
    kv_mul     = n_heads // cfg['n_kv_heads']
    T          = len(tokens)

    x = w['emb'][tokens].clone()  # [T, dim]

    for l in range(cfg['n_layers']):
        xb = rmsnorm(x, w['rms_att'][l])

        Q = torch.zeros(T, dim)
        K = torch.zeros(T, kv_dim)
        V = torch.zeros(T, kv_dim)

        for t in range(T):
            Q[t] = w['wq'][l] @ xb[t]
            K[t] = w['wk'][l] @ xb[t]
            V[t] = w['wv'][l] @ xb[t]
            rope_single(Q[t], K[t], dim, head_dim, kv_dim, t)

        att_out = torch.zeros(T, dim)
        for t in range(T):
            for h in range(n_heads):
                q_h = Q[t, h*head_dim:(h+1)*head_dim]
                scores = torch.zeros(t + 1)
                for t2 in range(t + 1):
                    k_h = K[t2, (h//kv_mul)*head_dim:(h//kv_mul+1)*head_dim]
                    scores[t2] = (q_h * k_h).sum() / math.sqrt(head_dim)
                a = torch.softmax(scores, dim=0)
                out_h = torch.zeros(head_dim)
                for t2 in range(t + 1):
                    v_h = V[t2, (h//kv_mul)*head_dim:(h//kv_mul+1)*head_dim]
                    out_h += a[t2] * v_h
                att_out[t, h*head_dim:(h+1)*head_dim] = out_h

        for t in range(T):
            x[t] = x[t] + w['wo'][l] @ att_out[t]

        x_mid = x.clone()  # post-attention residual

        xb = rmsnorm(x_mid, w['rms_ffn'][l])
        for t in range(T):
            hb  = w['w1'][l] @ xb[t]
            hb2 = w['w3'][l] @ xb[t]
            hb  = hb * torch.sigmoid(hb)  # SiLU
            hb  = hb * hb2
            x[t] = x_mid[t] + w['w2'][l] @ hb

    x = rmsnorm(x, w['rms_final'])
    logits = torch.zeros(T, cfg['vocab_size'])
    for t in range(T):
        logits[t] = w['wcls'] @ x[t]

    return logits


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <model.bin> [expected.txt] [T]", file=sys.stderr)
        sys.exit(1)

    model_path = sys.argv[1]
    out_path   = sys.argv[2] if len(sys.argv) > 2 else "expected.txt"
    T          = int(sys.argv[3]) if len(sys.argv) > 3 else T_DEFAULT

    cfg, w = load_checkpoint(model_path)
    print(f"config: dim={cfg['dim']} hidden={cfg['hidden_dim']} "
          f"layers={cfg['n_layers']} heads={cfg['n_heads']} "
          f"kv_heads={cfg['n_kv_heads']} vocab={cfg['vocab_size']} seq={cfg['seq_len']}")

    # Build token sequence via greedy decoding
    print(f"\nbuilding {T}-token greedy sequence from BOS...")

    seq_len = cfg['seq_len']
    kv_dim  = cfg['kv_dim']
    n_layers = cfg['n_layers']
    key_cache = torch.zeros(n_layers, seq_len, kv_dim)
    val_cache = torch.zeros(n_layers, seq_len, kv_dim)

    tokens = [1]  # BOS
    token = 1
    for step in range(T - 1):
        logits = forward_single(cfg, w, token, step, key_cache, val_cache)
        next_tok = logits.argmax().item()
        tokens.append(next_tok)
        token = next_tok
        print(f"  step {step}: token {tokens[step]} -> {next_tok}")

    print(f"\ntokens: {tokens}")

    # Run batched forward
    print(f"\nrunning batched forward on {T} tokens...")
    logits = forward_batched(cfg, w, tokens)

    # Cross-entropy loss
    # targets are tokens[1..T-1], predictions at positions 0..T-2
    targets = torch.tensor(tokens[1:], dtype=torch.long)
    pred_logits = logits[:-1]  # [T-1, V]
    loss = F.cross_entropy(pred_logits, targets).item()
    print(f"loss: {loss:.6f}")

    # Per-position top-5
    with open(out_path, 'w') as f:
        f.write(f"{T}\n")
        f.write(" ".join(str(t) for t in tokens) + "\n")
        f.write(f"{loss:.8f}\n")
        for t in range(T):
            vals, idxs = torch.topk(logits[t], 5)
            t5 = idxs.tolist()
            v5 = vals.tolist()
            f.write(f"{t5[0]} {t5[1]} {t5[2]} {t5[3]} {t5[4]}\n")
            print(f"  pos {t}: top5 [{t5[0]}, {t5[1]}, {t5[2]}, {t5[3]}, {t5[4]}] "
                  f"vals [{v5[0]:.4f}, {v5[1]:.4f}, {v5[2]:.4f}, {v5[3]:.4f}, {v5[4]:.4f}]")

    print(f"\nwrote {out_path}")


def forward_single(cfg, w, token, pos, key_cache, val_cache):
    """Single-token forward (same as reference.py) for building the greedy sequence."""
    dim        = cfg['dim']
    hidden_dim = cfg['hidden_dim']
    n_heads    = cfg['n_heads']
    head_dim   = cfg['head_dim']
    kv_dim     = cfg['kv_dim']
    kv_mul     = n_heads // cfg['n_kv_heads']

    x = w['emb'][token].clone()

    for l in range(cfg['n_layers']):
        xb = rmsnorm(x, w['rms_att'][l])
        q = w['wq'][l] @ xb
        k = w['wk'][l] @ xb
        v = w['wv'][l] @ xb
        rope_single(q, k, dim, head_dim, kv_dim, pos)

        key_cache[l][pos] = k.clone()
        val_cache[l][pos] = v.clone()

        xb_out = torch.zeros(dim)
        for h in range(n_heads):
            q_h = q[h*head_dim:(h+1)*head_dim]
            att = torch.zeros(pos + 1)
            for t in range(pos + 1):
                k_h = key_cache[l][t][(h//kv_mul)*head_dim:(h//kv_mul+1)*head_dim]
                att[t] = (q_h * k_h).sum() / math.sqrt(head_dim)
            att = torch.softmax(att, dim=0)

            xb_h = torch.zeros(head_dim)
            for t in range(pos + 1):
                v_h = val_cache[l][t][(h//kv_mul)*head_dim:(h//kv_mul+1)*head_dim]
                xb_h += att[t] * v_h
            xb_out[h*head_dim:(h+1)*head_dim] = xb_h

        x = x + w['wo'][l] @ xb_out

        xb = rmsnorm(x, w['rms_ffn'][l])
        hb  = w['w1'][l] @ xb
        hb2 = w['w3'][l] @ xb
        hb = hb * torch.sigmoid(hb)
        hb = hb * hb2
        x = x + w['w2'][l] @ hb

    x = rmsnorm(x, w['rms_final'])
    return w['wcls'] @ x


if __name__ == '__main__':
    main()
