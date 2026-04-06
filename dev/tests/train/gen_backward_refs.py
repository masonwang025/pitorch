#!/usr/bin/env python3
"""
Generate reference backward values for all primitive ops.
Writes backward_refs.bin consumed by test_backward_ops.c.

For each op: random inputs → PyTorch autograd → save gradients.
Uses fixed seed for reproducibility.
"""

import struct, math
import torch
import torch.nn.functional as F

# Must match test_backward_ops.c constants
DIM   = 16
HDIM  = 32
HD    = 4   # head_dim
VOCAB = 64


def write_floats(f, t):
    """Write [numel: int32][data: float32[numel]]."""
    data = t.detach().float().contiguous().view(-1)
    f.write(struct.pack('i', len(data)))
    f.write(data.numpy().tobytes())


def write_int(f, v):
    f.write(struct.pack('i', v))


def main():
    torch.manual_seed(42)

    with open('backward_refs.bin', 'wb') as f:

        # ── 1. rmsnorm ───────────────────────────────────────
        x  = torch.randn(DIM, requires_grad=True)
        w  = torch.randn(DIM, requires_grad=True)
        ss = (x * x).mean() + 1e-5
        o  = x * torch.rsqrt(ss) * w
        d_o = torch.randn(DIM)
        o.backward(d_o)

        write_floats(f, x)
        write_floats(f, w)
        write_floats(f, d_o)
        write_floats(f, x.grad)
        write_floats(f, w.grad)
        print(f"rmsnorm:   dim={DIM}  max|d_x|={x.grad.abs().max():.6f}  max|d_w|={w.grad.abs().max():.6f}")

        # ── 2. matmul ────────────────────────────────────────
        W_mat = torch.randn(HDIM, DIM, requires_grad=True)
        x_mat = torch.randn(DIM, requires_grad=True)
        y_mat = W_mat @ x_mat
        d_y   = torch.randn(HDIM)
        y_mat.backward(d_y)

        write_floats(f, W_mat)
        write_floats(f, x_mat)
        write_floats(f, d_y)
        write_floats(f, x_mat.grad)
        write_floats(f, W_mat.grad)
        print(f"matmul:    out={HDIM} in={DIM}  max|d_x|={x_mat.grad.abs().max():.6f}  max|d_W|={W_mat.grad.abs().max():.6f}")

        # ── 3. rope ──────────────────────────────────────────
        pos = 3
        q = torch.randn(DIM, requires_grad=True)
        k = torch.randn(DIM, requires_grad=True)

        cos_p = torch.zeros(DIM // 2)
        sin_p = torch.zeros(DIM // 2)
        for i in range(0, DIM, 2):
            head_i = i % HD
            freq = 1.0 / (10000.0 ** (head_i / HD))
            val  = pos * freq
            cos_p[i // 2] = math.cos(val)
            sin_p[i // 2] = math.sin(val)

        q_even, q_odd = q[0::2], q[1::2]
        q_out = torch.stack([q_even * cos_p - q_odd * sin_p,
                             q_even * sin_p + q_odd * cos_p], dim=-1).view(-1)

        k_even, k_odd = k[0::2], k[1::2]
        k_out = torch.stack([k_even * cos_p - k_odd * sin_p,
                             k_even * sin_p + k_odd * cos_p], dim=-1).view(-1)

        d_q_out = torch.randn(DIM)
        d_k_out = torch.randn(DIM)

        # backward both: q_out depends on q only, k_out depends on k only
        (q_out.unsqueeze(0) @ d_q_out.unsqueeze(1) +
         k_out.unsqueeze(0) @ d_k_out.unsqueeze(1)).sum().backward()

        write_int(f, pos)
        write_floats(f, d_q_out)
        write_floats(f, d_k_out)
        write_floats(f, q.grad)
        write_floats(f, k.grad)
        print(f"rope:      dim={DIM} hd={HD} pos={pos}  max|d_q|={q.grad.abs().max():.6f}  max|d_k|={k.grad.abs().max():.6f}")

        # ── 4. softmax ───────────────────────────────────────
        x_sm = torch.randn(DIM, requires_grad=True)
        y_sm = torch.softmax(x_sm, dim=0)
        d_y_sm = torch.randn(DIM)
        y_sm.backward(d_y_sm)

        write_floats(f, y_sm.detach())
        write_floats(f, d_y_sm)
        write_floats(f, x_sm.grad)
        print(f"softmax:   size={DIM}  max|d_x|={x_sm.grad.abs().max():.6f}")

        # ── 5. silu ──────────────────────────────────────────
        x_silu = torch.randn(DIM, requires_grad=True)
        y_silu = F.silu(x_silu)
        d_y_silu = torch.randn(DIM)
        y_silu.backward(d_y_silu)

        write_floats(f, x_silu)
        write_floats(f, d_y_silu)
        write_floats(f, x_silu.grad)
        print(f"silu:      size={DIM}  max|d_x|={x_silu.grad.abs().max():.6f}")

        # ── 6. embedding ─────────────────────────────────────
        tokens = [3, 7, 3, 12]   # token 3 repeated → tests accumulation
        table  = torch.randn(VOCAB, DIM, requires_grad=True)
        d_os   = [torch.randn(DIM) for _ in tokens]

        loss = sum(torch.dot(table[t], d) for t, d in zip(tokens, d_os))
        loss.backward()

        write_int(f, len(tokens))
        for t in tokens:
            write_int(f, t)
        for d in d_os:
            write_floats(f, d)
        write_floats(f, table.grad)
        print(f"embedding: vocab={VOCAB} dim={DIM} tokens={tokens}  max|d_tab|={table.grad.abs().max():.6f}")

    print(f"\nwrote backward_refs.bin")


if __name__ == '__main__':
    main()
