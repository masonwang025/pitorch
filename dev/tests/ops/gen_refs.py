#!/usr/bin/env python3
"""Generate hardcoded reference values for Phase 0 tests.
Run on Mac, paste output into test_phase0.c."""
import numpy as np
np.set_printoptions(precision=8)

def fmt(arr, name=""):
    """Format float array as C initializer."""
    vals = ", ".join(f"{v:.8f}f" for v in arr.flat)
    return f"{{ {vals} }}"

# ── expf ──
exp_inputs = np.array([-10, -5, -2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 5, 10], dtype=np.float32)
exp_outputs = np.exp(exp_inputs)
print("// expf")
print(f"float exp_in[]  = {fmt(exp_inputs)};")
print(f"float exp_ref[] = {fmt(exp_outputs)};")
print(f"int exp_n = {len(exp_inputs)};")
print()

# ── sinf ──
sin_inputs = np.array([0, 0.1, 0.5, 1.0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, -0.5, -1.0, -np.pi/2, -np.pi], dtype=np.float32)
sin_outputs = np.sin(sin_inputs)
print("// sinf")
print(f"float sin_in[]  = {fmt(sin_inputs)};")
print(f"float sin_ref[] = {fmt(sin_outputs)};")
print(f"int sin_n = {len(sin_inputs)};")
print()

# ── cosf ──
cos_inputs = sin_inputs.copy()
cos_outputs = np.cos(cos_inputs)
print("// cosf")
print(f"float cos_in[]  = {fmt(cos_inputs)};")
print(f"float cos_ref[] = {fmt(cos_outputs)};")
print(f"int cos_n = {len(cos_inputs)};")
print()

# ── rmsnorm ──
x_rms = np.array([1.0, 2.0, -1.0, 0.5, -0.5, 3.0], dtype=np.float32)
w_rms = np.array([0.5, 1.0, 0.5, 1.0, 0.5, 1.0], dtype=np.float32)
ss = np.mean(x_rms**2)
ss = 1.0 / np.sqrt(ss + 1e-5)
o_rms = x_rms * ss * w_rms
print("// rmsnorm (dim=6)")
print(f"float rms_x[] = {fmt(x_rms)};")
print(f"float rms_w[] = {fmt(w_rms)};")
print(f"float rms_ref[] = {fmt(o_rms)};")
print()

# ── softmax ──
x_sm = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0], dtype=np.float32)
e = np.exp(x_sm - np.max(x_sm))
o_sm = e / np.sum(e)
print("// softmax (size=6)")
print(f"float sm_x[] = {fmt(x_sm)};")
print(f"float sm_ref[] = {fmt(o_sm)};")
print()

# ── silu ──
x_silu = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
o_silu = x_silu * (1.0 / (1.0 + np.exp(-x_silu)))
print("// silu (size=6)")
print(f"float silu_x[] = {fmt(x_silu)};")
print(f"float silu_ref[] = {fmt(o_silu)};")
print()

# ── rope ──
# dim=12 (2 heads of head_dim=6), pos=3
dim_rope = 12
head_dim_rope = 6
pos_rope = 3
q_rope = np.array([1,2,3,4,5,6,7,8,9,10,11,12], dtype=np.float32)
k_rope = np.array([12,11,10,9,8,7,6,5,4,3,2,1], dtype=np.float32)
q_out = q_rope.copy()
k_out = k_rope.copy()
for i in range(0, dim_rope, 2):
    head_i = i % head_dim_rope
    freq = 1.0 / (10000.0 ** (float(head_i) / float(head_dim_rope)))
    val = pos_rope * freq
    cos_val = np.cos(val).astype(np.float32)
    sin_val = np.sin(val).astype(np.float32)
    q0, q1 = q_out[i], q_out[i+1]
    q_out[i]   = q0 * cos_val - q1 * sin_val
    q_out[i+1] = q0 * sin_val + q1 * cos_val
    k0, k1 = k_out[i], k_out[i+1]
    k_out[i]   = k0 * cos_val - k1 * sin_val
    k_out[i+1] = k0 * sin_val + k1 * cos_val
print(f"// rope (dim={dim_rope}, head_dim={head_dim_rope}, pos={pos_rope})")
print(f"float rope_q[] = {fmt(q_rope)};")
print(f"float rope_k[] = {fmt(k_rope)};")
print(f"float rope_q_ref[] = {fmt(q_out)};")
print(f"float rope_k_ref[] = {fmt(k_out)};")
print()

# ── vec_add ──
a_va = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
b_va = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
o_va = a_va + b_va
print("// vec_add (size=6)")
print(f"float va_a[] = {fmt(a_va)};")
print(f"float va_b[] = {fmt(b_va)};")
print(f"float va_ref[] = {fmt(o_va)};")
print()

# ── vec_mul ──
a_vm = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
b_vm = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
o_vm = a_vm * b_vm
print("// vec_mul (size=6)")
print(f"float vm_a[] = {fmt(a_vm)};")
print(f"float vm_b[] = {fmt(b_vm)};")
print(f"float vm_ref[] = {fmt(o_vm)};")
print()

# ── embedding_lookup ──
table = np.array([
    [1.0, 0.5, -0.5, 0.1, -0.1, 0.3],
    [0.2, -0.3, 0.8, -0.6, 0.4, 0.1],
    [0.9, 0.7, -0.2, 0.5, -0.8, 0.6],
    [-0.4, 0.3, 0.1, -0.9, 0.2, -0.5],
], dtype=np.float32)
tok = 2
print(f"// embedding_lookup (vocab=4, dim=6, token={tok})")
print(f"float emb_table[] = {fmt(table)};")
print(f"float emb_ref[] = {fmt(table[tok])};")
print()

# ── argmax ──
x_am = np.array([0.1, 0.5, 0.3, 0.9, 0.2, 0.4], dtype=np.float32)
print(f"// argmax (size=6)")
print(f"float am_x[] = {fmt(x_am)};")
print(f"int am_ref = {np.argmax(x_am)};")
print()

# ── smatvec_cpu ──
W = np.array([
    [1.0, 0.5, -0.3, 0.2, 0.1, -0.4],
    [-0.2, 0.8, 0.1, -0.5, 0.3, 0.6],
    [0.4, -0.1, 0.7, 0.3, -0.6, 0.2],
    [0.1, 0.3, -0.2, 0.9, 0.4, -0.1],
], dtype=np.float32)
x_mv = np.array([1.0, 2.0, 3.0, -1.0, 0.5, -0.5], dtype=np.float32)
y_mv = W @ x_mv
print(f"// smatvec_cpu (out_dim=4, in_dim=6)")
print(f"float mv_W[] = {fmt(W)};")
print(f"float mv_x[] = {fmt(x_mv)};")
print(f"float mv_ref[] = {fmt(y_mv)};")
