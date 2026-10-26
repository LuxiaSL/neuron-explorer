"""Direct-path connections for every neuron at every effective position of a kotodama checkpoint (the
build_connections.py computation with kotodama's layout): for positions i < j, P[B, A] = (W_gate_j[B] . v_A) *
(W_up_j[B] . v_A) with v_A = w_j * W_down_i[:, A], where w_j is position j's own ffn_norm weight (plain RMSNorm, no 1 +
w). Looped positions share W_* with their core position but keep their own norm, so pass-1 → pass-2 links (even a
neuron to its own second-pass self) appear. AttnRes routing is ignored: weight-only leads, as in the original.

    EXPLORER_KOTO_CKPT=… [EXPLORER_KOTO_SIZE=…] python explorer/build_connections_koto.py --out CACHE/connections.npz

Same output format as build_connections.py.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runtime_koto  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=24)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rt = runtime_koto.KotoRuntime("/nonexistent")
rt.ensure_loaded()
dev = rt.dev
blocks = rt.koto.layers
L, N, K = len(blocks), blocks[0].ffn.down_proj.in_features, a.k
dt = torch.bfloat16
down = [b.ffn.down_proj.weight.to(dt) for b in blocks]
gate = [b.ffn.gate_proj.weight.to(dt) for b in blocks]
up = [b.ffn.up_proj.weight.to(dt) for b in blocks]
norm = [b.ffn_norm.weight.to(dt) for b in blocks]

best = {s: (torch.full((L, N, K), -1e9, device=dev), torch.zeros((L, N, K), dtype=torch.int16, device=dev),
            torch.zeros((L, N, K), dtype=torch.int16, device=dev), torch.zeros((L, N, K), dtype=torch.int8, device=dev))
        for s in ("down", "up")}


def merge(side, l, z, idx, other_layer, sign):
    bz, bl, bn, bs = best[side]
    cz = torch.cat([bz[l], z], 1)
    cl = torch.cat([bl[l], torch.full_like(idx, other_layer, dtype=torch.int16)], 1)
    cn = torch.cat([bn[l], idx.to(torch.int16)], 1)
    cs = torch.cat([bs[l], sign], 1)
    top = cz.topk(K, dim=1)
    bz[l], bl[l], bn[l], bs[l] = top.values, cl.gather(1, top.indices), cn.gather(1, top.indices), cs.gather(1, top.indices)


t0 = time.time()
with torch.inference_mode():
    for j in range(1, L):
        for i in range(j):
            V = down[i] * norm[j][:, None]
            P = (gate[j] @ V).float() * (up[j] @ V).float()
            M = P.abs()
            zc = (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-8)
            top = zc.topk(K, dim=0)
            sgn = torch.where(P.gather(0, top.indices) > 0, 1, -1).to(torch.int8)
            merge("down", i, top.values.T, top.indices.T, j, sgn.T)
            zr = (M - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-8)
            top = zr.topk(K, dim=1)
            sgn = torch.where(P.gather(1, top.indices) > 0, 1, -1).to(torch.int8)
            merge("up", j, top.values, top.indices, i, sgn)
            del P, M, zc, zr
        print(f"position {j} done ({time.time() - t0:.0f}s)", flush=True)

out = {}
for side, (bz, bl, bn, bs) in best.items():
    out[f"{side}_z"] = bz.clamp(min=-60000).cpu().numpy().astype(np.float16)
    out[f"{side}_layer"] = bl.cpu().numpy().astype(np.int8)
    out[f"{side}_n"] = bn.cpu().numpy()
    out[f"{side}_sign"] = bs.cpu().numpy()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.savez_compressed(a.out, **out, k=K)
print("->", a.out, f"{os.path.getsize(a.out) / 1e6:.0f} MB")
