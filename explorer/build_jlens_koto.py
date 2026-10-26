"""Builds the J-lens (Jacobian lens) data the kotodama explorer reads, from a fitted J (the jlens fit: J_A.pt / J_B.pt).

    python explorer/build_jlens_koto.py --jdir <dir with J_A.pt, J_B.pt> --ckpt <checkpoint> --out <explorer cache dir>
        [--size 1b-tsumugi] [--k 12] [--research_dir <jlens fit code dir, for the parity check>]

What the fit holds (one dict per half): "sum" = (144, D, D) fp32, the per-prompt J summed over "n" prompts, where a
prompt's J_s = mean over source positions t >= "skip" of sum over target positions t' of d h_final[t'] / d partial_s[t].
Sites: s = 2 * position + 0 (after the attention sublayer) | 1 (after the FFN sublayer); state = the block-local running
sum `partial` right after that sublayer; target h_final = the final AttnRes routing output, before the last RMSNorm.
Rows of J index output dims. The explorer's J for position L is the pooled mean over both halves at the FFN site:

    J_L = (sum_A[2L+1] + sum_B[2L+1]) / (n_A + n_B)

(the FFN site is where a neuron's write lands: the FFN output a * W_down[:, n] is added to `partial` there; for looped
positions the down_proj module is shared, so W_down at position p and p+20 is the same matrix).

Writes into --out:
  * jlens_index.npz: up / down int32 [L, N, k] (the same layout as lens_index.npz): the top-k tokens of
    W_U (gamma * J_L W_down[:, n]) and of its negative, for every position L and neuron n (the RMS scale of the final
    norm does not change a ranking, so it is left out, as in the fit's readout); stab10 float16 [L, N]: top-10 overlap of
    the readout under J_A vs J_B (the split-half reproducibility of that neuron's readout); pair_cos float16
    [loop_len, N]: cosine of gamma * J_p d_n and gamma * J_{p+loop_len} d_n for core positions p (same write, the two
    passes' downstream maps); core = [loop_start, loop_len].
  * jlens_J.npy: float16 [L, D, D], J_L for live per-text readout (see the parity report for why fp16 holds).
  * jlens_meta.json: provenance (fit files, n prompts, skip, ckpt), shapes and the parity numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# fixed, non-corpus sentences for the live-state parity check (fp16 vs fp32 J on real model states)
PARITY_TEXTS = [
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived.",
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0))\n    return out + a + b",
    "She asked me what I thought of the plan, and I told her honestly that it seemed rushed.",
    "The bag holds the shell. Then the bag is emptied and the camera is put in it. In the end, the bag holds the",
]


def log(msg: str, **kw) -> None:
    print(json.dumps({"t": round(time.time(), 1), "msg": msg, **kw}), flush=True)


# ---------- the pure parts (tested on CPU in tests/test_jlens_koto.py) ----------
def pooled_ffn_J(sum_a: torch.Tensor, n_a: int, sum_b: torch.Tensor | None = None, n_b: int = 0) -> torch.Tensor:
    """(n_sites, D, D) fit sums -> (L, D, D) pooled mean at the FFN sites (s = 2L + 1)."""
    if sum_a.shape[0] % 2:
        raise ValueError(f"expected 2 sites per position, got {sum_a.shape[0]} sites")
    tot = sum_a[1::2].float()
    if sum_b is not None:
        tot = tot + sum_b[1::2].float()
    return tot / float(n_a + n_b)


def col_readout(J: torch.Tensor, Wd: torch.Tensor, W_U: torch.Tensor, gamma: torch.Tensor, k: int,
                chunk: int = 1024, want_down: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Top-k token ids of W_U (gamma * J Wd[:, n]) (up) and of its negative (down) for every column n.
    J (D, D), Wd (D, N), W_U (V, D), gamma (D,) -> up (N, k), down (N, k) | None. All fp32 math."""
    X = J.float() @ Wd.float()                                   # (D, N)
    ups, downs = [], []
    for i in range(0, X.shape[1], chunk):
        lg = (X[:, i:i + chunk] * gamma.float()[:, None]).T @ W_U.float().T   # (c, V)
        ups.append(lg.topk(k, dim=-1).indices)
        if want_down:
            downs.append((-lg).topk(k, dim=-1).indices)
    return torch.cat(ups), (torch.cat(downs) if want_down else None)


def overlap_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-row |a_i ∩ b_i| / k for (N, k) index tensors -> (N,) float."""
    return (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().mean(-1)


def pair_cosine(J1: torch.Tensor, J2: torch.Tensor, Wd: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Per-neuron cosine of gamma * J1 d_n and gamma * J2 d_n -> (N,)."""
    g = gamma.float()[:, None]
    return F.cosine_similarity((J1.float() @ Wd.float()) * g, (J2.float() @ Wd.float()) * g, dim=0)


def build_index(J: torch.Tensor, down_of, W_U: torch.Tensor, gamma: torch.Tensor, k: int,
                J_a: torch.Tensor | None = None, J_b: torch.Tensor | None = None,
                loop: tuple[int, int] | None = None, progress=None) -> dict[str, np.ndarray]:
    """The whole jlens_index: J (L, D, D) pooled; down_of(L) -> W_down (D, N) of position L; optional split halves
    J_a / J_b (L, D, D) for stab10; loop = (loop_start, loop_len) for pair_cos."""
    L = J.shape[0]
    N = down_of(0).shape[1]
    up = np.zeros((L, N, k), np.int32)
    down = np.zeros((L, N, k), np.int32)
    stab = np.full((L, N), np.nan, np.float16)
    for li in range(L):
        Wd = down_of(li)
        u, d = col_readout(J[li], Wd, W_U, gamma, k)
        up[li], down[li] = u.cpu().numpy(), d.cpu().numpy()
        if J_a is not None and J_b is not None:
            ta, _ = col_readout(J_a[li], Wd, W_U, gamma, 10, want_down=False)
            tb, _ = col_readout(J_b[li], Wd, W_U, gamma, 10, want_down=False)
            stab[li] = overlap_rows(ta, tb).cpu().numpy().astype(np.float16)
        if progress:
            progress(li, L)
    out = {"up": up, "down": down, "stab10": stab}
    if loop is not None:
        p0, ll = loop
        pc = np.zeros((ll, N), np.float16)
        for j in range(ll):
            pc[j] = pair_cosine(J[p0 + j], J[p0 + j + ll], down_of(p0 + j), gamma).cpu().numpy().astype(np.float16)
        out["pair_cos"] = pc
        out["core"] = np.array([p0, ll], np.int32)
    return out


def sha256_head(path: str, nbytes: int = 1 << 24) -> str:
    """sha256 of the file's first 16 MB plus its size (a cheap identity for multi-GB fit files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    h.update(str(os.path.getsize(path)).encode())
    return h.hexdigest()[:16]


# ---------- the build ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jdir", required=True, help="dir with the fit's J_A.pt and J_B.pt")
    ap.add_argument("--ckpt", required=True, help="the checkpoint J was fitted on (and the explorer serves)")
    ap.add_argument("--size", default="1b-tsumugi")
    ap.add_argument("--out", required=True, help="explorer cache dir (beside lens_index.npz)")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--research_dir", default="", help="the jlens fit code (jlens.py, run_battery.py) for parity")
    ap.add_argument("--battery_results", default="", help="the fit's results.json, to compare loop-neuron overlaps")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False      # exact fp32 ranking (ties at k matter for overlaps)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    from kotodama import presets
    from kotodama.eval.model_loader import load_model
    model = load_model(a.ckpt, config_section=a.size, device=dev, overrides={"rope_fp32": True})
    model.requires_grad_(False)
    shape = presets.shape(a.size)
    L_eff = len(model.layers)
    W_U = model.get_lm_head_weight().float()
    gamma = model.norm.weight.float()
    down_of = lambda li: model.layers[li].ffn.down_proj.weight.float()
    loop = (shape["loop_start"], shape["loop_len"]) if shape.get("loop_start") is not None else None
    log("model", layers=L_eff, loop=loop, dtype=str(next(model.parameters()).dtype))

    pa, pb = os.path.join(a.jdir, "J_A.pt"), os.path.join(a.jdir, "J_B.pt")
    jA = torch.load(pa, map_location="cpu")
    jB = torch.load(pb, map_location="cpu")
    for half, j in (("A", jA), ("B", jB)):
        if j["sum"].shape[0] != 2 * L_eff:
            raise SystemExit(f"J_{half} has {j['sum'].shape[0]} sites; the model has {2 * L_eff} (2 per position)")
    meta_fit = {"n_A": int(jA["n"]), "n_B": int(jB["n"]), "skip": int(jA["skip"]), "T": int(jA["T"]),
                "fit_dtype": jA.get("dtype"), "files": {"J_A.pt": sha256_head(pa), "J_B.pt": sha256_head(pb)}}
    J = pooled_ffn_J(jA["sum"], jA["n"], jB["sum"], jB["n"]).to(dev)
    JA = pooled_ffn_J(jA["sum"], jA["n"]).to(dev)
    JB = pooled_ffn_J(jB["sum"], jB["n"]).to(dev)
    del jA, jB
    log("J loaded", shape=list(J.shape), frob_median=float(J.flatten(1).norm(dim=1).median()),
        absmax=float(J.abs().max()), minutes=round((time.time() - t0) / 60, 1))

    idx = build_index(J, down_of, W_U, gamma, a.k, JA, JB, loop,
                      progress=lambda li, L: li % 12 == 11 and log("index", done=li + 1, of=L))
    log("index built", minutes=round((time.time() - t0) / 60, 1))

    # ---- fp16 storage of J: error and readout parity ----
    J16 = J.half()
    parity: dict = {"absmax_J": float(J.abs().max()), "fp16_max": 65504.0,
                    "rel_frob_err_fp16_max": float(((J16.float() - J).flatten(1).norm(dim=1)
                                                    / J.flatten(1).norm(dim=1)).max())}
    ov_up = []
    for li in range(L_eff):
        u16, _ = col_readout(J16[li], down_of(li), W_U, gamma, a.k, want_down=False)
        ov_up.append(overlap_rows(u16, torch.from_numpy(idx["up"][li]).to(dev)).mean().item())
    parity["neuron_topk_overlap_fp16_vs_fp32"] = {"min_layer": min(ov_up), "mean": float(np.mean(ov_up))}
    # live states: partial after each FFN on fixed sentences, read through fp32 vs fp16 J
    from kotodama.model import capture
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(presets.TOKENIZER_NAME)
    top1_agree, top5_ov = [], []
    for text in PARITY_TEXTS:
        ids = torch.tensor([tok(text, add_special_tokens=False).input_ids], device=dev)
        _, cap = capture.instrumented_forward(model, ids, light=True)
        H = torch.stack(cap["residual_partial"][1::2])[:, 0].float()        # (L, T, D)
        for li in range(L_eff):
            l32 = (H[li] @ J[li].T * gamma) @ W_U.T
            l16 = (H[li] @ J16[li].float().T * gamma) @ W_U.T
            t32, t16 = l32.topk(5, -1).indices, l16.topk(5, -1).indices
            top1_agree.append((t32[:, 0] == t16[:, 0]).float().mean().item())
            top5_ov.append(overlap_rows(t32, t16).mean().item())
    parity["live_states_fp16_vs_fp32"] = {"texts": len(PARITY_TEXTS), "top1_agree_mean": float(np.mean(top1_agree)),
                                         "top1_agree_min_layer_text": float(np.min(top1_agree)),
                                         "top5_overlap_mean": float(np.mean(top5_ov))}
    log("fp16 parity", **parity)

    # ---- against the research code: same neurons through its own readout (col_top) ----
    if a.research_dir:
        sys.path.insert(0, a.research_dir)
        import jlens as rj  # noqa: E402
        from run_battery import col_top  # noqa: E402
        Wr, gr = rj.unembed(model)
        g = torch.Generator().manual_seed(0)
        sample_layers = [3, 18, 19, 25, 38, 39, 45, 58, 65, 71]
        per = {}
        for li in sample_layers:
            cols = torch.randperm(down_of(li).shape[1], generator=g)[:256].to(dev)
            X = J[li] @ down_of(li)[:, cols]
            ref = col_top(Wr, gr, X, k=10)
            mine = torch.from_numpy(idx["up"][li]).to(dev)[cols, :10]
            per[li] = {"top10_set_overlap": overlap_rows(ref, mine).mean().item(),
                       "top10_exact_order": (ref == mine).all(-1).float().mean().item()}
        parity["vs_research_col_top"] = per
        log("research parity", **{str(k): v for k, v in per.items()})
    if a.battery_results and os.path.exists(a.battery_results) and loop is not None:
        r = json.load(open(a.battery_results))["loop_neurons"]
        p0, ll = loop
        mine_pp = [overlap_rows(torch.from_numpy(idx["up"][p0 + j, :, :10]),
                                torch.from_numpy(idx["up"][p0 + j + ll, :, :10])).mean().item() for j in range(ll)]
        mine_ab = [float(np.nanmean(idx["stab10"][p0 + j].astype(np.float32))) for j in range(ll)]
        mine_cos = [float(idx["pair_cos"][j].astype(np.float32).mean()) for j in range(ll)]
        parity["vs_battery_loop_neurons"] = {
            "top10_p_p20_max_abs_diff": float(np.max(np.abs(np.array(mine_pp) - np.array(r["top10_p_p20"])))),
            "ceiling_AB_max_abs_diff": float(np.max(np.abs(np.array(mine_ab) - np.array(r["top10_ceiling_AB_p"])))),
            "cos_p_p20_max_abs_diff": float(np.max(np.abs(np.array(mine_cos) - np.array(r["cos_p_p20"])))),
            "top10_p_p20_mean_mine": float(np.mean(mine_pp)), "top10_p_p20_mean_battery": float(np.mean(r["top10_p_p20"]))}
        log("battery parity", **parity["vs_battery_loop_neurons"])

    # ---- write (tmp + rename, so a reader never sees half a file) ----
    ip = os.path.join(a.out, "jlens_index.npz")
    np.savez(ip + ".tmp.npz", **idx)
    os.replace(ip + ".tmp.npz", ip)
    jp = os.path.join(a.out, "jlens_J.npy")
    np.save(jp + ".tmp.npy", J16.cpu().numpy())
    os.replace(jp + ".tmp.npy", jp)
    meta = {"what": "J-lens for the kotodama explorer (build_jlens_koto.py)", "ckpt": a.ckpt, "size": a.size,
            "site": "partial after each position's FFN sublayer (fit site 2L+1)",
            "target": "h_final: final AttnRes routing output before the last RMSNorm",
            "aggregation": "mean over prompts (both halves pooled), per prompt mean over sources t>=skip of the sum over targets",
            "readout": "softmax(W_U · RMSNorm_final(J_L h)); index ranks W_U (gamma * J_L d_n)",
            "k": a.k, "layers": L_eff, "hidden": int(J.shape[-1]), "J_dtype": "float16", "loop": list(loop) if loop else None,
            "fit": meta_fit, "parity": parity, "built_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(a.out, "jlens_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log("written", out=a.out, minutes=round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main()
