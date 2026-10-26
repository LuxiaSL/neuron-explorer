"""The explorer's live model for a kotodama checkpoint (looped AttnRes, e.g. kotodama-tsumugi): the same Runtime
interface as runtime.py, so server.py and the static app run unchanged.

    EXPLORER_BACKEND=koto EXPLORER_KOTO_CKPT=hf:LuxiaSL/kotodama-tsumugi-1b [EXPLORER_KOTO_SIZE=1b-tsumugi]
    [EXPLORER_KOTO_BOUNDS=0,1,7,9,17,19,39,59] python explorer/server.py
    (kotodama's src on PYTHONPATH; the capture walker is eager, so routing runs fp32 — the served numerics)

What differs from Qwen3.5, and how it is handled:
  * "layer" means an EFFECTIVE position (72 for tsumugi). Looped positions share their attention and FFN modules with
    the core (tsumugi: 39–58 reuse 19–38), so a hook on a shared down_proj fires once per pass. Each position keeps
    its own pre-norms, so pre-hooks on ffn_norm / attn_norm record which position is running, and the shared
    down_proj / o_proj hooks dispatch on it. Reads, pushes and captures are therefore per pass: neuron n at position
    25 and at position 45 has the same weights but different activations, and a push at 25 leaves pass 2 alone. To
    push "the neuron" in both passes, push both positions.
  * AttnRes has no single residual stream. The steering dose unit (runtime.py: "median residual norm leaving the
    layer") becomes the median norm of the state entering the position's FFN (the input to ffn_norm), the vector the
    neuron's write lands beside.
  * Direct-path connections read through each position's own ffn_norm weight (plain RMSNorm, no 1 + w), and ignore
    AttnRes routing (which is input-dependent): weight-only leads, as in the original.
  * Logit lens: final norm → tied embeddings.
  * Prompts are raw text (a base model): a conversation is its contents concatenated; no chat template.
Forward passes run through kotodama.model.capture (the eager AttnRes walker, parity-tested against the model, with a
bit-exact KV cache for generation), which calls every norm / attention / FFN as a module, so the hooks see them.
"""
import json
import logging
import os
import time
import types

import numpy as np
import torch

log = logging.getLogger("explorer.koto")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

import runtime as base

CKPT = os.environ.get("EXPLORER_KOTO_CKPT", "")
SIZE = os.environ.get("EXPLORER_KOTO_SIZE", "1b-tsumugi")
BOUNDS = os.environ.get("EXPLORER_KOTO_BOUNDS") or None
TURN_SEP = os.environ.get("EXPLORER_TURN_SEP", "\n\n")
# Position 0 carries "massive activation" units (e.g. L7 N3437, L12 N3852: ~10^2–10^4 at the first token, exactly 0
# everywhere after). Their typical range over positions >= 1 is 0, so any activation / typical-max view blows up at
# position 0. With EXPLORER_SKIP_POS0=1 (default) every read, "fires" and full-activation view leaves position 0 out;
# reads keep the raw value on the first unit as `pos0_act`.
SKIP_POS0 = os.environ.get("EXPLORER_SKIP_POS0", "1") == "1"
DEFAULT_SAMPLING = {"temperature": 0.75, "top_p": 1.0, "repetition_penalty": 1.2, "frequency_penalty": 0.0, "seed": 0}


def _finite(x, nd):
    """round(float(x), nd), or None for NaN / inf (JSON has no NaN)."""
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def arch():
    """The ARCH block server.py serves, from the kotodama preset (no weights needed)."""
    from kotodama import presets

    s = presets.shape(SIZE)
    n = s["num_layers"]
    lp0, ll, lt = s.get("loop_start"), s.get("loop_len", 0), s.get("loop_times", 1)
    src = [i if lp0 is None or not (lp0 <= i < lp0 + ll * lt) else lp0 + (i - lp0) % ll for i in range(n)]
    return {"n_layers": n, "hidden": s["hidden_size"], "inter": s["intermediate_size"],
            "layer_types": ["full_attention"] * n, "attn_heads": s["num_attention_heads"],
            "kv_heads": s["num_kv_heads"], "linear_k_heads": 0, "linear_v_heads": 0,
            "weights_of": src, "model": f"kotodama:{SIZE}:{os.path.basename(CKPT) or '?'}"}


class _Head:
    """lm_head for tied embeddings: x @ E.T."""

    def __init__(self, emb):
        self.emb = emb

    def __call__(self, x):
        return x @ self.emb.weight.T


class _ModelShim:
    """The slice of the HF model object that runtime.Runtime touches: __call__ → .logits, .lm_head, .dtype."""

    def __init__(self, rt):
        self._rt = rt
        self.lm_head = _Head(rt.koto.embed_tokens)
        self.dtype = next(rt.koto.parameters()).dtype

    def __call__(self, input_ids, **_):
        return types.SimpleNamespace(logits=self._rt._fwd(input_ids))


class _LMShim:
    """The language-model trunk object (runtime.Runtime uses .layers, .norm and calls it for hook-only passes)."""

    def __init__(self, rt):
        self._rt = rt
        self.layers = rt.koto.layers
        self.norm = rt.koto.norm

    def __call__(self, input_ids, **_):
        self._rt._fwd(input_ids)


class KotoRuntime(base.Runtime):
    def ensure_loaded(self):
        if self.model is not None:
            return
        if base.NO_MODEL:
            raise RuntimeError("live model disabled (EXPLORER_NO_MODEL)")
        if not CKPT:
            raise RuntimeError("set EXPLORER_KOTO_CKPT to a kotodama checkpoint")
        with self.lock:
            if self.model is not None:
                return
            self.status = "loading"
            from transformers import AutoTokenizer

            from kotodama import presets
            from kotodama.serve import server as ks

            ks._rope_fp32_override = True               # the kotodama-tsumugi serve law
            ks._attn_res_boundaries_override = BOUNDS
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            m, _ = ks.load_model(CKPT, device=dev, model_size=SIZE, engine="reference")  # no compiled decode engine
            m.eval()
            self.koto, self.dev = m, dev
            self.tok = AutoTokenizer.from_pretrained(presets.TOKENIZER_NAME)
            for blk in m.layers:                       # runtime.py reads layers[i].mlp.*; alias without registering
                blk.__dict__["mlp"] = blk.ffn
            self._pos = None
            self._hooks = {}
            for i, blk in enumerate(m.layers):
                blk.ffn_norm.register_forward_pre_hook(self._enter(i, ffn=True))
                blk.attn_norm.register_forward_pre_hook(self._enter(i, ffn=False))
            seen = set()
            for blk in m.layers:                       # shared (looped) modules get ONE dispatching hook each
                if id(blk.ffn) not in seen:
                    seen.add(id(blk.ffn))
                    blk.ffn.down_proj.register_forward_pre_hook(self._dispatch_mlp)
                if id(blk.attn) not in seen:
                    seen.add(id(blk.attn))
                    blk.attn.o_proj.register_forward_pre_hook(self._dispatch_heads)
            self.lm = _LMShim(self)
            self.model = _ModelShim(self)
            self.status = "ready"

    # ---------- hooks ----------
    def _enter(self, i, ffn):
        def hook(mod, inp):
            self._pos = i
            if ffn and self._seen.get("want_resid") == i:
                self._seen["resid"] = inp[0][0, 1:].float().norm(dim=-1).cpu()
        return hook

    def _dispatch_mlp(self, mod, inp):
        i = self._pos
        h = self._hooks.get(("mlp", i))
        if h is None:
            h = self._hooks[("mlp", i)] = self._hook(i)
        return h(mod, inp)

    def _dispatch_heads(self, mod, inp):
        i = self._pos
        h = self._hooks.get(("heads", i))
        if h is None:
            h = self._hooks[("heads", i)] = self._head_hook(i, self.n_heads(i))
        return h(mod, inp)

    def _fwd(self, ids):
        from kotodama.model import capture

        ids = ids if torch.is_tensor(ids) else torch.tensor(ids, device=self.dev)
        if ids.dim() == 1:
            ids = ids[None]
        return capture.instrumented_forward(self.koto, ids.to(self.dev), light=True)[0]

    def vocab_strings(self):
        if getattr(self, "_vocab", None) is None and self.tok is None:
            from transformers import AutoTokenizer

            from kotodama import presets
            self.tok = AutoTokenizer.from_pretrained(presets.TOKENIZER_NAME)
        return super().vocab_strings()

    # ---------- shape ----------
    def n_heads(self, layer):
        a = self.koto.layers[layer].attn
        return a.o_proj.in_features // a.head_dim

    def layer_resid_p50(self, layer):
        if layer in self.resid_p50:
            return self.resid_p50[layer]
        ids = self.tok(base.REFERENCE_TEXT, return_tensors="pt").input_ids.to(self.dev)
        self._seen["want_resid"] = layer
        with torch.inference_mode():
            self._fwd(ids)
        self._seen.pop("want_resid")
        v = float(self._seen.pop("resid").median())
        self.resid_p50[layer] = v
        return v

    # ---------- prompts: raw text ----------
    def messages_ids(self, messages):
        text = ""
        for m in messages:
            c = m.get("content", "")
            if m.get("role") == "assistant":
                text += c                               # a continuation: no separator before the model's own text
            else:
                text += (TURN_SEP if text else "") + c
        return self.tok(text, add_special_tokens=False).input_ids

    # ---------- steering: KV-cached greedy decode through the capture walker ----------
    activity = {"op": "idle"}   # what the model is doing right now (served at /api/debug)

    # ---------- position 0 ----------
    def all_activations(self, ids, skip_first=False):
        out = super().all_activations(ids, skip_first=skip_first or SKIP_POS0)
        if SKIP_POS0 and not skip_first:   # the base counts argmax/argmin from position 1 when skipping; re-base to 0
            out = {l: (mx, amx + 1, mn, amn + 1) for l, (mx, amx, mn, amn) in out.items()}
        return out

    def full_activations(self, ids):
        acts = super().full_activations(ids)
        if SKIP_POS0 and acts.shape[1]:
            acts[:, 0, :] = 0.0
        return acts

    def read(self, layer, n, text, chat=True, pushes=None, heads=None):
        out = super().read(layer, n, text, chat=chat, pushes=pushes, heads=heads)
        if SKIP_POS0 and out["units"]:
            u = out["units"][0]
            u["pos0_act"] = u["pieces"][0]
            u["pieces"] = [0.0] + u["pieces"][1:]
            u["act"] = max(u["pieces"], key=abs)
        return out

    # ---------- sampling ----------
    def steer_stream(self, layer, n, rows_messages, dose, mode, max_new=200, cancel=None, sampling=None):
        yield from self.steer_multi([(layer, n, dose, mode)], rows_messages, max_new, cancel, sampling=sampling)

    def _picker(self, sampling, nrows):
        """Per-row next-token choice. temperature <= 0 is greedy. Every row draws from its own generator seeded with
        the SAME seed (common random numbers), so the up / zero / down columns differ only by the push."""
        sp = {**DEFAULT_SAMPLING, **(sampling or {})}
        T, top_p = float(sp["temperature"]), float(sp["top_p"])
        rep, freq = float(sp["repetition_penalty"]), float(sp["frequency_penalty"])
        gens = []
        for _ in range(nrows):
            g = torch.Generator(device=self.dev)
            g.manual_seed(int(sp["seed"]))
            gens.append(g)

        def pick(logits, r, context, generated):
            x = logits.float().clone()
            if rep != 1.0 and context:
                ids = torch.tensor(sorted(set(context)), device=x.device)
                v = x[ids]
                x[ids] = torch.where(v > 0, v / rep, v * rep)
            if freq and generated:
                ids, cnt = torch.unique(torch.tensor(generated, device=x.device), return_counts=True)
                x[ids] -= freq * cnt.float()
            if T <= 0:
                return int(x.argmax())
            p = torch.softmax(x / T, dim=-1)
            if 0.0 < top_p < 1.0:
                sp_, si = torch.sort(p, descending=True)
                cut = torch.cumsum(sp_, 0) - sp_ >= top_p
                sp_[cut] = 0.0
                p = torch.zeros_like(p).scatter_(0, si, sp_)
            return int(torch.multinomial(p / p.sum(), 1, generator=gens[r]))
        return pick, sp

    def steer_multi(self, pushes, rows_messages, max_new=200, cancel=None, sampling=None):
        from kotodama.model import capture

        self.ensure_loaded()
        t_req = time.time()
        log.info("steer request: pushes=%s rows=%d max_new=%d (waiting for model lock; holder: %s)",
                 pushes, len(rows_messages), max_new, self.activity.get("op"))
        with self.lock:
            t_lock = time.time()
            log.info("steer: lock acquired after %.2fs", t_lock - t_req)
            self.activity = {"op": "steer", "pushes": [list(p) for p in pushes], "since": t_lock, "tokens": 0}
            if cancel is not None and cancel.is_set():
                yield None, None
                return
            try:
                nrows = len(rows_messages)
                signs = (1, 0, -1) if nrows == 3 else tuple([1] * nrows)
                push = [self._push_entry(layer, n, dose, mode, signs) for layer, n, dose, mode in pushes]
                seqs = [self.messages_ids(m) for m in rows_messages]
                groups = {}
                for r, q in enumerate(seqs):                    # rows of equal length share one batch and cache
                    groups.setdefault(len(q), []).append(r)
                eos = {self.tok.eos_token_id}
                pick, sp = self._picker(sampling, nrows)
                log.info("steer: sampling %s", sp)
                done, out_ids, sent = [False] * nrows, [[] for _ in range(nrows)], [""] * nrows
                state = {}
                log.info("steer: prompt lengths %s → %d batch group(s); push values %s", [len(q) for q in seqs],
                         len(groups), [[round(float(x), 2) for x in v.tolist()] for *_, v in push])
                t0 = time.time()
                with torch.inference_mode():
                    for T, rows in groups.items():
                        self._push = [(l, n, mo, v[rows]) for l, n, mo, v in push] or None
                        ids = torch.tensor([seqs[r] for r in rows], device=self.dev)
                        logits, _, cache = capture.cached_forward(self.koto, ids, past=None, position_offset=0,
                                                                  light=True, capture_hidden=False)
                        state[T] = [rows, cache, logits[:, -1], T, None]
                    log.info("steer: prefill %.2fs", time.time() - t0)
                    t1 = time.time()
                    for step in range(max_new):
                        if cancel is not None and cancel.is_set():
                            log.info("steer: cancelled by client at step %d", step)
                            break
                        self.activity["tokens"] = step
                        if step and step % 50 == 0:
                            log.info("steer: %d tokens (%.1f tok/s per row)", step, step / max(time.time() - t1, 1e-6))
                        for T, st in state.items():
                            rows, cache, last, length, _ = st
                            picks = [pick(last[j], r, seqs[r] + out_ids[r], out_ids[r]) for j, r in enumerate(rows)]
                            st[4] = torch.tensor(picks, device=self.dev)
                            for j, r in enumerate(rows):
                                if done[r]:
                                    continue
                                t = picks[j]
                                if t in eos:
                                    done[r] = True
                                    continue
                                out_ids[r].append(t)
                                text = self.tok.decode(out_ids[r], skip_special_tokens=True)
                                if not text.endswith("�"):
                                    yield r, text[len(sent[r]):]
                                    sent[r] = text
                        if all(done) or any(st[3] + 1 >= self.koto.rope_cos.shape[0] for st in state.values()):
                            break
                        for T, st in state.items():
                            rows, cache, last, length, nxt = st
                            self._push = [(l, n, mo, v[rows]) for l, n, mo, v in push] or None
                            logits, _, cache = capture.cached_forward(self.koto, nxt[:, None], past=cache,
                                                                      position_offset=length, light=True,
                                                                      capture_hidden=False)
                            st[1], st[2], st[3] = cache, logits[:, -1], length + 1
            except Exception:
                log.exception("steer failed")
                raise
            finally:
                self._push = None
                log.info("steer: done in %.2fs (rows done: %s, tokens per row: %s)", time.time() - t_lock, done,
                         [len(x) for x in out_ids])
                self.activity = {"op": "idle"}
        yield None, None

    # ---------- J-lens (Jacobian lens; data from build_jlens_koto.py) ----------
    # J_L maps a change of `partial` right after position L's FFN to the change of h_final (the final routed state
    # before the last norm), averaged over web text; the J-lens reads a state h as softmax(W_U · norm(J_L h)). The
    # logit lens is J = I. Index: jlens_index.npz (lens_index layout + stab10, pair_cos, core); matrices: jlens_J.npy
    # (float16 [L, D, D]); provenance: jlens_meta.json. Every J view is absent (None / available: false) without them.
    JLENS_COMMON_FRAC = 0.03   # a token in the top-12 of >= 3% of a position's neurons is flagged layer-common

    def jlens_index(self):
        if getattr(self, "_jl_idx", None) is None:
            path = os.path.join(base.CACHE, "jlens_index.npz")
            self._jl_idx = False
            if os.path.exists(path):
                try:
                    z = np.load(path)
                    self._jl_idx = {k: z[k] for k in z.files}
                except Exception:
                    log.exception("jlens_index.npz unreadable; J-lens views off")
        return self._jl_idx or None

    def jlens_meta(self):
        if getattr(self, "_jl_meta", None) is None:
            path = os.path.join(base.CACHE, "jlens_meta.json")
            try:
                self._jl_meta = json.load(open(path)) if os.path.exists(path) else {}
            except Exception:
                log.exception("jlens_meta.json unreadable")
                self._jl_meta = {}
            fit_ckpt = self._jl_meta.get("ckpt")
            if fit_ckpt and CKPT and os.path.basename(fit_ckpt) != os.path.basename(CKPT):
                log.warning("J-lens was fitted on %s but the explorer serves %s", fit_ckpt, CKPT)
        return self._jl_meta

    def jlens_common(self, layer):
        """Token ids that sit in the J-lens top-12 (either pole) of >= JLENS_COMMON_FRAC of the position's neurons:
        layer-wide directions the J readout favours, not specific to a neuron (shown faded, flagged)."""
        idx = self.jlens_index()
        if idx is None:
            return set()
        cache = self.__dict__.setdefault("_jl_common", {})
        if layer not in cache:
            ids = np.concatenate([idx["up"][layer].ravel(), idx["down"][layer].ravel()])
            cnt = np.bincount(ids)
            cache[layer] = set(np.nonzero(cnt >= self.JLENS_COMMON_FRAC * idx["up"].shape[1])[0].tolist())
        return cache[layer]

    def jlens_neuron(self, layer, n, k=12):
        """The neuron's write d_n read through J_layer: {jlens_up, jlens_down, jlens_stab, jlens_common} or None."""
        idx = self.jlens_index()
        if idx is None or not 0 <= layer < idx["up"].shape[0]:
            return None
        v, common = self.vocab_strings(), self.jlens_common(layer)
        up, down = idx["up"][layer, n, :k].tolist(), idx["down"][layer, n, :k].tolist()
        st = float(idx["stab10"][layer, n]) if "stab10" in idx else float("nan")
        return {"jlens_up": [v[t] for t in up], "jlens_down": [v[t] for t in down],
                "jlens_stab": None if st != st else round(st, 2),
                "jlens_common": sorted({v[t] for t in up + down if t in common})}

    def lens(self, layer, n, k=16):
        out = super().lens(layer, n, k)
        jl = self.jlens_neuron(layer, n, k)
        if jl is not None:
            out = {**out, **jl}
        return out

    def jlens_pair(self, layer, n, k=10, live=False):
        """A looped core neuron's write read through J at both passes (p, p + loop_len): the two top-k lists (each
        pole), their overlap, the cosine of the two J-images of d_n, and each pass's split-half overlap (the ceiling a
        pass-1 / pass-2 overlap can be read against). live=True recomputes both readouts from the served weights and
        jlens_J.npy (fp16), as a check of the index against this model."""
        idx = self.jlens_index()
        if idx is None or "core" not in idx:
            return {"available": False, "reason": "no jlens_index.npz with a loop core (build_jlens_koto.py)"}
        p0, ll = (int(x) for x in idx["core"])
        if not p0 <= layer < p0 + 2 * ll:
            return {"available": False, "reason": f"L{layer} is not a looped core position ({p0}–{p0 + 2 * ll - 1})"}
        p = layer if layer < p0 + ll else layer - ll
        q = p + ll
        v, out = self.vocab_strings(), {"available": True, "layer": layer, "n": n, "p": p, "p2": q}
        for name, L_ in (("pass1", p), ("pass2", q)):
            common = self.jlens_common(L_)
            out[name] = {"layer": L_, "up": [v[t] for t in idx["up"][L_, n, :k]],
                         "down": [v[t] for t in idx["down"][L_, n, :k]],
                         "stab10": _finite(idx["stab10"][L_, n], 2) if "stab10" in idx else None,
                         "common": sorted({v[t] for t in np.concatenate([idx["up"][L_, n, :k], idx["down"][L_, n, :k]])
                                           if t in common})}
        art = self.jlens_common(p) | self.jlens_common(q)
        for pole in ("up", "down"):
            a, b = set(idx[pole][p, n, :k].tolist()), set(idx[pole][q, n, :k].tolist())
            out[f"overlap_{pole}"] = round(len(a & b) / k, 2)
            # the same, counting only tokens that are not layer-wide J directions at either pass (those inflate overlap)
            out[f"overlap_{pole}_specific"] = round(len((a & b) - art) / k, 2)
            out[f"shared_{pole}"] = [v[t] for t in idx[pole][p, n, :k] if int(t) in b]
        out["cos"] = _finite(idx["pair_cos"][p - p0, n], 3) if "pair_cos" in idx else None
        out["note"] = ("overlap = shared top-k / k (overlap_*_specific: shared tokens that are not layer-wide J "
                       "directions at either pass); read it against stab10 (each pass's split-half top-10 overlap, the "
                       "ceiling) and the layer means: p vs p+20 top-10 overlap averages ~0.18 over core neurons")
        if live:
            J = self._jlens_J()
            if J is None:
                out["live"] = {"error": "no jlens_J.npy"}
            else:
                with self.lock, torch.inference_mode():
                    d = self.koto.layers[p].ffn.down_proj.weight[:, n].float()
                    W, g = self.koto.get_lm_head_weight().float(), self.koto.norm.weight.float()
                    live_out = {}
                    for name, L_ in (("pass1", p), ("pass2", q)):
                        lg = W @ (g * (J[L_].float() @ d))
                        live_out[name] = {"up": [v[t] for t in lg.topk(k).indices.tolist()],
                                          "down": [v[t] for t in (-lg).topk(k).indices.tolist()]}
                out["live"] = live_out
        return out

    def _jlens_J(self):
        """jlens_J.npy on the model's device as float16 [L, D, D], loaded once (None without it)."""
        if getattr(self, "_jl_J", None) is None:
            path = os.path.join(base.CACHE, "jlens_J.npy")
            self._jl_J = False
            if os.path.exists(path):
                self.ensure_loaded()
                try:
                    arr = np.load(path, mmap_mode="r")
                    D = self.koto.config.hidden_size
                    if arr.shape != (len(self.koto.layers), D, D):
                        raise ValueError(f"jlens_J.npy shape {arr.shape}, model wants {(len(self.koto.layers), D, D)}")
                    self._jl_J = torch.from_numpy(np.array(arr)).to(self.dev, torch.float16)
                except Exception:
                    log.exception("jlens_J.npy unusable; live J-lens off")
        return self._jl_J if self._jl_J is not False else None

    def jlens_grid(self, ids, layers=None, k=5, lens="j", at=None, targets=None):
        """Per position x token: the top-k tokens (and top-1 probability) of the state `partial` after each FFN,
        read through the J-lens (lens='j': softmax(W_U · norm(J_L h))) or the logit lens on the same state
        (lens='logit': J = I). ids: token ids; layers: [lo, hi] inclusive; at: one token index (negative from the end)
        to return only that column; targets: token ids whose rank (1 = top) is returned at every cell."""
        from kotodama.model import capture

        if lens not in ("j", "logit"):
            raise ValueError("lens must be 'j' or 'logit'")
        J = self._jlens_J() if lens == "j" else None
        if lens == "j" and J is None:
            return {"available": False, "reason": "no jlens_J.npy in the cache (build_jlens_koto.py)"}
        self.ensure_loaded()
        nL = len(self.koto.layers)
        lo, hi = (0, nL - 1) if not layers else (max(0, int(layers[0])), min(nL - 1, int(layers[1])))
        if lo > hi:
            raise ValueError(f"empty layer range {layers}")
        with self.lock, torch.inference_mode():
            x = torch.tensor([ids], device=self.dev)
            _, cap = capture.instrumented_forward(self.koto, x, light=True)
            parts = cap["residual_partial"][1::2]                       # after each position's FFN: (1, T, D) each
            T = x.shape[1]
            cols = list(range(T)) if at is None else [int(at) % T]
            W, g, eps = self.koto.get_lm_head_weight().float(), self.koto.norm.weight.float(), self.koto.norm.eps
            tg = torch.tensor(targets or [], device=self.dev, dtype=torch.long)
            top, prob, ranks, common = [], [], [], []
            for L_ in range(lo, hi + 1):
                h = parts[L_][0, cols].float()                          # (C, D)
                u = h @ J[L_].float().T if J is not None else h
                u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + eps) * g
                lg = u @ W.T                                            # (C, V)
                p = torch.softmax(lg, -1)
                tv, ti = p.topk(k, -1)
                top.append(ti.tolist())
                cm = self.jlens_common(L_) if lens == "j" else set()
                common.append(sorted({t for row in top[-1] for t in row if t in cm}))
                prob.append([round(float(r[0]), 3) for r in tv])
                if len(tg):
                    ranks.append(((lg.unsqueeze(1) > lg[:, tg].unsqueeze(-1)).sum(-1) + 1).tolist())   # (C, n_tg)
        v = self.vocab_strings()
        return {"available": True, "lens": lens, "layers": [lo, hi], "positions": cols, "k": k,
                "top": [[[v[t] for t in row] for row in layer] for layer in top], "p_top1": prob,
                "common": [[v[t] for t in c] for c in common],
                **({"target_ranks": ranks} if len(tg) else {})}

    # ---------- connections: plain RMSNorm weights ----------
    def connections(self, layer, n, k=24, lens_k=5):
        ci = self.con_index()
        if ci is not None and k <= ci["k"]:
            return super().connections(layer, n, k, lens_k)
        key = (layer, n, k, lens_k)
        if key in self._con_cache:
            return self._con_cache[key]
        self.ensure_loaded()
        with self.lock, torch.inference_mode():
            dt = self.model.dtype
            L = self.koto.layers
            d = L[layer].ffn.down_proj.weight[:, n]
            w_me = L[layer].ffn_norm.weight.to(dt)
            r_g, r_u = L[layer].ffn.gate_proj.weight[n] * w_me, L[layer].ffn.up_proj.weight[n] * w_me
            out = {"down": [], "up": []}
            for j, blk in enumerate(L):
                if j == layer:
                    continue
                if j > layer:
                    v = d * blk.ffn_norm.weight.to(dt)
                    g, u = (blk.ffn.gate_proj.weight @ v).float(), (blk.ffn.up_proj.weight @ v).float()
                    side = "down"
                else:
                    W = blk.ffn.down_proj.weight
                    g, u = (r_g @ W).float(), (r_u @ W).float()
                    side = "up"
                prod = g * u
                mag = prod.abs()
                z = (mag - mag.mean()) / (mag.std() + 1e-8)
                top = z.topk(min(k, z.numel()))
                out[side] += [{"layer": j, "n": int(i), "z": round(float(sc), 2), "sign": 1 if prod[i] > 0 else -1}
                              for sc, i in zip(top.values.tolist(), top.indices.tolist())]
            for side in out:
                out[side] = sorted(out[side], key=lambda c: -c["z"])[:k]
            every = out["down"] + out["up"]
            if every and lens_k:
                D = torch.stack([L[c["layer"]].ffn.down_proj.weight[:, c["n"]] for c in every])
                logits = self.model.lm_head(self.lm.norm(D[:, None]))[:, 0].float()
                for c, row, row_neg in zip(every, logits.topk(lens_k).indices.tolist(),
                                           (-logits).topk(lens_k).indices.tolist()):
                    c["lens"] = [self.tok.decode([t]) for t in row]
                    c["lens_neg"] = [self.tok.decode([t]) for t in row_neg]
        self._con_cache[key] = out
        return out
