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
import logging
import os
import time
import types

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
