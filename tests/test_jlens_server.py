"""CPU checks for the J-lens views of the kotodama explorer: graceful absence (no J data: no fields, available:
false), the neuron payload and pass-1 / pass-2 pair from a fake index, and the live per-text grid on a tiny looped
AttnRes model (J = I must equal the logit lens; a random J must match a direct computation)."""
import os
import sys
import tempfile
import types

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOTO_SRC = os.environ.get("KOTODAMA_SRC", os.path.expanduser("~/projects/kotodama/src"))
DATA = tempfile.mkdtemp(prefix="jlens-test-")
os.environ.update(EXPLORER_BACKEND="koto", EXPLORER_NO_MODEL="1", EXPLORER_KOTO_DATA=DATA,
                  EXPLORER_KOTO_SIZE="1b-tsumugi")
for var in ("EXPLORER_CACHE", "EXPLORER_RUNS", "EXPLORER_STATS", "EXPLORER_NAMES"):
    os.environ.pop(var, None)
sys.path.insert(0, KOTO_SRC)
sys.path.insert(0, os.path.join(ROOT, "explorer"))
pytest.importorskip("kotodama")

import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

CACHE = os.path.join(DATA, "cache")
L, N, K, V = 72, 8, 12, 60
VOCAB = [f"t{i}" for i in range(V)]
client = TestClient(server.app, client=("127.0.0.1", 50000))
HDR = {"host": f"127.0.0.1:{server.PORT}"}


def reset_rt():
    rt = server.rt
    for a in ("_jl_idx", "_jl_J", "_jl_meta"):
        rt.__dict__.pop(a, None)
    rt.__dict__.pop("_jl_common", None)
    rt._vocab = VOCAB
    rt.JLENS_COMMON_FRAC = 0.9      # 8 fake neurons: only the planted token 7 (in every neuron) is layer-wide
    return rt


def write_lens_index():
    rng = np.random.default_rng(0)
    np.savez(os.path.join(CACHE, "lens_index.npz"), up=rng.integers(0, V, (L, N, K)).astype(np.int32),
             down=rng.integers(0, V, (L, N, K)).astype(np.int32))


def write_jlens_index():
    rng = np.random.default_rng(1)
    up = np.stack([np.stack([rng.permutation(V)[:K] for _ in range(N)]) for _ in range(L)]).astype(np.int32)
    down = np.stack([np.stack([rng.permutation(V)[:K] for _ in range(N)]) for _ in range(L)]).astype(np.int32)
    up[45, 3, :10] = up[25, 3, :10]                     # neuron 3: pass 2 reads the same top-10 as pass 1
    up[:, :, 0] = 7                                      # token 7 tops every neuron: a layer-wide artifact
    stab = np.full((L, N), 0.5, np.float16)
    stab[30, 0] = np.nan
    pc = np.full((20, N), 0.25, np.float16)
    np.savez(os.path.join(CACHE, "jlens_index.npz"), up=up, down=down, stab10=stab, pair_cos=pc,
             core=np.array([19, 20], np.int32))
    return up, down


def test_absent_j_data_is_graceful():
    for f in ("jlens_index.npz", "jlens_J.npy", "jlens_meta.json"):
        if os.path.exists(os.path.join(CACHE, f)):
            os.remove(os.path.join(CACHE, f))
    write_lens_index()
    reset_rt()
    r = client.get("/api/lens?layer=25&n=3", headers=HDR).json()
    assert "up" in r and "jlens_up" not in r
    assert client.get("/api/jlens/pair?layer=25&n=3", headers=HDR).json()["available"] is False
    assert client.get("/api/jlens/meta", headers=HDR).json()["available"] is False
    # live model off: the per-text grid refuses with a reason instead of failing
    assert "error" in client.post("/api/jlens", json={"text": "hi"}, headers=HDR).json()


def test_neuron_payload_and_pair_from_index():
    write_lens_index()
    up, down = write_jlens_index()
    reset_rt()
    r = client.get("/api/lens?layer=25&n=3", headers=HDR).json()
    assert r["jlens_up"] == [VOCAB[t] for t in up[25, 3]] and r["jlens_down"] == [VOCAB[t] for t in down[25, 3]]
    assert r["jlens_stab"] == 0.5 and "t7" in r["jlens_common"]
    assert client.get("/api/lens?layer=30&n=0", headers=HDR).json()["jlens_stab"] is None   # NaN -> null
    for layer in (25, 45):                               # either pass opens the same pair
        p = client.get(f"/api/jlens/pair?layer={layer}&n=3", headers=HDR).json()
        assert p["available"] and (p["p"], p["p2"]) == (25, 45)
        assert p["overlap_up"] == 1.0 and p["overlap_up_specific"] == 0.9 and p["cos"] == 0.25   # t7 is layer-wide
        assert p["pass1"]["up"] == [VOCAB[t] for t in up[25, 3, :10]]
    assert client.get("/api/jlens/pair?layer=10&n=3", headers=HDR).json()["available"] is False
    assert client.get("/api/jlens/pair?layer=59&n=3", headers=HDR).json()["available"] is False


@pytest.fixture()
def tiny_rt(monkeypatch):
    from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
    torch.manual_seed(0)
    cfg = KotodamaModelConfig(hidden_size=32, num_layers=7, num_attention_heads=4, num_kv_heads=2, head_dim=8,
                              intermediate_size=48, vocab_size=V, max_position_embeddings=64, attn_impl="sdpa",
                              attn_res=True, attn_res_boundaries=[0, 1, 2, 4, 6], use_liger=False,
                              loop_start=2, loop_len=2, loop_times=2)
    m = KotodamaModel(cfg).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "res_query" in name:
                p.normal_(0, 0.5)
    rt = reset_rt()
    tok = types.SimpleNamespace(decode=lambda ids: "".join(VOCAB[i] for i in ids))
    tok_call = lambda text, add_special_tokens=True: types.SimpleNamespace(input_ids=[1 + ord(c) % (V - 1) for c in text])
    monkeypatch.setattr(rt, "koto", m, raising=False)
    monkeypatch.setattr(rt, "dev", "cpu", raising=False)
    monkeypatch.setattr(rt, "model", object())
    monkeypatch.setattr(rt, "tok", type("Tok", (), {"__call__": staticmethod(tok_call), "decode": staticmethod(tok.decode)})())
    monkeypatch.setattr(server, "NO_MODEL", False)
    return rt, m


def _write_J(J):
    np.save(os.path.join(CACHE, "jlens_J.npy"), J.astype(np.float16))


def test_grid_identity_J_equals_logit_lens(tiny_rt):
    rt, m = tiny_rt
    _write_J(np.tile(np.eye(32), (7, 1, 1)))
    ids = [3, 9, 14, 2, 30, 5]
    gj = rt.jlens_grid(ids, None, 4, "j")
    gl = rt.jlens_grid(ids, None, 4, "logit")
    assert gj["available"] and gj["layers"] == [0, 6] and len(gj["top"]) == 7 and len(gj["top"][0]) == len(ids)
    assert gj["top"] == gl["top"]
    one = rt.jlens_grid(ids, [2, 5], 3, "j", at=-1, targets=[5, 9])
    assert one["positions"] == [5] and len(one["top"]) == 4 and len(one["top"][0][0]) == 3
    assert len(one["target_ranks"][0][0]) == 2 and min(one["target_ranks"][0][0]) >= 1


def test_grid_random_J_matches_direct_readout(tiny_rt):
    from kotodama.model import capture
    rt, m = tiny_rt
    J = np.random.default_rng(3).normal(size=(7, 32, 32)).astype(np.float32)
    _write_J(J)
    ids = [3, 9, 14, 2, 30, 5]
    g = rt.jlens_grid(ids, None, 3, "j")
    _, cap = capture.instrumented_forward(m, torch.tensor([ids]), light=True)
    Jt = torch.from_numpy(J.astype(np.float16).astype(np.float32))
    for li in (0, 3, 6):
        h = cap["residual_partial"][2 * li + 1][0].float()
        u = h @ Jt[li].T
        u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + m.norm.eps) * m.norm.weight.float()
        want = (u @ m.get_lm_head_weight().float().T).topk(3, -1).indices
        assert g["top"][li] == [[VOCAB[t] for t in row] for row in want.tolist()]


def test_jlens_endpoint_shape(tiny_rt):
    _write_J(np.tile(np.eye(32), (7, 1, 1)))
    r = client.post("/api/jlens", json={"text": "abcdef", "layers": [1, 3], "k": 2, "targets": ["a"]}, headers=HDR).json()
    assert r["available"] and len(r["tokens"]) == 6 and r["layers"] == [1, 3]
    assert len(r["top"]) == 3 and all(len(row) == 6 for row in r["top"]) and len(r["top"][0][0]) == 2
    assert len(r["p_top1"][0]) == 6 and r["targets"] == ["a"] and len(r["common"]) == 3
    assert client.post("/api/jlens", json={"text": "abc", "at": 7}, headers=HDR).json().get("error")
    assert client.post("/api/jlens", json={"text": "abc", "lens": "nope"}, headers=HDR).status_code == 400


def test_qwen_runtime_has_no_jlens():
    import runtime
    assert not any(a.startswith("jlens") for a in dir(runtime.Runtime))
