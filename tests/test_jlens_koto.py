"""CPU checks for the J-lens build (explorer/build_jlens_koto.py) on tiny fake J and weights."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explorer"))
import build_jlens_koto as bj  # noqa: E402

D, N, V, L, K = 8, 6, 20, 4, 3


def _fake(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64).float()
    return r(2 * L, D, D), r(2 * L, D, D), [r(D, N) for _ in range(L)], r(V, D), r(D).abs() + 0.1


def test_pooled_ffn_J_takes_ffn_sites_and_pools_halves():
    sa, sb, *_ = _fake()
    J = bj.pooled_ffn_J(sa, 50, sb, 50)
    assert J.shape == (L, D, D)
    for li in range(L):
        torch.testing.assert_close(J[li], (sa[2 * li + 1] + sb[2 * li + 1]) / 100)
    torch.testing.assert_close(bj.pooled_ffn_J(sa, 5)[2], sa[5] / 5)


def test_col_readout_matches_brute_force():
    sa, _, Wd, W_U, gamma = _fake()
    J = sa[1]
    up, down = bj.col_readout(J, Wd[0], W_U, gamma, K, chunk=4)   # chunk < N exercises the chunking
    for n in range(N):
        logits = W_U @ (gamma * (J @ Wd[0][:, n]))
        assert up[n].tolist() == logits.topk(K).indices.tolist()
        assert down[n].tolist() == (-logits).topk(K).indices.tolist()


def test_identity_J_is_the_logit_lens():
    _, _, Wd, W_U, gamma = _fake(1)
    up, _ = bj.col_readout(torch.eye(D), Wd[0], W_U, gamma, K)
    ll = ((Wd[0] * gamma[:, None]).T @ W_U.T).topk(K, -1).indices
    assert torch.equal(up, ll)


def test_build_index_shapes_stab_and_pairs():
    sa, sb, Wd, W_U, gamma = _fake(2)
    J = bj.pooled_ffn_J(sa, 1, sb, 1)
    idx = bj.build_index(J, lambda li: Wd[li], W_U, gamma, K, bj.pooled_ffn_J(sa, 1), bj.pooled_ffn_J(sa, 1),
                         loop=(1, 1))
    assert idx["up"].shape == idx["down"].shape == (L, N, K) and idx["up"].dtype == np.int32
    assert np.all(idx["stab10"] == 1.0)                 # identical halves: full split-half overlap
    assert idx["pair_cos"].shape == (1, N) and idx["core"].tolist() == [1, 1]
    want = torch.nn.functional.cosine_similarity((J[1] @ Wd[1]) * gamma[:, None], (J[2] @ Wd[1]) * gamma[:, None], dim=0)
    np.testing.assert_allclose(idx["pair_cos"][0].astype(np.float32), want.numpy(), atol=2e-3)


def test_overlap_rows():
    a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    b = torch.tensor([[3, 2, 9], [7, 8, 9]])
    np.testing.assert_allclose(bj.overlap_rows(a, b).numpy(), [2 / 3, 0.0], rtol=1e-6)
