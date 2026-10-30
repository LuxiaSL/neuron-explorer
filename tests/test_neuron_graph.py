"""CPU checks for the measured neuron graph: the accessor on a tiny synthetic graph file (padding, overflowed edges,
NaN reliability, silent neurons, bounds), graceful absence (no EXPLORER_GRAPH, a missing, broken or foreign file: no
graph), and the /api/graph and agent routes on the kotodama backend."""
import json
import os
import sys
import tempfile

import numpy as np
import pytest
from safetensors.numpy import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "explorer"))

import neuron_graph  # noqa: E402

P, W, KI, KO = 3, 4, 5, 3          # positions, neurons per position, stored inputs, stored outputs
R = P * W
INF = float("inf")


def gid(layer, n):
    return layer * W + n


def make_graph(path, **override):
    """A graph with three positions of four neurons. L1 N2 has four inputs (one padded slot), the first an overflowed
    +inf, and two outputs; L1 N3 has two -inf inputs; L2 N0 is near-silent (fan_in 0, NaN reliability); position 0 has
    no inputs (fan_in 0, NaN reliability), the last position no outputs."""
    in_idx = np.full((R, KI), -1, np.int32)
    in_val = np.zeros((R, KI), np.float32)
    out_idx = np.full((R, KO), -1, np.int32)
    out_val = np.zeros((R, KO), np.float32)
    fan_in = np.zeros((R, 2), np.float32)
    fan_out = np.zeros((R, 2), np.float32)
    rel = np.full((R, 3), np.nan, np.float32)
    n_chunks = np.zeros(R, np.int16)
    for g in range(W, R):              # every neuron past position 0 has measured input
        fan_in[g] = (100.0, 10.0)
        rel[g] = (0.9, 0.75, 0.2)
        n_chunks[g] = 8
    for g in range(0, 2 * W):          # every neuron before the last position has measured output
        fan_out[g] = (50.0, -5.0)
    i = gid(1, 2)
    in_idx[i, :4] = [gid(0, 1), gid(0, 3), gid(0, 0), gid(0, 2)]
    in_val[i, :4] = [INF, -8.0, 4.0, 2.0]
    out_idx[i, :2] = [gid(2, 1), gid(2, 3)]
    out_val[i, :2] = [-10.0, 5.0]
    rel[i] = (0.65, 0.5, 0.18)
    j = gid(1, 3)
    in_idx[j, :3] = [gid(0, 0), gid(0, 1), gid(0, 2)]
    in_val[j, :3] = [-INF, -INF, 3.0]
    rel[j] = (0.4, 0.25, 0.1)
    s = gid(2, 0)
    fan_in[s] = (0.0, 0.0)
    rel[s] = np.nan
    t = {"in_idx": in_idx, "in_val": in_val, "out_idx": out_idx, "out_val": out_val, "fan_in": fan_in,
         "fan_out": fan_out, "reliability": rel, "n_chunks": n_chunks,
         "fan_out_chunks": np.array([0, 7, 8], np.int16)}
    t.update(override)
    t = {k: v for k, v in t.items() if v is not None}
    save_file(t, str(path), metadata={"format": "test-graph"})
    return path


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(neuron_graph, "_graph", {"key": None, "graph": None})


@pytest.fixture
def g(tmp_path):
    return neuron_graph.load(str(make_graph(tmp_path / "graph.safetensors")))


def test_absent_means_off(tmp_path):
    assert neuron_graph.load(None) is None
    assert neuron_graph.load("") is None
    assert neuron_graph.load(str(tmp_path / "missing.safetensors")) is None


def test_broken_or_foreign_files_are_off(tmp_path, capsys):
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a safetensors file")
    assert neuron_graph.load(str(junk)) is None
    assert neuron_graph.load(str(make_graph(tmp_path / "noin.safetensors", in_val=None))) is None
    bad = make_graph(tmp_path / "rows.safetensors", fan_in=np.zeros((R + 1, 2), np.float32))
    assert neuron_graph.load(str(bad)) is None
    split = make_graph(tmp_path / "split.safetensors", fan_out_chunks=np.zeros(5, np.int16))
    assert neuron_graph.load(str(split)) is None
    assert "no graph shown" in capsys.readouterr().err


def test_shape(g):
    assert (g.positions, g.width, g.k_in, g.k_out) == (P, W, KI, KO)
    assert g.meta == {"format": "test-graph"}


def test_inputs_shares_padding_and_overflow(g):
    e = g.edges(1, 2, k=12)
    ins = e["inputs"]
    assert ins["total"] == 100.0 and ins["net"] == 10.0
    assert ins["stored"] == 4  # the -1 slot is not an edge
    got = [(c["layer"], c["n"], c["edge"], c["clipped"]) for c in ins["edges"]]
    # the overflowed +inf edge shows as at least the largest finite edge in its list, with its own sign
    assert got == [(0, 1, 8.0, True), (0, 3, -8.0, False), (0, 0, 4.0, False), (0, 2, 2.0, False)]
    assert [c["share"] for c in ins["edges"]] == [0.08, -0.08, 0.04, 0.02]
    assert ins["shown_share"] == pytest.approx(0.22) and ins["stored_share"] == pytest.approx(0.22)
    outs = e["outputs"]
    assert [(c["layer"], c["n"], c["share"]) for c in outs["edges"]] == [(2, 1, -0.2), (2, 3, 0.1)]
    assert outs["stored"] == 2 and outs["net"] == -5.0
    assert e["reliability"] == {"rho": pytest.approx(0.65), "top64_overlap": 0.5, "coverage": pytest.approx(0.18),
                                "grade": "mid"}
    assert e["n_chunks"] == 8 and e["fan_out_chunks"] == 7
    json.dumps(e, allow_nan=False)  # no NaN or inf reaches a reply


def test_k_limits_the_list_not_the_stored_share(g):
    ins = g.edges(1, 2, k=2)["inputs"]
    assert len(ins["edges"]) == 2
    assert ins["shown_share"] == pytest.approx(0.16) and ins["stored_share"] == pytest.approx(0.22)
    assert len(g.edges(1, 2, k=0)["inputs"]["edges"]) == 1


def test_list_without_a_finite_edge(g):
    ins = g.edges(1, 3)["inputs"]
    assert [(c["edge"], c["clipped"]) for c in ins["edges"]] == [(-3.0, True), (-3.0, True), (3.0, False)]
    assert g.edges(1, 3)["reliability"]["grade"] == "low"
    # every stored edge overflowed: shown as at least 0, never as inf
    g2 = make_graph(os.path.join(tempfile.mkdtemp(), "g.safetensors"),
                    in_val=np.full((R, KI), INF, np.float32), in_idx=np.zeros((R, KI), np.int32))
    neuron_graph._graph.update(key=None)
    e = neuron_graph.load(str(g2)).edges(1, 0)
    assert all(c["edge"] == 0.0 and c["clipped"] for c in e["inputs"]["edges"])
    json.dumps(e, allow_nan=False)


def test_unmeasured_sides_and_nan_reliability(g):
    first = g.edges(0, 1)
    assert first["inputs"] is None and first["outputs"] is not None
    assert first["reliability"] == {"rho": None, "top64_overlap": None, "coverage": None, "grade": None}
    assert g.edges(2, 3)["outputs"] is None
    silent = g.edges(2, 0)
    assert silent["inputs"] is None and silent["reliability"]["rho"] is None
    json.dumps(silent, allow_nan=False)


def test_grade_thresholds():
    assert [neuron_graph.grade(x) for x in (None, 0.95, 0.8, 0.79, 0.6, 0.59)] == [None, "high", "high", "mid", "mid",
                                                                                   "low"]


def test_bounds(g):
    for layer, n in ((-1, 0), (P, 0), (0, W), (0, -1)):
        with pytest.raises(IndexError):
            g.edges(layer, n)


def test_reopens_when_the_file_changes(tmp_path):
    path = make_graph(tmp_path / "graph.safetensors")
    g1 = neuron_graph.load(str(path))
    assert neuron_graph.load(str(path)) is g1
    make_graph(path, fan_in=np.full((R, 2), 200.0, np.float32))
    os.utime(path, ns=(10**18, 10**18))
    g2 = neuron_graph.load(str(path))
    assert g2 is not g1 and g2.edges(1, 2)["inputs"]["total"] == 200.0


# ---------------------------------------------------------------- server routes (kotodama backend, no model)

def _server():
    koto_src = os.environ.get("KOTODAMA_SRC", os.path.expanduser("~/projects/kotodama/src"))
    sys.path.insert(0, koto_src)
    pytest.importorskip("kotodama")
    if "server" not in sys.modules:
        data = tempfile.mkdtemp(prefix="graph-test-")
        os.environ.update(EXPLORER_BACKEND="koto", EXPLORER_NO_MODEL="1", EXPLORER_KOTO_DATA=data,
                          EXPLORER_KOTO_SIZE="1b-tsumugi")
        for var in ("EXPLORER_CACHE", "EXPLORER_RUNS", "EXPLORER_STATS", "EXPLORER_NAMES", "EXPLORER_DRAFTS",
                    "EXPLORER_GRAPH"):
            os.environ.pop(var, None)
    import server
    from fastapi.testclient import TestClient
    return server, TestClient(server.app, client=("127.0.0.1", 50000)), {"host": f"127.0.0.1:{server.PORT}"}


@pytest.fixture
def srv(tmp_path, monkeypatch):
    server, client, hdr = _server()
    monkeypatch.setattr(server, "GRAPH", "")
    monkeypatch.setattr(server, "NAMES", str(tmp_path / "names.json"))
    monkeypatch.setattr(server, "DRAFTS", str(tmp_path / "drafts.json"))
    monkeypatch.setattr(server, "_drafts", {"key": None, "neurons": {}})
    monkeypatch.setitem(server.ARCH, "n_layers", P)
    monkeypatch.setitem(server.ARCH, "inter", W)
    return server, client, hdr


def test_routes_without_a_graph(srv):
    server, client, hdr = srv
    assert server.graph() is None
    assert client.get("/api/graph", params={"layer": 1, "n": 2}, headers=hdr).json() == {"available": False}
    assert client.get("/api/agent/graph", params={"layer": 1, "n": 2}, headers=hdr).json()["available"] is False
    assert client.get("/api/agent/overview", headers=hdr).json()["neuron_graph"] is False


def test_routes_with_a_graph(srv, tmp_path, monkeypatch):
    server, client, hdr = srv
    monkeypatch.setattr(server, "GRAPH", str(make_graph(tmp_path / "graph.safetensors")))
    (tmp_path / "names.json").write_text(json.dumps({"L0/N1": "hand name"}))
    (tmp_path / "drafts.json").write_text(json.dumps({"neurons": {"L0/N3": {"cluster_name": "drafty"}}}))
    r = client.get("/api/graph", params={"layer": 1, "n": 2, "k": 3}, headers=hdr).json()
    assert r["available"] and len(r["inputs"]["edges"]) == 3
    first, second = r["inputs"]["edges"][:2]
    assert (first["name"], first["clipped"], second["draft"]) == ("hand name", True, {"cluster_name": "drafty"})
    assert r["reliability"]["grade"] == "mid"
    assert client.get("/api/graph", params={"layer": 0, "n": 1}, headers=hdr).json()["inputs"] is None
    assert "outside" in client.get("/api/graph", params={"layer": 9, "n": 0}, headers=hdr).json()["error"]
    a = client.get("/api/agent/graph", params={"layer": 1, "n": 2}, headers=hdr).json()
    assert a["inputs"]["edges"][0] == {"neuron": "L0 N1", "share_pct": 8.0, "edge": 8.0, "at_least": True,
                                       "name": "hand name", "draft": None}
    assert a["inputs"]["edges"][1]["draft"] == "drafty" and "at_least" not in a["inputs"]["edges"][1]
    assert a["reliability"] == {"split_half_rho": 0.65, "top64_overlap": 0.5, "grade": "mid"}
    assert client.get("/api/index", headers=hdr).json()["graph"] is True


def test_a_graph_of_another_model_is_off(srv, tmp_path, monkeypatch, capsys):
    server, client, hdr = srv
    monkeypatch.setattr(server, "GRAPH", str(make_graph(tmp_path / "graph.safetensors")))
    monkeypatch.setitem(server.ARCH, "inter", W * 2)
    assert server.graph() is None
    assert client.get("/api/graph", params={"layer": 1, "n": 2}, headers=hdr).json() == {"available": False}
    assert "no graph shown" in capsys.readouterr().err
