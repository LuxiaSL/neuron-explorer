"""CPU checks for the draft naming layer: a missing or broken drafts file is an empty layer, the file is re-read when it
changes, units and unit pages carry `draft` beside an unchanged hand `name`, and search lists hand-name and draft-name
hits ahead of lens hits."""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOTO_SRC = os.environ.get("KOTODAMA_SRC", os.path.expanduser("~/projects/kotodama/src"))
DATA = tempfile.mkdtemp(prefix="drafts-test-")
os.environ.update(EXPLORER_BACKEND="koto", EXPLORER_NO_MODEL="1", EXPLORER_KOTO_DATA=DATA,
                  EXPLORER_KOTO_SIZE="1b-tsumugi")
for var in ("EXPLORER_CACHE", "EXPLORER_RUNS", "EXPLORER_STATS", "EXPLORER_NAMES", "EXPLORER_DRAFTS"):
    os.environ.pop(var, None)
sys.path.insert(0, KOTO_SRC)
sys.path.insert(0, os.path.join(ROOT, "explorer"))
pytest.importorskip("kotodama")

from fastapi.testclient import TestClient  # noqa: E402

# server is imported at test time, not at collection, so that a module collected later which imports it with its own
# data directory (test_jlens_server.py) still gets that directory; these checks hold for either
server = client = HDR = None


@pytest.fixture(autouse=True, scope="module")
def _server():
    global server, client, HDR
    import server as srv
    server, client = srv, TestClient(srv.app, client=("127.0.0.1", 50000))
    HDR = {"host": f"127.0.0.1:{srv.PORT}"}
VOCAB = ["apple", "banana", "cherry", "date", "elder", "fig"]
DRAFT = {"cluster": 7, "cluster_name": "fruit words", "category": "lexical", "family": "food", "agreement": "high",
         "label": None}


@pytest.fixture
def files(tmp_path, monkeypatch):
    names, drafts = tmp_path / "names.json", tmp_path / "drafts.json"
    monkeypatch.setattr(server, "NAMES", str(names))
    monkeypatch.setattr(server, "DRAFTS", str(drafts))
    monkeypatch.setattr(server, "_drafts", {"key": None, "neurons": {}})
    return names, drafts


def write(path, obj, mtime=None):
    path.write_text(json.dumps(obj))
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_default_path_sits_beside_names():
    assert os.path.dirname(server.DRAFTS) == os.path.dirname(os.path.abspath(server.NAMES))
    assert os.path.basename(server.DRAFTS) == "drafts.json"


def test_missing_and_broken_files_are_empty(files):
    _, drafts = files
    assert server.drafts() == {}
    drafts.write_text("{not json")
    assert server.drafts() == {}
    write(drafts, {"meta": {}, "neurons": ["not", "a", "map"]}, mtime=1_000)
    assert server.drafts() == {}


def test_reloads_when_the_file_changes(files):
    _, drafts = files
    write(drafts, {"meta": {"k": 1}, "neurons": {"L3/N5": DRAFT, "L3/N6": "junk"}}, mtime=1_000)
    assert server.drafts() == {"L3/N5": DRAFT}
    write(drafts, {"neurons": {"L4/N1": {**DRAFT, "label": "orchard"}}}, mtime=2_000)
    assert list(server.drafts()) == ["L4/N1"]
    assert server.draft_name(server.drafts()["L4/N1"]) == "orchard"
    assert server.draft_name(DRAFT) == "fruit words"
    assert server.draft_brief(DRAFT) == "fruit words · lexical · high agreement"
    assert server.draft_brief(None) is None


def fake_layer(layer):
    units = [{"kind": "neuron", "neuron": 5, "z": 3.0, "null_pct": 0.1, "words": [["apple", 3, 2.0]]},
             {"kind": "neuron", "neuron": 6, "z": 2.0, "null_pct": 0.2, "words": [["fig", 2, 1.0]]},
             {"kind": "random", "neuron": 5, "z": 0.5, "null_pct": 0.9, "words": []}]
    return {"units": units, "rows": [], "by_unit": {}, "meta": {"prompts": {}}, "path": "/x/run.jsonl"}


def test_units_and_unit_carry_the_draft(files, monkeypatch):
    names, drafts = files
    write(names, {"L3/N6": "hand name"})
    write(drafts, {"neurons": {"L3/N5": DRAFT, "L3/N6": {**DRAFT, "label": "fig"}}}, mtime=1_000)
    monkeypatch.setattr(server, "load_layer", fake_layer)
    rows = client.get("/api/units", params={"layer": 3}, headers=HDR).json()
    by = {(r["kind"], r["n"]): r for r in rows}
    assert by[("neuron", 5)]["name"] == "" and by[("neuron", 5)]["draft"] == DRAFT
    assert by[("neuron", 6)]["name"] == "hand name" and by[("neuron", 6)]["draft"]["label"] == "fig"
    assert by[("random", 5)]["draft"] is None
    u = client.get("/api/unit", params={"layer": 3, "n": 5}, headers=HDR).json()
    assert u["name"] == "" and u["draft"] == DRAFT
    assert client.get("/api/unit", params={"layer": 9, "n": 1}, headers=HDR).json()["draft"] is None


def test_search_lists_name_and_draft_hits_first(files, monkeypatch):
    names, drafts = files
    write(names, {"L1/N2": "fruit bowl", "L0/N0": "unrelated"})
    write(drafts, {"neurons": {"L1/N2": DRAFT, "L2/N3": DRAFT, "L0/N1": {**DRAFT, "cluster_name": "x", "label": "fruity"},
                               "L2/N0": {**DRAFT, "cluster_name": "colours"}}}, mtime=1_000)
    up = np.zeros((4, 8, 12), dtype=np.int32)
    up[3, 7, 0] = VOCAB.index("fig")
    monkeypatch.setattr(server, "lens_index", lambda: {"up": up, "down": np.zeros_like(up)})
    monkeypatch.setattr(server.rt, "vocab_strings", lambda: VOCAB, raising=False)
    monkeypatch.setattr(server.rt, "steer_index", lambda: None, raising=False)
    r = client.get("/api/search", params={"q": "fruit"}, headers=HDR).json()
    got = [(x["layer"], x["n"], x["src"]) for x in r["results"]]
    assert got == [(1, 2, "name"), (0, 1, "draft"), (2, 3, "draft")]
    assert r["named"] == 1 and r["drafted"] == 2 and r["tokens"] == 0
    assert all(x["sign"] is None and x["rank"] is None for x in r["results"])
    assert r["results"][1]["draft"]["label"] == "fruity" and r["results"][0]["name"] == "fruit bowl"
    r = client.get("/api/search", params={"q": "fig"}, headers=HDR).json()
    assert [(x["layer"], x["n"], x["src"], x["sign"]) for x in r["results"]] == [(3, 7, "lens", "+")]
    assert r["results"][0]["draft"] is None
