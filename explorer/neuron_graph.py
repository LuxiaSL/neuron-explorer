"""The measured neuron graph: each neuron's strongest inputs and outputs over a text corpus, read from one safetensors
file (EXPLORER_GRAPH; without it the explorer has no graph). For kotodama-tsumugi the file is the public dataset
https://huggingface.co/datasets/LuxiaSL/kotodama-tsumugi-1b-neuron-graph, whose card defines every array.

An edge u -> v is the summed activation x gradient of v's activation with respect to u's over the corpus: how much u's
activity drives v by every path (attention, residual routing, other neurons), signed. This is a measured graph, unlike
the weight-only direct-path links of /api/connections. A neuron's global id is position * width + index; on kotodama a
position is an effective position, the explorer's "layer".

Arrays (R = positions * width rows; K_in, K_out = the stored list lengths):
  in_idx / in_val    [R, K_in]   strongest inputs, strongest first (global ids, -1 = none) and their signed edges
  out_idx / out_val  [R, K_out]  strongest outputs among the stored input lists, the same way
  fan_in / fan_out   [R, 2]      sum |edge| and sum edge over ALL sources / targets (shares are taken against these)
  reliability        [R, 3]      split-half Spearman rho of the input list, overlap of the halves' top 64, share of
                                 input mass the stored inputs carry (NaN where the neuron has no measured input)
  n_chunks           [R]         corpus chunks behind the neuron's input list
  fan_out_chunks     [positions] chunks per position that recorded fan-out

Rows are read on demand through safetensors' memory map, so opening the file costs no RAM. An edge too large for the
measurement's storage is +-inf in the file: it is shown as at least the largest finite edge in the same list, flagged
`clipped`, and never breaks a reply.
"""
from __future__ import annotations

import math
import os
import sys
import threading
from typing import Optional, TypedDict

import numpy as np

KEYS = ("in_idx", "in_val", "out_idx", "out_val", "fan_in", "fan_out", "reliability", "n_chunks", "fan_out_chunks")


class Edge(TypedDict):
    layer: int
    n: int
    edge: float      # the signed edge (a clipped edge: the largest finite edge in its list, with its own sign)
    share: float     # edge / the neuron's total |edge| on that side, signed
    clipped: bool    # true: the stored edge overflowed; the real edge is at least this large


class Side(TypedDict):
    total: float         # sum |edge| over every source (inputs) or target (outputs)
    net: float           # sum edge over the same
    stored: int          # how many edges the file keeps for this neuron on this side
    shown_share: float   # sum |edge| of the listed edges / total
    stored_share: float  # sum |edge| of every stored edge / total
    edges: list[Edge]


class Reliability(TypedDict):
    rho: Optional[float]            # split-half Spearman rho of the input list
    top64_overlap: Optional[float]  # overlap of the two halves' top 64 inputs, 0-1
    coverage: Optional[float]       # share of the input mass the stored inputs carry
    grade: Optional[str]            # "high" (rho >= .8), "mid" (.6-.8), "low" (< .6); None without a rho


class NeuronEdges(TypedDict):
    layer: int
    n: int
    inputs: Optional[Side]    # None: nothing measured feeds it (the first position, or a near-silent neuron)
    outputs: Optional[Side]   # None: nothing measured reads it (the last position)
    reliability: Reliability
    n_chunks: int
    fan_out_chunks: int


def _num(x: float) -> Optional[float]:
    """float(x), or None for NaN / inf (JSON has no NaN)."""
    x = float(x)
    return x if math.isfinite(x) else None


def grade(rho: Optional[float]) -> Optional[str]:
    """The reliability badge: high (rho >= .8), mid (.6-.8), low (< .6); None without a rho."""
    if rho is None:
        return None
    return "high" if rho >= 0.8 else "mid" if rho >= 0.6 else "low"


class NeuronGraph:
    """One graph file, opened lazily through safetensors' memory map. Raises ValueError when the file is not a
    neuron graph (missing arrays or inconsistent shapes)."""

    def __init__(self, path: str):
        from safetensors import safe_open  # only needed when a graph is configured
        self.path = path
        self._f = safe_open(path, framework="numpy")
        self._lock = threading.Lock()
        missing = [k for k in KEYS if k not in self._f.keys()]
        if missing:
            raise ValueError(f"not a neuron graph: missing {', '.join(missing)}")
        shape = {k: tuple(self._f.get_slice(k).get_shape()) for k in KEYS}
        rows = shape["in_idx"][0] if len(shape["in_idx"]) == 2 else -1
        self.positions = shape["fan_out_chunks"][0] if len(shape["fan_out_chunks"]) == 1 else 0
        expect = {"in_val": shape["in_idx"], "out_val": shape["out_idx"], "fan_in": (rows, 2), "fan_out": (rows, 2),
                  "reliability": (rows, 3), "n_chunks": (rows,)}
        bad = [k for k, s in expect.items() if shape[k] != s]
        if rows <= 0 or len(shape["out_idx"]) != 2 or shape["out_idx"][0] != rows or bad:
            raise ValueError(f"not a neuron graph: inconsistent shapes {shape}")
        if self.positions <= 0 or rows % self.positions:
            raise ValueError(f"not a neuron graph: {rows} rows do not split into {self.positions} positions")
        self.rows = rows
        self.width = rows // self.positions
        self.k_in = shape["in_idx"][1]
        self.k_out = shape["out_idx"][1]
        self.meta = dict(self._f.metadata() or {})

    def _row(self, key: str, i: int) -> np.ndarray:
        with self._lock:
            return np.asarray(self._f.get_slice(key)[i:i + 1])[0]

    def gid(self, layer: int, n: int) -> int:
        if not (0 <= layer < self.positions and 0 <= n < self.width):
            raise IndexError(f"L{layer} N{n} is outside the graph ({self.positions} positions x {self.width})")
        return layer * self.width + n

    def _side(self, idx: np.ndarray, val: np.ndarray, fan: np.ndarray, k: int) -> Optional[Side]:
        total, net = float(fan[0]), float(fan[1])
        if not (math.isfinite(total) and total > 0):
            return None
        keep = (idx >= 0) & ~np.isnan(val)
        idx, val = idx[keep].astype(np.int64), val[keep].astype(np.float64)
        finite = np.isfinite(val)
        ceiling = float(np.abs(val[finite]).max()) if finite.any() else 0.0
        shown_val = np.where(finite, val, np.copysign(ceiling, val))
        edges: list[Edge] = [{"layer": int(g // self.width), "n": int(g % self.width), "edge": float(v),
                              "share": float(v) / total, "clipped": not bool(f)}
                             for g, v, f in zip(idx[:k], shown_val[:k], finite[:k])]
        return {"total": total, "net": net if math.isfinite(net) else 0.0, "stored": int(len(idx)),
                "shown_share": float(np.abs(shown_val[:k]).sum()) / total,
                "stored_share": float(np.abs(shown_val).sum()) / total, "edges": edges}

    def edges(self, layer: int, n: int, k: int = 12) -> NeuronEdges:
        """A neuron's k strongest inputs and outputs (each a signed share of its total), its fan totals and the
        split-half reliability of its input list."""
        i = self.gid(layer, n)
        k = max(1, int(k))
        rel = self._row("reliability", i)
        rho = _num(rel[0])
        return {"layer": layer, "n": n,
                "inputs": self._side(self._row("in_idx", i), self._row("in_val", i), self._row("fan_in", i), k),
                "outputs": self._side(self._row("out_idx", i), self._row("out_val", i), self._row("fan_out", i), k),
                "reliability": {"rho": rho, "top64_overlap": _num(rel[1]), "coverage": _num(rel[2]),
                                "grade": grade(rho)},
                "n_chunks": int(self._row("n_chunks", i)),
                "fan_out_chunks": int(self._row("fan_out_chunks", layer))}


_graph: dict = {"key": None, "graph": None}
_graph_lock = threading.Lock()


def load(path: Optional[str]) -> Optional[NeuronGraph]:
    """The graph at `path`, reopened when the file changes; None when no path is given or the file is missing or not
    a neuron graph (the explorer then shows no graph)."""
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_mtime_ns, st.st_size)
    with _graph_lock:
        if _graph["key"] != key:
            try:
                g: Optional[NeuronGraph] = NeuronGraph(path)
            except Exception as e:  # a truncated or foreign file: no graph, the rest of the explorer runs
                print(f"graph: {path} unreadable ({e}); no graph shown", file=sys.stderr)
                g = None
            _graph.update(key=key, graph=g)
        return _graph["graph"]
