"""
Tests for the accelerator backends (hyphi.backends).

Three kinds of check, mirroring the project testing philosophy:

- known-answer: the unweighted 1d Forman curvature has a closed form
  ``F(u, v) = 4 - deg(u) - deg(v)`` (star and complete graphs), asserted
  independently of the reference library;
- parity: every available backend agrees with the GraphRicciCurvature reference,
  at float64 tolerance for the float64 backends and float32 tolerance for the
  Metal/MLX backend;
- contract: registry resolution, capability probes that never raise, graph_io
  round-trip, degenerate (empty) input, and transparent fallback for a method a
  backend does not implement.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from hyphi import backends
from hyphi.backends.graph_io import graph_to_arrays


def _weighted_ws(n=120, k=6, p=0.3, seed=0):
    g = nx.watts_strogatz_graph(n, k, p, seed=seed)
    rng = np.random.default_rng(seed)
    for u, v in g.edges():
        g[u][v]["weight"] = float(rng.uniform(0.1, 1.0))
    return g


# --- known-answer (independent of the reference library) ---------------------


def test_known_answer_star_unweighted():
    n_leaves = 7
    g = nx.star_graph(n_leaves)  # node 0 is the hub, 1..n_leaves are leaves
    curv = backends.forman_curvature(g, "1d", backend="numpy")
    # every edge is hub(deg n_leaves)-leaf(deg 1): F = 4 - n_leaves - 1
    assert np.allclose(curv, 4 - n_leaves - 1)


def test_known_answer_complete_unweighted():
    g = nx.complete_graph(5)  # every node degree 4
    curv = backends.forman_curvature(g, "1d", backend="numpy")
    assert np.allclose(curv, 4 - 4 - 4)


# --- parity vs the GraphRicciCurvature reference -----------------------------


@pytest.mark.parametrize("method", ["1d", "augmented"])
def test_numpy_parity_with_reference(method):
    g = _weighted_ws()
    ref = backends.forman_curvature(g, method, backend="networkx")
    got = backends.forman_curvature(g, method, backend="numpy")
    assert np.max(np.abs(got - ref)) < 1e-10


def test_mlx_parity_with_reference_float32():
    if "mlx" not in backends.available_backends():
        pytest.skip("MLX/Metal backend not available on this machine")
    g = _weighted_ws()
    ref = backends.forman_curvature(g, "1d", backend="networkx")
    got = backends.forman_curvature(g, "1d", backend="mlx")
    # float32 GPU path: bounded by float32 precision, not float64
    denom = np.maximum(np.abs(ref), 1e-9)
    assert np.max(np.abs((got - ref) / denom)) < 1e-4


def test_cupy_parity_when_available():
    if "cupy" not in backends.available_backends():
        pytest.skip("CuPy/CUDA backend not available on this machine")
    g = _weighted_ws()
    ref = backends.forman_curvature(g, "1d", backend="networkx")
    got = backends.forman_curvature(g, "1d", backend="cupy")
    assert np.max(np.abs(got - ref)) < 1e-10


# --- contract ----------------------------------------------------------------


def test_registry_and_default():
    assert "numpy" in backends.available_backends()
    assert "networkx" in backends.available_backends()
    assert backends.get_backend(None).name == "numpy"
    assert backends.get_backend("numpy").name == "numpy"


def test_auto_selects_available_backend():
    assert backends.get_backend("auto").name in backends.available_backends()


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        backends.get_backend("does-not-exist")


def test_capability_probes_never_raise():
    caps = backends.detect()
    assert caps.cpu_count >= 1
    assert isinstance(backends.install_hint(), str)
    for cls in (backends.NumpyBackend, backends.CupyBackend, backends.MlxBackend, backends.NativeExtBackend):
        assert isinstance(cls.is_available(), bool)


def test_graph_io_roundtrip_order():
    g = _weighted_ws(n=40, k=4, seed=2)
    arrays = graph_to_arrays(g)
    assert arrays.n_edges == g.number_of_edges()
    assert arrays.n_nodes == g.number_of_nodes()
    edge_weights = [d["weight"] for _, _, d in g.edges(data=True)]
    assert np.allclose(arrays.we, edge_weights)


def test_empty_graph_returns_empty():
    g = nx.empty_graph(5)  # nodes, no edges
    curv = backends.forman_curvature(g, "1d", backend="numpy")
    assert curv.shape == (0,)


def test_method_fallback_is_transparent():
    # MLX implements 1d only; augmented must fall back to a CPU backend and match
    if "mlx" not in backends.available_backends():
        pytest.skip("MLX/Metal backend not available on this machine")
    g = _weighted_ws()
    ref = backends.forman_curvature(g, "augmented", backend="networkx")
    got = backends.forman_curvature(g, "augmented", backend="mlx")
    assert np.max(np.abs(got - ref)) < 1e-10


def test_series_and_annotate():
    g = _weighted_ws(n=60, seed=5)
    series = [g, g, g]
    arrays_list = backends.forman_curvature(series, "1d", backend="numpy")
    assert isinstance(arrays_list, list) and len(arrays_list) == 3
    annotated = backends.forman_curvature(series, "1d", backend="numpy", annotate=True)
    first_edge = next(iter(annotated[0].edges(data=True)))
    assert "formanCurvature" in first_edge[2]
