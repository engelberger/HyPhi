"""
Array boundary between NetworkX graphs and the accelerator kernels.

The curvature backends do not operate on NetworkX objects; they operate on a
flat structure-of-arrays (SoA) representation of an undirected weighted graph,
which is what GPUs and vectorized CPU code want. This module is the single,
explicit conversion point.

A graph with ``N`` nodes and ``E`` edges is represented by

- ``n_nodes`` : int
- ``ei``, ``ej`` : int64 arrays of length ``E``, the endpoints of each edge
  (each undirected edge appears exactly once, ``ei < ej`` is not required)
- ``we`` : float64 array of length ``E``, the edge weights
- ``node_order`` : the node labels in index order, so curvature values can be
  mapped back onto the original NetworkX edges in ``G.edges()`` order

Node weights are assumed to be 1.0 (the ``GraphRicciCurvature`` default and the
universal convention for HyPhi PLV/CCORR graphs). Backends that receive a graph
with non-unit node weights must say so; see :func:`graph_to_arrays`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

__all__ = ["GraphArrays", "graph_to_arrays"]


@dataclass(frozen=True)
class GraphArrays:
    """
    Flat structure-of-arrays view of one undirected weighted graph.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    ei, ej : numpy.ndarray
        ``int64`` endpoint indices for each edge, length ``E``.
    we : numpy.ndarray
        ``float64`` edge weights, length ``E``.
    node_order : list
        Node labels in index order (``index -> original label``), so a
        per-edge curvature array lines up with ``G.edges()`` iteration order.

    """

    n_nodes: int
    ei: np.ndarray
    ej: np.ndarray
    we: np.ndarray
    node_order: list

    @property
    def n_edges(self) -> int:
        """Number of edges ``E``."""
        return int(self.we.shape[0])


def graph_to_arrays(graph: nx.Graph, weight: str = "weight") -> GraphArrays:
    """
    Convert a NetworkX graph to its :class:`GraphArrays` SoA representation.

    Parameters
    ----------
    graph : networkx.Graph
        Undirected weighted graph. Missing edge weights default to 1.0.
    weight : str
        Edge attribute holding the weight (default ``"weight"``).

    Returns
    -------
    GraphArrays
        Flat arrays in ``graph.edges()`` order, ready for a kernel.

    Notes
    -----
    Edges are emitted in exactly ``graph.edges()`` order so that a returned
    per-edge curvature array can be zipped back onto the graph without a
    reordering step. Self-loops are dropped (a self-loop is not a 1-simplex
    edge in the Forman sense and the library skips it).

    """
    node_order = list(graph.nodes())
    index = {label: i for i, label in enumerate(node_order)}

    ei_list: list[int] = []
    ej_list: list[int] = []
    we_list: list[float] = []
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        ei_list.append(index[u])
        ej_list.append(index[v])
        we_list.append(float(data.get(weight, 1.0)))

    ei = np.asarray(ei_list, dtype=np.int64)
    ej = np.asarray(ej_list, dtype=np.int64)
    we = np.asarray(we_list, dtype=np.float64)
    return GraphArrays(n_nodes=len(node_order), ei=ei, ej=ej, we=we, node_order=node_order)
