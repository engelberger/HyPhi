"""
Dyadic graph layout for hyperscanning networks.

Renders an inter-brain network with the two brains side by side: brain A on the left, brain B on
the right, intra-brain edges within each half and inter-brain edges spanning the gap. Edges are
coloured by curvature when the graph carries it, using the shared diverging curvature colormap.
"""

# %% Import
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .style import CURVATURE_CMAP, curvature_norm

__all__ = [
    "plot_dyadic_graph",
]


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def plot_dyadic_graph(
    graph: nx.Graph,
    split: int | None = None,
    curvature: str = "formanCurvature",
    node_size: int = 120,
    ax: Any = None,
) -> Any:
    """
    Draw a dyadic inter-brain graph with the two brains side by side.

    Parameters
    ----------
    graph : nx.Graph
        Inter-brain graph. Nodes are ordered so the first ``split`` belong to brain A and the
        rest to brain B.
    split : int, optional
        Number of brain-A nodes; defaults to half the nodes.
    curvature : str
        Edge attribute used to colour edges, if present.
    node_size : int
        Node marker size.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the dyadic layout.

    """
    nodes = sorted(graph.nodes())
    n = len(nodes)
    if split is None:
        split = n // 2
    a_nodes = nodes[:split]
    b_nodes = nodes[split:]

    # Brain A in a left column, brain B in a right column, each spread vertically.
    pos: dict[Any, tuple[float, float]] = {}
    for i, node in enumerate(a_nodes):
        pos[node] = (0.0, -float(i))
    for i, node in enumerate(b_nodes):
        pos[node] = (1.0, -float(i))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    else:
        fig = ax.figure

    curvatures = nx.get_edge_attributes(graph, curvature)
    if curvatures:
        edges = list(curvatures.keys())
        values = np.asarray(list(curvatures.values()), dtype=float)
        norm = curvature_norm(values)
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=edges,
            edge_color=values,
            edge_cmap=plt.get_cmap(CURVATURE_CMAP),
            edge_vmin=norm.vmin,
            edge_vmax=norm.vmax,
            ax=ax,
        )
    else:
        nx.draw_networkx_edges(graph, pos, alpha=0.4, ax=ax)

    nx.draw_networkx_nodes(graph, pos, nodelist=a_nodes, node_color="tab:blue", node_size=node_size, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=b_nodes, node_color="tab:orange", node_size=node_size, ax=ax)
    ax.set_title("Dyadic inter-brain network")
    ax.set_axis_off()
    return fig


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
