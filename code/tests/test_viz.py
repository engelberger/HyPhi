"""Tests for the viz package.

Render every figure helper on a headless backend and check it returns a Matplotlib Figure (or
animation) and that the curvature normalisation is symmetric and does not clip out-of-range
curvature, the bug the centralised colormap fixes.
"""

# %% Import
import matplotlib

matplotlib.use("Agg")  # headless backend, no display required

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from hyphi.modeling.graph_curvatures import compute_frc  # noqa: E402
from hyphi.viz import (  # noqa: E402
    animate_curvature_entropy,
    curvature_norm,
    plot_curvature_entropy,
    plot_dyadic_graph,
    plot_node_topography,
)

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestCurvatureNorm:
    """The shared curvature normalisation is symmetric about 0 and does not clip."""

    def test_symmetric_and_no_clip(self):
        """Limits are symmetric and cover the most extreme value, even beyond [-1, 1]."""
        norm = curvature_norm([-4.0, -1.0, 0.0, 2.0, 5.0])
        assert norm.vcenter == 0.0
        assert norm.vmin == -5.0
        assert norm.vmax == 5.0

    def test_all_zero_falls_back_to_unit(self):
        """An all-zero (or empty) input falls back to a unit range rather than erroring."""
        norm = curvature_norm([0.0, 0.0, 0.0])
        assert norm.vmin == -1.0
        assert norm.vmax == 1.0


class TestFigures:
    """Every plotting function returns a Figure on a headless backend."""

    def test_plot_dyadic_graph(self):
        """A dyadic graph with curvature renders to a Figure."""
        graph = compute_frc(nx.complete_graph(6), method="1d")
        fig = plot_dyadic_graph(graph, split=3)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_node_topography(self):
        """Per-node values at 2D positions render to a Figure."""
        rng = np.random.default_rng(0)
        fig = plot_node_topography(rng.standard_normal(8), rng.standard_normal((8, 2)))
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_node_topography_rejects_bad_positions(self):
        """Mismatched positions raise a clear error."""
        with pytest.raises(ValueError, match="positions must be"):
            plot_node_topography(np.zeros(4), np.zeros((3, 2)))

    def test_plot_curvature_entropy(self):
        """An entropy series renders to a Figure."""
        fig = plot_curvature_entropy(np.array([0.1, 0.5, 0.3, 0.8]))
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestAnimation:
    """The animation returns a FuncAnimation that renders frames without error."""

    def test_animate_curvature_entropy(self):
        """The animation object is a FuncAnimation and a frame can be drawn."""
        entropy = np.array([0.1, 0.4, 0.2, 0.9, 0.5])
        anim = animate_curvature_entropy(entropy)
        assert isinstance(anim, FuncAnimation)
        # Drawing the canvas exercises the frame-update function.
        anim._fig.canvas.draw()
        plt.close(anim._fig)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
