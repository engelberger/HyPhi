"""
Time-series plots and animation of the geometric metrics.

The curvature-distribution entropy (and other per-window scalars) describe how a dyadic graph
reshapes itself across an interaction. These helpers render that series as a static plot and as
an animation that reveals it window by window. Both return standard Matplotlib objects.
"""

# %% Import
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

__all__ = [
    "animate_curvature_entropy",
    "plot_curvature_entropy",
]


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def plot_curvature_entropy(
    entropy: np.ndarray,
    x: np.ndarray | None = None,
    ax: Any = None,
    ylabel: str = "curvature entropy (nats)",
) -> Any:
    """
    Plot a curvature-entropy (or any per-window scalar) series.

    Parameters
    ----------
    entropy : array-like
        One value per window.
    x : array-like, optional
        The horizontal axis (window index by default).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    ylabel : str
        Label for the value axis.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the series.

    """
    entropy = np.asarray(entropy, dtype=float)
    x = np.asarray(x) if x is not None else np.arange(len(entropy))
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    ax.plot(x, entropy, marker="o", markersize=3)
    ax.set_xlabel("window")
    ax.set_ylabel(ylabel)
    ax.set_title("Curvature-distribution entropy over time")
    return fig


def animate_curvature_entropy(
    entropy: np.ndarray,
    x: np.ndarray | None = None,
    interval: int = 100,
) -> FuncAnimation:
    """
    Animate a curvature-entropy series being traced window by window.

    Parameters
    ----------
    entropy : array-like
        One value per window.
    x : array-like, optional
        The horizontal axis (window index by default).
    interval : int
        Delay between frames in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation; save it with ``.save(...)`` or embed it in a notebook.

    """
    entropy = np.asarray(entropy, dtype=float)
    x = np.asarray(x) if x is not None else np.arange(len(entropy))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(float(np.min(x)), float(np.max(x)) if len(x) > 1 else float(np.min(x)) + 1.0)
    spread = float(np.nanmax(entropy) - np.nanmin(entropy)) or 1.0
    ax.set_ylim(float(np.nanmin(entropy)) - 0.05 * spread, float(np.nanmax(entropy)) + 0.05 * spread)
    ax.set_xlabel("window")
    ax.set_ylabel("curvature entropy (nats)")
    ax.set_title("Curvature-distribution entropy over time")
    (line,) = ax.plot([], [], marker="o", markersize=3)

    def _update(frame: int) -> tuple[Any, ...]:
        line.set_data(x[: frame + 1], entropy[: frame + 1])
        return (line,)

    return FuncAnimation(fig, _update, frames=len(entropy), interval=interval, blit=True)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
