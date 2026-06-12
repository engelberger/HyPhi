"""
Hyperbolic embedding of multichannel signals.

Embeds a multichannel signal (for example simulated Kuramoto phases or Watts-Strogatz-derived
activity) into the Poincare ball: it builds a dissimilarity matrix between channels, finds a
low-dimensional Euclidean configuration with metric MDS, then maps that configuration into the
negatively curved Poincare ball through the exponential map at the origin.

This is the lightweight, dependency-free embedding for simulated data. The full HEEGNet model
(Li et al. 2026), with its two-stage DSMDBN moment alignment and Hyperbolic Horospherical
Sliced-Wasserstein matching, is a deep network that needs a learning framework and is out of
scope here; how far to take that integration (a replacement for the connectivity-to-graph step,
a parallel alternative geometry, or a validation layer) is an open design decision.
"""

# %% Import
from __future__ import annotations

import numpy as np
from sklearn.manifold import MDS

from .hyperbolic import poincare_exp0

__all__ = [
    "hyperbolic_embedding",
]


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def hyperbolic_embedding(
    data: np.ndarray,
    n_components: int = 2,
    metric: str = "correlation",
    random_state: int = 0,
) -> np.ndarray:
    """
    Embed channels of a multichannel signal into the Poincare ball.

    Parameters
    ----------
    data : np.ndarray
        Signal of shape ``(n_channels, n_times)``.
    n_components : int
        Dimension of the embedding (2 for the Poincare disk).
    metric : {"correlation", "euclidean"}
        Channel dissimilarity. ``"correlation"`` uses ``1 - |corr|`` so correlated channels are
        close; ``"euclidean"`` runs MDS on the raw channel vectors.
    random_state : int
        Seed for the MDS initialisation, for reproducibility.

    Returns
    -------
    np.ndarray
        Embedding of shape ``(n_channels, n_components)``, with every point inside the open
        Poincare ball.

    """
    data = np.asarray(data, dtype=float)
    if metric == "correlation":
        dissimilarity = 1.0 - np.abs(np.corrcoef(data))
        np.fill_diagonal(dissimilarity, 0.0)
        mds = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=random_state,
            normalized_stress="auto",
        )
        euclidean = mds.fit_transform(dissimilarity)
    elif metric == "euclidean":
        mds = MDS(n_components=n_components, random_state=random_state, normalized_stress="auto")
        euclidean = mds.fit_transform(data)
    else:
        msg = f"Unknown metric {metric!r}; choose 'correlation' or 'euclidean'"
        raise ValueError(msg)
    return poincare_exp0(euclidean)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
