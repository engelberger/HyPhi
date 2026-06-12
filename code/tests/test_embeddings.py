"""Tests for the hyperbolic embeddings module.

Checks the Poincare-ball geometry utilities and that a hyperbolic embedding of simulated
multichannel data lands inside the open ball with the right shape.
"""

# %% Import
import numpy as np
import pytest
from hyphi.embeddings import (
    hyperbolic_embedding,
    poincare_distance,
    poincare_distance_matrix,
    poincare_exp0,
)

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _simulated_signal(n_channels=8, n_times=400, seed=0):
    """Simulated multichannel data: a few correlated groups, as a stand-in for Kuramoto output."""
    rng = np.random.default_rng(seed)
    groups = [rng.standard_normal(n_times) for _ in range(3)]
    channels = []
    for c in range(n_channels):
        base = groups[c % 3]
        channels.append(base + 0.5 * rng.standard_normal(n_times))
    return np.stack(channels)


class TestPoincareGeometry:
    """The Poincare-ball utilities behave as a hyperbolic metric should."""

    def test_exp0_maps_into_open_ball(self):
        """The exponential map sends any Euclidean vector strictly inside the unit ball."""
        rng = np.random.default_rng(0)
        euclidean = rng.standard_normal((50, 2)) * 10.0  # large vectors
        ball = poincare_exp0(euclidean)
        radii = np.linalg.norm(ball, axis=-1)
        assert np.all(radii < 1.0)

    def test_distance_is_a_metric(self):
        """Poincare distance is non-negative, zero on the diagonal, and symmetric."""
        rng = np.random.default_rng(1)
        points = poincare_exp0(rng.standard_normal((6, 2)))
        matrix = poincare_distance_matrix(points)
        assert np.all(matrix >= 0.0)
        np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-9)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)

    def test_distance_grows_toward_the_boundary(self):
        """Two points near the boundary are hyperbolically far apart."""
        center = np.array([0.0, 0.0])
        near_edge = np.array([0.9, 0.0])
        far_edge = np.array([-0.9, 0.0])
        assert poincare_distance(near_edge, far_edge) > poincare_distance(center, near_edge)


class TestHyperbolicEmbedding:
    """A hyperbolic embedding of simulated data has the right shape and lands in the ball."""

    def test_embedding_shape_and_in_ball(self):
        """The embedding has one point per channel, in the open Poincare ball."""
        signal = _simulated_signal(n_channels=8)
        embedding = hyperbolic_embedding(signal, n_components=2)
        assert embedding.shape == (8, 2)
        assert np.all(np.linalg.norm(embedding, axis=-1) < 1.0)

    def test_reproducible(self):
        """A fixed random_state gives the same embedding."""
        signal = _simulated_signal()
        a = hyperbolic_embedding(signal, random_state=3)
        b = hyperbolic_embedding(signal, random_state=3)
        np.testing.assert_allclose(a, b)

    def test_unknown_metric_raises(self):
        """An unknown dissimilarity metric raises a clear error."""
        with pytest.raises(ValueError, match="Unknown metric"):
            hyperbolic_embedding(_simulated_signal(), metric="not_a_metric")


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
