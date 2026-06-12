"""Smoke test for the connectome-delayed Kuramoto ground truth.

The delayed Kuramoto integrator is built on jitcdde, which compiles the system through symengine
and currently needs sympy for the symbolic simplification step. sympy is not yet a declared
dependency, so this test skips when it is absent rather than failing CI; it runs once sympy is
installed (it is pinned in the development environment used to write this).
"""

# %% Import
import numpy as np
import pytest

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _small_connectome(n=4, seed=0):
    """A tiny synthetic connectome: symmetric weights, tract lengths (m), and frequencies."""
    rng = np.random.default_rng(seed)
    w = np.abs(rng.standard_normal((n, n)))
    w = (w + w.T) / 2.0
    np.fill_diagonal(w, 0.0)
    tract = np.abs(rng.standard_normal((n, n))) * 0.05 + 0.02  # roughly 2 to 7 cm
    tract = (tract + tract.T) / 2.0
    np.fill_diagonal(tract, 0.0)
    omega = 2.0 * np.pi * (8.0 + rng.standard_normal(n))  # around 8 Hz
    return w, tract, omega


class TestDelayedKuramoto:
    """run_delayed_kuramoto produces finite phases and order parameters on a small connectome."""

    def test_smoke(self):
        """A short delayed-Kuramoto run returns aligned (times, phases, order parameters)."""
        pytest.importorskip("jitcdde")
        pytest.importorskip("symengine")
        pytest.importorskip("sympy")
        from hyphi.simulation.simulations import run_delayed_kuramoto, setup_delayed_kuramoto

        n = 4
        w, tract, omega = _small_connectome(n=n)
        solver = setup_delayed_kuramoto(w, tract, omega, noise_strength=0.0, seed=1)
        times, theta, order = run_delayed_kuramoto(solver, dt=0.01, t_max=0.5, n_osc=n, t_skip=0.1)

        assert times.shape[0] == theta.shape[0] == order.shape[0]
        assert theta.shape[1] == n
        assert np.all(np.isfinite(theta))
        assert np.all(theta >= 0.0) and np.all(theta < 2.0 * np.pi + 1e-6)
        # The order parameter is the Kuramoto synchrony measure, in [0, 1].
        assert np.all(order >= 0.0) and np.all(order <= 1.0 + 1e-9)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
