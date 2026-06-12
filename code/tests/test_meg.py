"""Tests for the MEG module.

Checks FIF reading, the sensor-space PLV path (reusing the shared adapter), the volume-conduction
robust wPLI connectivity, and the end-to-end path from MEG to a curvature-entropy series.
"""

# %% Import
import numpy as np
import pytest
from hyphi.analyses import build_sliding_window_graphs, compute_entropy, compute_windowed_curvatures
from hyphi.meg import load_fif, meg_to_plv_graphs, windowed_wpli

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _meg_raw(n_channels=6, n_times=800, freq=10.0, fs=200.0, seed=0):
    """A tiny synthetic narrowband MEG Raw with magnetometer channels."""
    mne = pytest.importorskip("mne")
    mne.set_log_level("ERROR")
    rng = np.random.default_rng(seed)
    t = np.arange(n_times) / fs
    data = np.stack([np.cos(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi)) for _ in range(n_channels)])
    info = mne.create_info([f"MEG{i:03d}" for i in range(n_channels)], fs, ch_types="mag")
    return mne.io.RawArray(data, info)


class TestLoadFif:
    """load_fif reads a FIF file back into an MNE Raw."""

    def test_fif_roundtrip(self, tmp_path):
        """A Raw written to FIF reads back with the same channels."""
        raw = _meg_raw(n_channels=5)
        path = tmp_path / "sample_raw.fif"
        raw.save(path, overwrite=True)
        loaded = load_fif(path)
        assert loaded.ch_names == raw.ch_names


class TestMEGPLV:
    """meg_to_plv_graphs reuses the shared phase-to-PLV adapter."""

    def test_returns_graphs(self):
        """A MEG recording yields one PLV graph per window with n_channels nodes."""
        graphs = meg_to_plv_graphs(_meg_raw(n_channels=6), win_size=200, win_stride=100)
        assert len(graphs) > 0
        for graph in graphs:
            assert graph.number_of_nodes() == 6


class TestWindowedWPLI:
    """windowed_wpli produces a (windows, nodes, nodes) tensor in [0, 1] with a zero diagonal."""

    def test_shape_and_properties(self):
        rng = np.random.default_rng(0)
        signal = rng.standard_normal((5, 400))
        tensor = windowed_wpli(signal, win_size=100, win_stride=50)
        n_windows = (400 - 100) // 50 + 1
        assert tensor.shape == (n_windows, 5, 5)
        for mat in tensor:
            np.testing.assert_allclose(mat, mat.T, atol=1e-10)
            np.testing.assert_allclose(np.diag(mat), 0.0, atol=1e-12)
            assert np.all(mat >= 0.0) and np.all(mat <= 1.0 + 1e-9)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="n_channels, n_times"):
            windowed_wpli(np.zeros((2, 3, 4)), win_size=2, win_stride=1)


class TestEndToEnd:
    """A MEG wPLI series flows to a finite curvature-entropy time series."""

    def test_wpli_to_curvature_entropy(self):
        """wPLI connectivity -> graphs -> Forman curvature -> entropy, all finite."""
        rng = np.random.default_rng(1)
        signal = rng.standard_normal((8, 500))
        tensor = windowed_wpli(signal, win_size=100, win_stride=50)
        graphs = build_sliding_window_graphs(tensor)
        entropies = compute_entropy(compute_windowed_curvatures(graphs))
        assert len(entropies) == len(graphs)
        assert np.all(np.isfinite(entropies))


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
