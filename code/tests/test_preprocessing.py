"""Tests for the HyPyP / MNE preprocessing adapter.

Checks that a preprocessed, band-limited signal (a plain array or an MNE Epochs object) flows
through epochs_to_phase into a (n_channels, n_times) phase array, and through
epochs_to_plv_graphs into a sliding-window PLV graph series.
"""

# %% Import
import networkx as nx
import numpy as np
import pytest
from hyphi.preprocessing import epochs_to_phase, epochs_to_plv_graphs

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _narrowband_signal(n_channels=4, n_times=512, freq=10.0, fs=128.0, seed=0):
    """A band-limited multichannel signal: sinusoids at one frequency with random phase offsets."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_times) / fs
    offsets = rng.uniform(0, 2 * np.pi, size=n_channels)
    return np.stack([np.cos(2 * np.pi * freq * t + off) for off in offsets])


class TestEpochsToPhase:
    """epochs_to_phase returns a (n_channels, n_times) phase array within (-pi, pi]."""

    def test_array_2d_shape_and_range(self):
        """A 2D array yields a phase array of the same shape, in radians."""
        signal = _narrowband_signal()
        phase = epochs_to_phase(signal)
        assert phase.shape == signal.shape
        assert np.all(phase >= -np.pi) and np.all(phase <= np.pi)

    def test_array_3d_epochs_concatenated(self):
        """A 3D (epochs) array concatenates per-epoch phase along time."""
        epoched = np.stack([_narrowband_signal(seed=s) for s in range(3)])  # (3, 4, 512)
        phase = epochs_to_phase(epoched)
        assert phase.shape == (4, 3 * 512)

    def test_recovers_known_frequency(self):
        """The instantaneous phase of a 10 Hz sinusoid advances at about 2*pi*10 rad/s."""
        fs = 128.0
        signal = _narrowband_signal(n_channels=1, n_times=1024, freq=10.0, fs=fs)
        phase = epochs_to_phase(signal)[0]
        # Median instantaneous angular frequency, ignoring the +-2pi wraps.
        dphi = np.diff(np.unwrap(phase)) * fs
        assert np.median(dphi) == pytest.approx(2 * np.pi * 10.0, rel=0.05)


class TestMNEAdapter:
    """The adapter accepts an MNE Epochs object and drives the full PLV-graph path."""

    def _epochs(self):
        mne = pytest.importorskip("mne")
        mne.set_log_level("ERROR")
        data = np.stack([_narrowband_signal(n_channels=6, n_times=256, seed=s) for s in range(2)])
        info = mne.create_info(ch_names=[f"c{i}" for i in range(6)], sfreq=128.0, ch_types="eeg")
        return mne.EpochsArray(data, info)

    def test_epochs_to_phase(self):
        """An MNE EpochsArray yields a (n_channels, n_epochs*n_times) phase array."""
        phase = epochs_to_phase(self._epochs())
        assert phase.shape == (6, 2 * 256)
        assert np.all(np.isfinite(phase))

    def test_epochs_to_plv_graphs(self):
        """The plug-and-play path returns one PLV graph per window, each with n_channels nodes."""
        graphs = epochs_to_plv_graphs(self._epochs(), win_size=128, win_stride=64)
        assert len(graphs) > 0
        for graph in graphs:
            assert isinstance(graph, nx.Graph)
            assert graph.number_of_nodes() == 6


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
