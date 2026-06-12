"""
Adapter from HyPyP / MNE preprocessed data to HyPhi phase arrays.

HyPhi is complementary to HyPyP and MNE: those tools clean and band-limit the recordings, and
this adapter turns their output into the input HyPhi's geometric pipeline expects. It takes a
preprocessed, band-limited signal (an MNE ``Raw`` or ``Epochs``, a HyPyP-preprocessed object
that exposes ``get_data``, or a plain array), extracts the instantaneous phase via the analytic
signal, and returns a ``(n_channels, n_times)`` phase array that feeds
``hyphi.modeling.windowing.sliding_window_plv``.

Phase is only meaningful for a narrowband signal, so the input should already be filtered to a
single band upstream by HyPyP or MNE; this adapter does not filter.
"""

# %% Import
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.signal import hilbert

from hyphi.modeling.windowing import sliding_window_plv

if TYPE_CHECKING:
    import networkx as nx

__all__ = [
    "epochs_to_phase",
    "epochs_to_plv_graphs",
]

# An MNE Epochs array is (n_epochs, n_channels, n_times); Raw and 2D inputs are (n_channels, n_times).
_EPOCHED_NDIM = 3
_CHANNELS_TIME_NDIM = 2


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _to_data_array(inst: Any, picks: Any = None) -> np.ndarray:
    """Resolve an MNE/HyPyP object or an array to a NumPy data array."""
    if hasattr(inst, "get_data"):  # MNE Raw / Epochs, or a HyPyP wrapper exposing the same API
        data = inst.get_data(picks=picks) if picks is not None else inst.get_data()
        return np.asarray(data)
    return np.asarray(inst)


def epochs_to_phase(inst: Any, picks: Any = None) -> np.ndarray:
    """
    Extract the instantaneous phase of a band-limited signal as a ``(n_channels, n_times)`` array.

    Parameters
    ----------
    inst : mne.Epochs, mne.io.Raw, object with ``get_data``, or np.ndarray
        A preprocessed, band-limited signal. An epoched input (3D) has its phase computed per
        epoch and the epochs concatenated along time, which avoids the boundary artifacts of
        Hilbert-transforming across epoch joins.
    picks : optional
        Channel selection, forwarded to ``get_data`` for MNE/HyPyP objects; ignored for arrays.

    Returns
    -------
    np.ndarray
        Instantaneous phase in radians, shape ``(n_channels, n_times)``, values in (-pi, pi].

    """
    data = _to_data_array(inst, picks=picks)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.ndim == _EPOCHED_NDIM:
        per_epoch = [np.angle(hilbert(data[e], axis=-1)) for e in range(data.shape[0])]
        return np.concatenate(per_epoch, axis=-1)
    if data.ndim == _CHANNELS_TIME_NDIM:
        return np.angle(hilbert(data, axis=-1))
    msg = f"Expected 1D, 2D, or 3D data, got shape {data.shape}"
    raise ValueError(msg)


def epochs_to_plv_graphs(inst: Any, win_size: int, win_stride: int, picks: Any = None) -> list[nx.Graph]:
    """
    Run the full adapter path: preprocessed signal to a series of sliding-window PLV graphs.

    This is the one-call plug-and-play path for users who want "preprocessed data to HyPhi
    graphs" without handling the phase array themselves.

    Parameters
    ----------
    inst : mne.Epochs, mne.io.Raw, object with ``get_data``, or np.ndarray
        A preprocessed, band-limited signal.
    win_size : int
        Number of time steps per window.
    win_stride : int
        Stride between consecutive windows.
    picks : optional
        Channel selection forwarded to the adapter.

    Returns
    -------
    list[nx.Graph]
        One weighted PLV graph per window.

    """
    phases = epochs_to_phase(inst, picks=picks)
    return sliding_window_plv(phases, win_size, win_stride)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
