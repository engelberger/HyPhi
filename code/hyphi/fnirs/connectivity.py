"""
Connectivity for fNIRS haemoglobin time series.

fNIRS measures slow haemodynamics (HbO/HbR), not band-limited oscillations, so the phase-based
PLV used for EEG and MEG does not transfer. The modality-appropriate connectivity is
correlation based (or wavelet-transform coherence). This module provides windowed Pearson
correlation, producing the ``(windows, nodes, nodes)`` tensor that
:func:`hyphi.analyses.build_sliding_window_graphs` consumes.

Wavelet-transform coherence (WTC) is the other standard fNIRS connectivity choice; it needs a
continuous wavelet transform (an extra dependency) and is left as a follow-up. Which of the two
is canonical for fNIRS is a methodological choice for the maintainers.
"""

# %% Import
from __future__ import annotations

import numpy as np

__all__ = [
    "windowed_correlation",
]

_CHANNELS_TIME_NDIM = 2  # connectivity expects (n_channels, n_times)


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def windowed_correlation(data: np.ndarray, win_size: int, win_stride: int) -> np.ndarray:
    """
    Windowed Pearson-correlation connectivity for a multichannel time series.

    Parameters
    ----------
    data : np.ndarray
        Haemoglobin time series of shape ``(n_channels, n_times)``.
    win_size : int
        Number of time steps per window.
    win_stride : int
        Stride between consecutive windows.

    Returns
    -------
    np.ndarray
        Connectivity tensor of shape ``(n_windows, n_channels, n_channels)``: one symmetric
        correlation matrix per window, with unit diagonal and values in [-1, 1].

    """
    data = np.asarray(data, dtype=float)
    if data.ndim != _CHANNELS_TIME_NDIM:
        msg = f"Expected (n_channels, n_times) data, got shape {data.shape}"
        raise ValueError(msg)

    n_times = data.shape[1]
    mats = []
    for start in range(0, n_times - win_size + 1, win_stride):
        window = data[:, start : start + win_size]
        mats.append(np.corrcoef(window))
    return np.stack(mats, axis=0)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
