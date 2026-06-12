"""
Connectivity for fMRI ROI time series.

fMRI BOLD is very slow and is analysed on parcellated ROI time series, so connectivity is
correlation based, not phase based. This module provides windowed Pearson correlation (the
simple default) and windowed partial correlation (which controls for the other ROIs via the
precision matrix), each producing the ``(windows, ROIs, ROIs)`` tensor that
:func:`hyphi.analyses.build_sliding_window_graphs` consumes.

Which of the two is canonical for a given fMRI analysis (Pearson is simpler and more stable;
partial correlation is sparser and more specific) is a methodological choice for the maintainers.
"""

# %% Import
from __future__ import annotations

import numpy as np

__all__ = [
    "roi_correlation",
    "roi_partial_correlation",
]

_ROI_TIME_NDIM = 2  # connectivity expects (n_rois, n_times)


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _check_2d(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim != _ROI_TIME_NDIM:
        msg = f"Expected (n_rois, n_times) data, got shape {arr.shape}"
        raise ValueError(msg)
    return arr


def _windows(n_times: int, win_size: int, win_stride: int) -> range:
    return range(0, n_times - win_size + 1, win_stride)


def roi_correlation(roi_timeseries: np.ndarray, win_size: int, win_stride: int) -> np.ndarray:
    """
    Windowed Pearson-correlation connectivity for ROI time series.

    Parameters
    ----------
    roi_timeseries : np.ndarray
        ROI time series of shape ``(n_rois, n_times)``.
    win_size : int
        Number of time steps per window.
    win_stride : int
        Stride between consecutive windows.

    Returns
    -------
    np.ndarray
        Connectivity tensor ``(n_windows, n_rois, n_rois)``: one symmetric correlation matrix
        per window, unit diagonal, values in [-1, 1].

    """
    data = _check_2d(roi_timeseries)
    mats = [np.corrcoef(data[:, s : s + win_size]) for s in _windows(data.shape[1], win_size, win_stride)]
    return np.stack(mats, axis=0)


def roi_partial_correlation(roi_timeseries: np.ndarray, win_size: int, win_stride: int) -> np.ndarray:
    """
    Windowed partial-correlation connectivity for ROI time series.

    Partial correlation between two ROIs controls for all the others, computed from the
    precision (inverse covariance) matrix: ``pcorr_ij = -P_ij / sqrt(P_ii * P_jj)``. The
    pseudo-inverse is used for numerical robustness on short or collinear windows.

    Parameters
    ----------
    roi_timeseries : np.ndarray
        ROI time series of shape ``(n_rois, n_times)``.
    win_size : int
        Number of time steps per window.
    win_stride : int
        Stride between consecutive windows.

    Returns
    -------
    np.ndarray
        Connectivity tensor ``(n_windows, n_rois, n_rois)``: one symmetric partial-correlation
        matrix per window, unit diagonal.

    """
    data = _check_2d(roi_timeseries)
    mats = []
    for s in _windows(data.shape[1], win_size, win_stride):
        window = data[:, s : s + win_size]
        precision = np.linalg.pinv(np.cov(window))
        scale = np.sqrt(np.diag(precision))
        pcorr = -precision / np.outer(scale, scale)
        np.fill_diagonal(pcorr, 1.0)
        mats.append(pcorr)
    return np.stack(mats, axis=0)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
