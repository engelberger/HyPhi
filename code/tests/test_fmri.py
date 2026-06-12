"""Tests for the fMRI module.

Checks the ROI correlation and partial-correlation connectivity, the end-to-end path from ROI
time series to a curvature-entropy series, and that parcellation raises a clear error when
nilearn is absent.
"""

# %% Import
import importlib.util

import numpy as np
import pytest
from hyphi.analyses import build_sliding_window_graphs, compute_entropy, compute_windowed_curvatures
from hyphi.fmri import parcellate, roi_correlation, roi_partial_correlation

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _roi_timeseries(n_rois=8, n_times=600, seed=0):
    """A synthetic ROI time series with some shared structure between ROIs."""
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal(n_times)
    return np.stack([0.5 * shared + rng.standard_normal(n_times) for _ in range(n_rois)])


class TestROIConnectivity:
    """Both connectivity functions produce a (windows, ROIs, ROIs) tensor."""

    @pytest.mark.parametrize("fn", [roi_correlation, roi_partial_correlation])
    def test_shape_and_symmetry(self, fn):
        """Each window is a symmetric matrix with unit diagonal."""
        data = _roi_timeseries(n_rois=6, n_times=400)
        tensor = fn(data, win_size=120, win_stride=60)
        n_windows = (400 - 120) // 60 + 1
        assert tensor.shape == (n_windows, 6, 6)
        for mat in tensor:
            np.testing.assert_allclose(mat, mat.T, atol=1e-8)
            np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-8)

    def test_pearson_in_range(self):
        """Pearson correlation stays within [-1, 1]."""
        tensor = roi_correlation(_roi_timeseries(), win_size=150, win_stride=75)
        assert np.all(tensor >= -1.0 - 1e-9) and np.all(tensor <= 1.0 + 1e-9)

    @pytest.mark.parametrize("fn", [roi_correlation, roi_partial_correlation])
    def test_rejects_non_2d(self, fn):
        """A non-2D input is rejected with a clear error."""
        with pytest.raises(ValueError, match="n_rois, n_times"):
            fn(np.zeros((2, 3, 4)), win_size=2, win_stride=1)


class TestEndToEnd:
    """An fMRI ROI series flows to a finite curvature-entropy time series."""

    @pytest.mark.parametrize("fn", [roi_correlation, roi_partial_correlation])
    def test_roi_to_curvature_entropy(self, fn):
        """connectivity -> graphs -> Forman curvature -> entropy, all finite."""
        tensor = fn(_roi_timeseries(n_rois=10, n_times=500), win_size=100, win_stride=50)
        graphs = build_sliding_window_graphs(tensor)
        assert len(graphs) == tensor.shape[0]
        entropies = compute_entropy(compute_windowed_curvatures(graphs))
        assert len(entropies) == len(graphs)
        assert np.all(np.isfinite(entropies))


class TestParcellation:
    """Parcellation delegates to nilearn and errors clearly when it is absent."""

    def test_requires_nilearn_when_absent(self):
        """Without nilearn, parcellate raises an informative ImportError."""
        if importlib.util.find_spec("nilearn") is not None:
            pytest.skip("nilearn is installed; the missing-dependency path is not exercised")
        with pytest.raises(ImportError, match="requires nilearn"):
            parcellate(func_img=None, atlas_img=None)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
