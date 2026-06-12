"""Tests for the end-to-end pipeline entry point.

Checks the demo connectivity, the run_pipeline function, and that the command-line entry point
writes the entropy series, quantiles, and a figure.
"""

# %% Import
import matplotlib

matplotlib.use("Agg")  # headless backend for the figure the CLI saves

import numpy as np  # noqa: E402
from hyphi.main import demo_connectivity, main, run_pipeline  # noqa: E402

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestRunPipeline:
    """run_pipeline takes connectivity to a curvature-entropy series."""

    def test_demo_connectivity_shape(self):
        """The demo connectivity is a (windows, nodes, nodes) tensor."""
        connectivity = demo_connectivity(n_nodes=15, n_windows=6)
        assert connectivity.shape == (6, 15, 15)

    def test_pipeline_outputs(self):
        """run_pipeline returns aligned graphs, a finite entropy trace, and quantiles."""
        connectivity = demo_connectivity(n_nodes=20, n_windows=8)
        results = run_pipeline(connectivity)
        assert set(results) == {"graphs", "entropy", "quantiles"}
        assert len(results["graphs"]) == 8
        assert results["entropy"].shape == (8,)
        assert results["quantiles"].shape == (8, 5)
        assert np.all(np.isfinite(results["entropy"]))

    def test_pipeline_accepts_single_window(self):
        """A single 2D connectivity matrix is treated as one window."""
        connectivity = demo_connectivity(n_nodes=12, n_windows=1)[0]  # (12, 12)
        results = run_pipeline(connectivity)
        assert len(results["graphs"]) == 1


class TestMainCLI:
    """The command-line entry point runs and writes its outputs."""

    def test_writes_outputs(self, tmp_path):
        """main writes entropy.csv, quantiles.csv, and entropy.png to the output directory."""
        out = tmp_path / "run"
        main(["--output", str(out), "--nodes", "15", "--windows", "5"])
        assert (out / "entropy.csv").exists()
        assert (out / "quantiles.csv").exists()
        assert (out / "entropy.png").exists()
        # The entropy CSV has one value per window.
        entropy = np.loadtxt(out / "entropy.csv", delimiter=",")
        assert entropy.shape == (5,)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
