"""
MEG hyperscanning support: FIF input and phase-based connectivity.

MEG records fast oscillations like EEG, so the phase-to-PLV path transfers directly through the
shared preprocessing adapter; this subpackage adds FIF input (via MNE) and the volume-conduction
robust weighted phase-lag index. Source-space node definitions are supported through MNE's
inverse pipeline (feed source-estimate time courses to the connectivity functions like sensor
channels). Import on demand (``import hyphi.meg``).
"""

# %% Imports
from .connectivity import meg_to_plv_graphs, windowed_wpli
from .io import load_fif

__all__ = [
    "load_fif",
    "meg_to_plv_graphs",
    "windowed_wpli",
]
