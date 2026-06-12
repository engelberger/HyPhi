"""
MEG input, built on MNE core.

MEG, like EEG, records fast neuronal oscillations, so the phase-based connectivity machinery
(Hilbert phase to PLV) transfers directly; the MEG specifics are the FIF format and a far
larger sensor count, with an optional projection to source space. This module reads FIF
recordings (via MNE) and leaves the phase-to-graph path to the shared preprocessing adapter.

Source-space node definitions (projecting sensor data through an inverse operator to cortical
sources) are supported by MNE's inverse pipeline and require a forward model and an inverse
operator built from the subject anatomy; once you have source-estimate time courses, feed them
to the connectivity functions exactly like sensor channels. This module provides the sensor-space
path; the source-space hook is documented rather than reimplemented, since the optics belong to MNE.
"""

# %% Import
from __future__ import annotations

from typing import Any

from mne.io import read_raw_fif

__all__ = [
    "load_fif",
]


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def load_fif(path: Any, preload: bool = True) -> Any:
    """
    Read a FIF MEG recording into an MNE ``Raw`` object.

    Parameters
    ----------
    path : str or path-like
        Path to a ``.fif`` recording.
    preload : bool
        If True, load the data into memory.

    Returns
    -------
    mne.io.Raw
        The MEG recording. Band-limit it (filter to a narrow band) before extracting phase,
        since phase is only meaningful for a narrowband signal.

    """
    return read_raw_fif(path, preload=preload)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
