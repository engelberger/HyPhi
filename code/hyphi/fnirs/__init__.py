"""
fNIRS hyperscanning support: SNIRF input, haemoglobin conversion, and windowed connectivity.

This subpackage turns an fNIRS recording into the ``(windows, nodes, nodes)`` connectivity
tensor that HyPhi's graph pipeline consumes. The optics (SNIRF reading, optical density, the
modified Beer-Lambert law, TDDR motion correction) are handled by MNE; HyPhi adds the
modality-appropriate connectivity. Import on demand (``import hyphi.fnirs``); it pulls MNE.
"""

# %% Imports
from .connectivity import windowed_correlation
from .io import haemoglobin_data, load_snirf, to_haemoglobin

__all__ = [
    "haemoglobin_data",
    "load_snirf",
    "to_haemoglobin",
    "windowed_correlation",
]
