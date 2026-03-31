"""
Unified analysis toolkit for Active Shell simulations.

Provides:
    - Simulation: unified access to meshes, fields, times
    - GeometryProcessor: triangulation, COM, transforms
    - MatplotlibAnimator: lightweight animation
    - SimulationDiagnostics: blow-up checks, stability
    - SpectralTools: temporal FFTs, frequency spectra
    - CorrelationTools: spatial correlations (stubs)
    - PeakTracker: bright-spot tracking (stub)
    - LCSTools: Lagrangian coherent structures (stub)
"""

from .simulation import Simulation
from .geometry import GeometryProcessor
from .animator import MatplotlibAnimator
from .diagnostics import SimulationDiagnostics
from .spectral import SpectralTools
from .correlation import CorrelationTools
from .peaks import PeakTracker
from .lcs import LCSTools

__all__ = [
    "Simulation",
    "GeometryProcessor",
    "MatplotlibAnimator",
    "SimulationDiagnostics",
    "SpectralTools",
    "CorrelationTools",
    "PeakTracker",
    "LCSTools",
]