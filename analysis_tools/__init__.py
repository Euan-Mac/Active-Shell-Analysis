"""
analysis_tools

Unified Lagrangian, area‑weighted analysis toolkit for deforming
surface simulations (e.g., Active Shell VTP outputs).

This package provides:

Core Objects
------------
Simulation
    Main container for VTP time‑series:
    - loads frames
    - maintains Lagrangian ID maps
    - exposes Lagrangian field & area access
    - provides high‑level extraction helpers
    - interfaces to all analysis modules

GeometryProcessor
    Geometry and surface‑calculus utilities:
    - triangulation
    - area‑weighted COM
    - instantaneous integration
    - spatial & spatiotemporal series
    - per‑point Lagrangian operators

Analysis Modules
----------------
SpectralTools
    Lagrangian per‑point PSD and area‑weighted ensemble spectra.

CorrelationTools
    Placeholder for geodesic and spatial correlation analysis.

PeakTracker
    Simple per‑frame peak detection (Lagrangian).

LCSTools
    Future module for FTLE / LCS computation from Lagrangian velocity.

Plotting Utilities
------------------
Functions for plotting:
    - field values on surface at a given time
    - time‑series at a material point
    - spatial mean / variance over time

Usage Examples
--------------
>>> from analysis_tools import Simulation
>>> sim = Simulation("./output/*.vtp")

# Plot field on surface at time t=5
>>> from analysis_tools import plot_field_on_surface
>>> plot_field_on_surface(sim, "c", 5)

# Get temporal spectrum
>>> freqs, P = sim.spectrum("c")

# Get spatial mean time‑series
>>> mean_c = sim.spatial_mean_over_time("c")
"""

from .simulation import Simulation
from .geometry import GeometryProcessor
from .spectral import SpectralTools
from .correlation import CorrelationTools
from .peaks import PeakTracker
from .lcs import LCSTools

# plotting helpers
from .plotting import (
    plot_field_on_surface,
    plot_spatial_mean,
    plot_spatial_std,
    plot_time_series
)

__all__ = [
    # core
    "Simulation",
    "GeometryProcessor",

    # analysis
    "SpectralTools",
    "CorrelationTools",
    "PeakTracker",
    "LCSTools",

    # plotting
    "plot_field_on_surface",
    "plot_spatial_mean",
    "plot_spatial_std",
    "plot_time_series",
]