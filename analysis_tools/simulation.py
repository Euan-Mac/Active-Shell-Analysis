import glob
import numpy as np
import pyvista as pv

from .geometry import GeometryProcessor


class Simulation:
    """
    Core container for Active Shell simulation data.

    Loads .vtp frames, caches geometry, exposes fields,
    and acts as the unified data source for all analysis tools.
    """

    def __init__(self, pattern):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No VTP files match pattern: {pattern}")

        self.frames = [pv.read(f) for f in self.files]

        # Extract time from VTK FieldData
        self.times = np.array([
            frame.field_data.get("TimeValue", [None])[0]
            for frame in self.frames
        ])

        # Cached geometric computations
        self._triangulation = None
        self._com_cache = {}

        # Precompute convenient derived fields
        self._prepare_derived_fields()

    # --------------------------
    # Derived field preparation
    # --------------------------
    def _prepare_derived_fields(self):
        for frame in self.frames:
            if "vel" in frame.point_data:
                v = frame.point_data["vel"]
                frame.point_data["vel_mag"] = np.linalg.norm(v, axis=1)

    # --------------------------
    # Mesh / field access
    # --------------------------
    def mesh(self, i):
        return self.frames[i]

    def field(self, name):
        """Return list of arrays, one per frame."""
        out = []
        for f in self.frames:
            if name in f.point_data:
                out.append(f.point_data[name])
            elif name in f.cell_data:
                out.append(f.cell_data[name])
            else:
                raise KeyError(f"Field '{name}' not found in any frame.")
        return out

    # --------------------------
    # COM and triangulation
    # --------------------------
    def com(self, i, density=None):
        key = (i, density)
        if key in self._com_cache:
            return self._com_cache[key]

        c = GeometryProcessor.center_of_mass(self.frames[i], density)
        self._com_cache[key] = c
        return c

    def triangulation(self):
        if self._triangulation is None:
            tri, z = GeometryProcessor.triangulation(self.frames[0])
            self._triangulation = (tri, z)
        return self._triangulation

    # --------------------------
    # Analysis API
    # --------------------------
    def animate(self, field, mode="static", outfile=None):
        from .animator import MatplotlibAnimator
        anim = MatplotlibAnimator(self, field, mode)
        if outfile:
            anim.save(outfile)
        return anim

    def diagnostics(self):
        from .diagnostics import SimulationDiagnostics
        diag = SimulationDiagnostics(self)
        return diag.run()

    def spectrum(self, field, reducer="max"):
        from .spectral import SpectralTools
        return SpectralTools.temporal_spectrum(self, field, reducer)

    def correlate(self, field):
        from .correlation import CorrelationTools
        return CorrelationTools.spatial(self, field)

    def track_peaks(self, field):
        from .peaks import PeakTracker
        return PeakTracker(self, field).run()

    def LCS(self, vel="vel"):
        from .lcs import LCSTools
        return LCSTools.compute(self, vel)