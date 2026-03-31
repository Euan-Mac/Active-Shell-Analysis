import glob
import numpy as np
import pyvista as pv

from .geometry import GeometryProcessor


class Simulation:
    """
    Fully vectorised Simulation container.
    Adds:
        - lagrangian_indices: (n_times, n_points)
        - fast field_lagrangian (no nested loops)
        - fast area_lagrangian
    """

    def __init__(self, pattern):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No VTP files match pattern: {pattern}")

        self.frames = [pv.read(f) for f in self.files]

        # Time array
        self.times = np.array([
            f.field_data.get("TimeValue", [None])[0]
            for f in self.frames
        ])

        # Build ID maps
        self.id_maps = []
        self._build_id_maps()

        # Lagrangian ordering (material point IDs)
        self.material_ids = self.frames[0].point_data["id"].astype(int)

        # Build Lagrangian index matrix (FAST access)
        self.lagrangian_indices = self._build_lagrangian_indices()

        # Cached triangulation
        self._triangulation = None

        # Derived fields
        self._prepare_derived_fields()

    # --------------------------------------------------------------
    def _prepare_derived_fields(self):
        for frame in self.frames:
            if "vel" in frame.point_data:
                v = frame.point_data["vel"]
                frame.point_data["vel_mag"] = np.linalg.norm(v, axis=1)

    # --------------------------------------------------------------
    # ID maps: per frame dict(mid -> vertex_index)
    # --------------------------------------------------------------
    def _build_id_maps(self):
        for f in self.frames:
            ids = f.point_data["id"].astype(int)
            self.id_maps.append({mid: i for i, mid in enumerate(ids)})

    # --------------------------------------------------------------
    # Build Lagrangian index matrix (FAST)
    # --------------------------------------------------------------
    def _build_lagrangian_indices(self):
        ids0 = self.material_ids
        n_times = len(self.frames)
        n_points = len(ids0)

        idx_mat = np.zeros((n_times, n_points), dtype=int)

        for t in range(n_times):
            mapping = self.id_maps[t]
            # vectorised mapping using list comprehension only once
            idx_mat[t] = [mapping[mid] for mid in ids0]

        return idx_mat

    # --------------------------------------------------------------
    # Eulerian access (unchanged)
    # --------------------------------------------------------------
    def field(self, name):
        out = []
        for f in self.frames:
            if name in f.point_data:
                out.append(f.point_data[name])
            elif name in f.cell_data:
                out.append(f.cell_data[name])
            else:
                raise KeyError(f"Field '{name}' not found.")
        return out

    # --------------------------------------------------------------
    # FAST Lagrangian field extraction
    # --------------------------------------------------------------
    def field_lagrangian(self, field):
        """
        Return array (n_points, n_times)
        Uses lagrangian_indices for fast traversal.
        """

        n_times, n_points = self.lagrangian_indices.shape
        out = np.zeros((n_points, n_times))

        for t, frame in enumerate(self.frames):
            vals = frame.point_data[field]
            out[:, t] = vals[self.lagrangian_indices[t]]

        return out

    # --------------------------------------------------------------
    # FAST Lagrangian area extraction
    # --------------------------------------------------------------
    def area_lagrangian(self, area_field="bary_area"):
        n_times, n_points = self.lagrangian_indices.shape
        out = np.zeros((n_times, n_points))

        for t, frame in enumerate(self.frames):
            A = frame.point_data[area_field]
            out[t] = A[self.lagrangian_indices[t]]

        return out

    # --------------------------------------------------------------
    def triangulation(self):
        if self._triangulation is None:
            self._triangulation = GeometryProcessor.triangulation(self.frames[0])
        return self._triangulation

    # --------------------------------------------------------------
    # Analysis helpers (unchanged)
    # --------------------------------------------------------------
    def field_at_time(self, field, time_index):
        return GeometryProcessor.spatial_series(self, field, time_index)

    def time_series(self, field, material_id):
        raw = self.field_lagrangian(field)
        idx = np.where(self.material_ids == material_id)[0][0]
        return raw[idx]

    def spatial_mean(self, field, time_index, area_field="bary_area"):
        φ = self.field_at_time(field, time_index)
        A = self.area_lagrangian(area_field)[time_index]
        return np.sum(φ * A) / np.sum(A)

    def spatial_variance(self, field, time_index, area_field="bary_area"):
        φ = self.field_at_time(field, time_index)
        A = self.area_lagrangian(area_field)[time_index]
        μ = np.sum(φ * A) / np.sum(A)
        return np.sum(((φ - μ)**2) * A) / np.sum(A)

    def spatial_mean_over_time(self, field, area_field="bary_area"):
        return GeometryProcessor.spatiotemporal_series(
            self, field, area_field, reducer="surface_mean"
        )

    def spatial_std_over_time(self, field, area_field="bary_area"):
        means = self.spatial_mean_over_time(field, area_field)
        φ = self.field_lagrangian(field)
        A = self.area_lagrangian(area_field)
        vars = np.sum(((φ - means) ** 2) * A.T, axis=0) / np.sum(A, axis=1)
        return np.sqrt(vars)

    # --------------------------------------------------------------
    # External module wrappers unchanged
    # --------------------------------------------------------------
    def spectrum(self, *args, **kwargs):
        from .spectral import SpectralTools
        return SpectralTools.temporal_spectrum(self, *args, **kwargs)

    def animate(self, *args, **kwargs):
        from .animator import MatplotlibAnimator
        anim = MatplotlibAnimator(self, *args, **kwargs)
        return anim

    def diagnostics(self):
        from .diagnostics import SimulationDiagnostics
        return SimulationDiagnostics(self).run()

    def correlate(self, field):
        from .correlation import CorrelationTools
        return CorrelationTools.spatial(self, field)

    def track_peaks(self, field):
        from .peaks import PeakTracker
        return PeakTracker(self, field).run()

    def LCS(self, vel_field="vel"):
        from .lcs import LCSTools
        return LCSTools.compute(self, vel_field)