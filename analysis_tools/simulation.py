import glob
import numpy as np
import pyvista as pv

from .geometry import GeometryProcessor


class Simulation:
    """
    Unified simulation container:
      - loads VTP frames
      - builds material ID maps
      - provides field_lagrangian(), area_lagrangian()
      - high-level analysis helpers
    """

    def __init__(self, pattern):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No VTP files match pattern: {pattern}")

        self.frames = [pv.read(f) for f in self.files]
        self.times = np.array([
            f.field_data.get("TimeValue", [None])[0]
            for f in self.frames
        ])

        self._triangulation = None

        self.id_maps = []
        self._build_id_maps()
        self.material_ids = self.frames[0].point_data["id"].astype(int)

        self._prepare_derived_fields()

    def _prepare_derived_fields(self):
        for f in self.frames:
            if "vel" in f.point_data:
                v = f.point_data["vel"]
                f.point_data["vel_mag"] = np.linalg.norm(v, axis=1)

    def _build_id_maps(self):
        for f in self.frames:
            ids = f.point_data["id"].astype(int)
            self.id_maps.append({mid: i for i, mid in enumerate(ids)})

    # ------------------------------------------------------------
    # Access methods
    # ------------------------------------------------------------
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

    def field_lagrangian(self, field):
        ids0 = self.material_ids
        raw = []
        for mid in ids0:
            vals = []
            for k, frame in enumerate(self.frames):
                idx = self.id_maps[k][mid]
                vals.append(frame.point_data[field][idx])
            raw.append(vals)
        return np.array(raw)  # (n_points, n_times)

    def area_lagrangian(self, area_field="bary_area"):
        ids0 = self.material_ids
        allA = []
        for k, frame in enumerate(self.frames):
            A = frame.point_data[area_field]
            map_k = self.id_maps[k]
            aligned = [A[map_k[mid]] for mid in ids0]
            allA.append(aligned)
        return np.array(allA)  # (n_times, n_points)

    def triangulation(self):
        if self._triangulation is None:
            self._triangulation = GeometryProcessor.triangulation(self.frames[0])
        return self._triangulation

    # ------------------------------------------------------------
    # Extraction helpers requested by user
    # ------------------------------------------------------------
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
        vars = []
        for t in range(φ.shape[1]):
            μ = means[t]
            vars.append(np.sum(((φ[:, t] - μ)**2) * A[t]) / np.sum(A[t]))
        return np.sqrt(np.array(vars))

    # ------------------------------------------------------------
    # Links to other modules
    # ------------------------------------------------------------
    def spectrum(self, field, area_field="bary_area", weighting="mean-area"):
        from .spectral import SpectralTools
        return SpectralTools.temporal_spectrum(self, field, area_field, weighting)

    def animate(self, field, mode="static", outfile=None):
        from .animator import MatplotlibAnimator
        anim = MatplotlibAnimator(self, field, mode)
        if outfile:
            anim.save(outfile)
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