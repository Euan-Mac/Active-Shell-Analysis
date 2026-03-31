import numpy as np
from matplotlib.tri import Triangulation


class GeometryProcessor:
    """
    Geometry + surface calculus utilities.
    Now fully vectorised for speed.
    """

    # --------------------------------------------------------------
    # Triangulation (already fast)
    # --------------------------------------------------------------
    @staticmethod
    def triangulation(mesh):
        pts = mesh.points
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
        tri = Triangulation(pts[:, 0], pts[:, 1], faces)
        z = pts[:, 2]
        return tri, z

    # --------------------------------------------------------------
    # Area-weighted COM (already vectorised)
    # --------------------------------------------------------------
    @staticmethod
    def center_of_mass(mesh, density=None, area_field="bary_area"):
        pts = mesh.points
        A = mesh.point_data.get(area_field)
        if A is None:
            raise ValueError(f"Missing dual area field '{area_field}'")

        if density and density in mesh.point_data:
            w = mesh.point_data[density] * A
        else:
            w = A

        w = w / np.sum(w)
        return np.sum(pts * w[:, None], axis=0)

    # --------------------------------------------------------------
    # Instantaneous integration (already vectorised)
    # --------------------------------------------------------------
    @staticmethod
    def integrate_vertex_field(mesh, field, area_field="bary_area"):
        φ = mesh.point_data[field]
        A = mesh.point_data[area_field]
        integral = np.sum(φ * A)
        average = integral / np.sum(A)
        return integral, average

    # --------------------------------------------------------------
    # Lagrangian extraction (fast)
    # --------------------------------------------------------------
    @staticmethod
    def spatial_series(sim, field, time_index=None):
        """
        Already fast: uses sim.field_lagrangian (vectorised).
        """
        raw = sim.field_lagrangian(field)  # (n_points, n_times)
        if time_index is None:
            return raw
        return raw[:, time_index]

    # --------------------------------------------------------------
    # FAST vectorised spatiotemporal integration
    # --------------------------------------------------------------
    @staticmethod
    def spatiotemporal_series(sim,
                              field,
                              area_field="bary_area",
                              reducer="surface_mean"):
        """
        Vectorised spatial integration over time.
        φ:  (n_points, n_times)
        A:  (n_times, n_points) -> transposed to (n_points, n_times)
        """

        φ = sim.field_lagrangian(field)          # (n_points, n_times)
        A = sim.area_lagrangian(area_field).T    # (n_points, n_times)

        if reducer == "surface_mean":
            num = np.sum(φ * A, axis=0)
            den = np.sum(A, axis=0)
            return num / den

        elif reducer == "surface_integral":
            return np.sum(φ * A, axis=0)

        elif reducer == "surface_L2":
            num = np.sum((φ ** 2) * A, axis=0)
            den = np.sum(A, axis=0)
            return np.sqrt(num / den)

        else:
            raise ValueError(f"Unknown reducer '{reducer}'")

    # --------------------------------------------------------------
    # FAST barycentric dual areas
    # --------------------------------------------------------------
    @staticmethod
    def compute_barycentric_dual_areas(mesh):
        pts = mesh.points
        faces = mesh.faces.reshape(-1, 4)[:, 1:]

        # Extract triangle vertex coords in vectorised form
        p0 = pts[faces[:, 0]]
        p1 = pts[faces[:, 1]]
        p2 = pts[faces[:, 2]]

        # Area for each triangle
        tri_areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)

        # Allocate dual areas
        A = np.zeros(len(pts))

        # Distribute 1/3 of each triangle area to each incident vertex
        third = tri_areas / 3.0
        np.add.at(A, faces[:, 0], third)
        np.add.at(A, faces[:, 1], third)
        np.add.at(A, faces[:, 2], third)

        return A

    # --------------------------------------------------------------
    # Voronoi dual areas (kept mostly as-is)
    # --------------------------------------------------------------
    @staticmethod
    def compute_voronoi_dual_areas(mesh):
        pts = mesh.points.astype(float)
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
        n = pts.shape[0]
        A = np.zeros(n)

        # Edge → list of adjacent triangles
        edge_tris = {}
        for tri in faces:
            for a, b in ((0,1),(1,2),(2,0)):
                key = tuple(sorted((tri[a], tri[b])))
                edge_tris.setdefault(key, []).append(tri)

        # Loop edges
        for (i, j), tris in edge_tris.items():
            p_i = pts[i]
            p_j = pts[j]
            e = p_j - p_i
            e2 = np.dot(e, e)

            cot_sum = 0.0
            for tri in tris:
                k = [v for v in tri if v not in (i, j)][0]
                u = p_i - pts[k]
                v = p_j - pts[k]
                cross = np.cross(u, v)
                denom = np.linalg.norm(cross)
                if denom > 1e-14:
                    cot_sum += np.dot(u, v) / denom

            contrib = 0.125 * cot_sum * e2
            A[i] += contrib
            A[j] += contrib

        return A

    # --------------------------------------------------------------
    # Wrapper
    # --------------------------------------------------------------
    @staticmethod
    def compute_dual_areas(mesh, mode="bary", store_as=None):
        if mode == "bary":
            A = GeometryProcessor.compute_barycentric_dual_areas(mesh)
            name = store_as or "bary_area"
        elif mode == "voro":
            A = GeometryProcessor.compute_voronoi_dual_areas(mesh)
            name = store_as or "voro_area"
        else:
            raise ValueError("mode must be 'bary' or 'voro'")

        mesh.point_data[name] = A
        return A
