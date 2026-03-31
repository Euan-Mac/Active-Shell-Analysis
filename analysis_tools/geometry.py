import numpy as np
from matplotlib.tri import Triangulation


class GeometryProcessor:
    """
    Geometry + surface calculus utilities.

    Provides:
        - triangulation
        - area-weighted center of mass
        - instantaneous integration
        - Lagrangian per-point spatial series
        - Lagrangian area-weighted spatiotemporal series
    """

    @staticmethod
    def triangulation(mesh):
        pts = mesh.points
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
        tri = Triangulation(pts[:, 0], pts[:, 1], faces)
        z = pts[:, 2]
        return tri, z

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

        w /= np.sum(w)
        return np.sum(pts * w[:, None], axis=0)

    @staticmethod
    def integrate_vertex_field(mesh, field, area_field="bary_area"):
        φ = mesh.point_data[field]
        A = mesh.point_data[area_field]
        integral = np.sum(φ * A)
        average = integral / np.sum(A)
        return integral, average

    # ------------------------------------------------------------------
    # Lagrangian extraction modes
    # ------------------------------------------------------------------
    @staticmethod
    def spatial_series(sim, field, time_index=None):
        raw = np.array(sim.field_lagrangian(field))  # (n_points, n_times)
        if time_index is None:
            return raw
        return raw[:, time_index]

    @staticmethod
    def spatiotemporal_series(sim,
                              field,
                              area_field="bary_area",
                              reducer="surface_mean"):
        φ = np.array(sim.field_lagrangian(field))     # (n_points, n_times)
        A = sim.area_lagrangian(area_field)          # (n_times, n_points)

        n_times = φ.shape[1]
        vals = []

        for t in range(n_times):
            phi_t = φ[:, t]
            A_t = A[t]

            if reducer == "surface_mean":
                val = np.sum(phi_t * A_t) / np.sum(A_t)
            elif reducer == "surface_integral":
                val = np.sum(phi_t * A_t)
            elif reducer == "surface_L2":
                val = np.sqrt(np.sum((phi_t**2) * A_t) / np.sum(A_t))
            else:
                raise ValueError(f"Unknown reducer '{reducer}'")
            vals.append(val)

        return np.array(vals)
    
    @staticmethod
    def compute_barycentric_dual_areas(mesh):
        """
        Compute barycentric dual area:
            A_i = 1/3 * sum(area of all incident triangles)
        This produces positive, robust dual areas for any triangulated surface.
        """
        pts = mesh.points
        faces = mesh.faces.reshape(-1, 4)[:, 1:]   # VTK format: [3, i,j,k]

        n_points = pts.shape[0]
        A = np.zeros(n_points)

        for tri in faces:
            p0, p1, p2 = pts[tri]
            area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
            # distribute uniformly to its 3 vertices
            A[tri] += area / 3.0

        return A

    @staticmethod
    def compute_voronoi_dual_areas(mesh):
        """
        Compute Voronoi dual area (a.k.a. mixed area):
        Using cotangent formula on triangulated surface.

        A_i = 1/8 * sum( (cot α + cot β) * |e_ij|^2 )
        where α, β are angles opposite the edge e_ij in adjacent triangles.

        This approximate discrete Voronoi area is widely used in DDG.
        """

        pts = mesh.points.astype(float)
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
        n = pts.shape[0]
        A = np.zeros(n)

        # Build adjacency: for each unordered edge, store adjacent triangles
        edge_tris = {}
        for tri in faces:
            for a, b in [(0,1),(1,2),(2,0)]:
                i = tri[a]
                j = tri[b]
                key = tuple(sorted((i, j)))
                if key not in edge_tris:
                    edge_tris[key] = []
                edge_tris[key].append(tri)

        # Loop over edges, handle those with 1 or 2 adjacent triangles
        for (i, j), tris in edge_tris.items():

            p_i = pts[i]
            p_j = pts[j]
            e = p_j - p_i
            e2 = np.dot(e, e)

            cot_sum = 0.0

            for tri in tris:
                # find the vertex opposite the edge (i, j)
                k = [v for v in tri if v not in (i, j)][0]
                p_k = pts[k]

                # Compute angle at vertex k in triangle i-j-k
                u = p_i - p_k
                v = p_j - p_k
                # cot(angle_k) = dot(u,v) / |cross(u,v)|
                cross = np.cross(u, v)
                denom = np.linalg.norm(cross)
                if denom > 1e-14:
                    cot_sum += np.dot(u, v) / denom

            # Mixed Voronoi area contribution
            A[i] += 0.125 * cot_sum * e2
            A[j] += 0.125 * cot_sum * e2

        return A

    @staticmethod
    def compute_dual_areas(mesh, mode="bary", store_as=None):
        """
        Compute dual area and store it into mesh.point_data.

        mode:
            "bary"  -> barycentric dual area (always safe)
            "voro"  -> Voronoi dual area (cotan weights)

        store_as:
            name of point_data field. If None, choose:
                "bary_area" or "voro_area".
        """
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
