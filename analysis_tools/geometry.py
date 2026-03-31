import numpy as np
from matplotlib.tri import Triangulation


class GeometryProcessor:
    """Geometry utilities for triangulation, COM, transforms."""

    @staticmethod
    def center_of_mass(mesh, density=None):
        pts = mesh.points
        if density and density in mesh.point_data:
            w = mesh.point_data[density]
            w = w / w.sum()
            return (pts * w[:, None]).sum(axis=0)
        return pts.mean(axis=0)

    @staticmethod
    def triangulation(mesh):
        pts = mesh.points
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
        tri = Triangulation(pts[:, 0], pts[:, 1], faces)
        z = pts[:, 2]
        return tri, z