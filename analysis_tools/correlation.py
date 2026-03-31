import numpy as np
from .geometry import GeometryProcessor


class CorrelationTools:
    """
    Placeholder for real geodesic correlations.
    """

    @staticmethod
    def spatial(sim, field, time_index=-1):
        φ = GeometryProcessor.spatial_series(sim, field, time_index)
        φ -= φ.mean()
        var = np.mean(φ**2)
        return {
            "variance": float(var),
            "note": "Geodesic correlation not implemented."
        }