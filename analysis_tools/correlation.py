import numpy as np


class CorrelationTools:
    """
    Spatial correlation functions.
    Initial implementation: Euclidean correlation.
    Later: geodesic, spherical harmonics, etc.
    """

    @staticmethod
    def spatial(sim, field):
        vals = sim.field(field)
        final = vals[-1]      # last frame
        pts = sim.frames[-1].points

        x = final - final.mean()
        corr = np.dot(x, x) / len(x)

        return {
            "simple_variance": float(corr),
            "note": "This is a placeholder; implement geodesic correlations later."
        }