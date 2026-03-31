import numpy as np
from scipy.ndimage import label
from .geometry import GeometryProcessor


class PeakTracker:
    """
    Simple per-time-step peak finder.
    """

    def __init__(self, sim, field, threshold=None):
        self.sim = sim
        self.field = field
        self.threshold = threshold

    def run(self, time_index=-1):
        φ = GeometryProcessor.spatial_series(self.sim, self.field, time_index)
        if self.threshold is None:
            self.threshold = np.mean(φ) + 2*np.std(φ)
        mask = φ > self.threshold
        labeled, n = label(mask.astype(int))
        return {
            "num_peaks": int(n),
            "threshold": float(self.threshold)
        }