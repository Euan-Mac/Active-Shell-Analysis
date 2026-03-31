import numpy as np
from scipy.ndimage import label


class PeakTracker:
    """
    Threshold + peak tracking.
    Placeholder for now; just thresholds last frame.
    """

    def __init__(self, sim, field, threshold=None):
        self.sim = sim
        self.field = field
        self.threshold = threshold

    def run(self):
        data = self.sim.field(self.field)[-1]  # last frame
        if self.threshold is None:
            self.threshold = np.mean(data) + 2*np.std(data)

        mask = data > self.threshold

        # Dummy peak count
        labels, n = label(mask.astype(int))

        return {
            "num_peaks": n,
            "threshold": float(self.threshold),
            "note": "Full peak tracking will be implemented later."
        }