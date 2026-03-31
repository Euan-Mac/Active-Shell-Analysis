import numpy as np


class SimulationDiagnostics:
    """
    Stability checks to detect blow-ups, NaNs, excessive velocity, etc.
    """

    def __init__(self, sim):
        self.sim = sim

    # -----------------------------------
    # Basic metrics
    # -----------------------------------
    def velocity_max(self):
        vlist = self.sim.field("vel_mag")
        return np.array([np.max(v) for v in vlist])

    def area_min(self):
        out = []
        for frame in self.sim.frames:
            if "area" in frame.cell_data:
                out.append(np.min(frame.cell_data["area"]))
            else:
                out.append(None)
        return np.array(out)

    # -----------------------------------
    # Checks
    # -----------------------------------
    def check_blowup(self, vel_thresh=1e3):
        vmax = self.velocity_max()
        if np.any(np.isnan(vmax)):
            return True, "NaN in velocity"
        if np.any(vmax > vel_thresh):
            return True, f"Velocity exceeded threshold {vel_thresh}"
        return False, "OK"

    def check_collapse(self, area_thresh=1e-8):
        Amin = self.area_min()
        good = [(a is None) or (a > area_thresh) for a in Amin]
        if not all(good):
            return True, "Triangle collapse / degeneracy"
        return False, "OK"

    # -----------------------------------
    # Run all diagnostics
    # -----------------------------------
    def run(self):
        blow, blow_reason = self.check_blowup()
        col, col_reason = self.check_collapse()

        summary = {
            "blowup": blow,
            "blowup_reason": blow_reason,
            "collapse": col,
            "collapse_reason": col_reason,
            "velocity_max": self.velocity_max().tolist(),
            "area_min": self.area_min().tolist(),
        }
        return summary