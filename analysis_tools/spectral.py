import numpy as np


class SpectralTools:
    """
    Temporal FFT utilities for scalar fields over time.
    """

    @staticmethod
    def temporal_spectrum(sim, field, reducer="max"):
        """
        reducer = "max", "mean", "min", "L2"
        Computes FFT of the chosen field over time.
        """

        raw = sim.field(field)
        vals = []

        for arr in raw:
            if reducer == "max":
                vals.append(np.max(arr))
            elif reducer == "mean":
                vals.append(np.mean(arr))
            elif reducer == "min":
                vals.append(np.min(arr))
            elif reducer == "L2":
                vals.append(np.sqrt(np.mean(arr**2)))
            else:
                raise ValueError(f"Unknown reducer: {reducer}")

        vals = np.array(vals)
        dt = sim.times[1] - sim.times[0]

        freqs = np.fft.rfftfreq(len(vals), dt)
        spec = np.abs(np.fft.rfft(vals))**2

        return freqs, spec