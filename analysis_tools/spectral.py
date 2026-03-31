import numpy as np
from scipy.signal import periodogram


class SpectralTools:
    """
    Lagrangian temporal spectra:
      - compute PSD for every material point individually
      - subtract mean per point
      - average PSD across points using mean dual-area weights
    """

    @staticmethod
    def temporal_spectrum(sim,
                          field,
                          area_field="bary_area",
                          weighting="mean-area"):
        raw = np.array(sim.field_lagrangian(field))  # (n_points, n_times)
        dt = sim.times[1] - sim.times[0]

        demeaned = raw - raw.mean(axis=1, keepdims=True)

        if weighting == "mean-area":
            A_all = sim.area_lagrangian(area_field)
            A_mean = A_all.mean(axis=0)
            W = A_mean / A_mean.sum()
        elif weighting == "area0":
            A0 = sim.area_lagrangian(area_field)[0]
            W = A0 / A0.sum()
        elif weighting == "uniform":
            N = raw.shape[0]
            W = np.ones(N) / N
        else:
            raise ValueError("weighting must be 'mean-area', 'area0', or 'uniform'.")

        freqs = None
        spectra = []

        for i in range(raw.shape[0]):
            f, Pxx = periodogram(demeaned[i], fs=1/dt)
            if freqs is None:
                freqs = f
            spectra.append(Pxx)

        spectra = np.array(spectra)
        P = np.sum(spectra * W[:, None], axis=0)
        return freqs, P