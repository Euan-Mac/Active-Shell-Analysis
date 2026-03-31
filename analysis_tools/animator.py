import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from .geometry import GeometryProcessor


class MatplotlibAnimator:
    """
    Simple Matplotlib surface animation of 3D triangular meshes.
    Modes: static, com0, comtrack, zoomfit.
    """

    def __init__(self, sim, field, mode="static"):
        self.sim = sim
        self.field = field
        self.mode = mode

        self.com0 = sim.com(0)
        self.com_track = [sim.com(i) for i in range(len(sim.frames))]

        self.processed = []
        for i, mesh in enumerate(sim.frames):
            m = mesh.copy()
            pts = m.points.copy()

            if mode == "com0":
                pts -= (sim.com(i) - self.com0)
            elif mode == "comtrack":
                pts -= (self.com_track[i] - self.com_track[0])

            m.points = pts
            tri, z = GeometryProcessor.triangulation(m)
            scal = m.point_data.get(field, np.zeros(len(pts)))
            self.processed.append((tri, scal))

    def save(self, outfile, fps=25):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.set_axis_off()

        tri0, scal0 = self.processed[0]
        tpc = ax.tripcolor(tri0, scal0, shading='flat', cmap='viridis')
        plt.colorbar(tpc)

        def update(i):
            ax.clear()
            ax.set_aspect('equal')
            ax.set_axis_off()

            tri, scal = self.processed[i]
            tpc = ax.tripcolor(tri, scal, shading='flat', cmap='viridis')

            if self.mode == "zoomfit":
                ax.set_xlim(tri.x.min(), tri.x.max())
                ax.set_ylim(tri.y.min(), tri.y.max())

            return tpc,

        ani = FuncAnimation(fig, update,
                            frames=len(self.processed),
                            interval=40)

        ani.save(outfile, writer=FFMpegWriter(fps=fps))
        print(f"[animator] Saved: {outfile}")