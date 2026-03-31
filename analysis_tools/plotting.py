import matplotlib.pyplot as plt
from .geometry import GeometryProcessor


def plot_field_on_surface(sim, field, time_index, cmap="viridis"):
    φ = sim.field_at_time(field, time_index)
    tri, _ = sim.triangulation()

    fig, ax = plt.subplots(figsize=(6, 6))
    tpc = ax.tripcolor(tri, φ, shading="flat", cmap=cmap)
    ax.set_aspect("equal")
    plt.colorbar(tpc)
    ax.set_title(f"{field} at t={sim.times[time_index]:.3f}")
    return fig, ax


def plot_spatial_mean(sim, field, area_field="bary_area"):
    means = sim.spatial_mean_over_time(field, area_field)
    fig, ax = plt.subplots()
    ax.plot(sim.times, means)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"<{field}>")
    ax.set_title(f"Spatial mean of {field}(t)")
    return fig, ax


def plot_spatial_std(sim, field, area_field="bary_area"):
    stds = sim.spatial_std_over_time(field, area_field)
    fig, ax = plt.subplots()
    ax.plot(sim.times, stds)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"std({field})")
    ax.set_title(f"Spatial std of {field}(t)")
    return fig, ax


def plot_time_series(sim, field, material_id):
    ys = sim.time_series(field, material_id)
    fig, ax = plt.subplots()
    ax.plot(sim.times, ys)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{field}(id={material_id})")
    ax.set_title(f"Time series of {field} at ID {material_id}")
    return fig, ax