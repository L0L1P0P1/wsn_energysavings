import matplotlib.pyplot as plt
import numpy as np
from src.physics import received_power, connection_probability
from src.geometry import count_segment_collisions
from scipy.ndimage import gaussian_filter, zoom


def draw_room_on_ax(ax, room_vector):
    """Draws the room layout directly onto a provided matplotlib axis."""
    for segment in room_vector:
        x = [segment[0], segment[2]]
        y = [segment[1], segment[3]]
        ax.plot(x, y, color="black", linewidth=6, zorder=5)
    ax.set_aspect("equal", adjustable="box")


def draw_rssi_heatmap_on_ax(ax, sender, room_vector, resolution=0.2):
    max_x, max_y = np.max(room_vector[:, 0]), np.max(room_vector[:, 1])
    X, Y = np.meshgrid(
        np.arange(0, max_x, resolution), np.arange(0, max_y, resolution), indexing="xy"
    )
    grid_points = np.column_stack((X.ravel(), Y.ravel()))

    flat_powers = received_power(
        sender, grid_points, room_vector, d0=1, obstacle_loss=10, n=1.5
    )
    Z = flat_powers.reshape(X.shape)

    draw_room_on_ax(ax, room_vector)
    im = ax.imshow(
        Z,
        origin="lower",
        extent=[0, max_x, 0, max_y],
        aspect="equal",
        zorder=2,
        alpha=0.95,
    )
    ax.scatter(sender[0], sender[1], zorder=3, color="r")
    ax.set_title("RSSI Continuous Heatmap")
    return im


def draw_prob_heatmap_on_ax(
    ax, sender, room_vector, resolution=0.2, threshold=-53, sigma=2
):
    max_x, max_y = np.max(room_vector[:, 0]), np.max(room_vector[:, 1])
    X, Y = np.meshgrid(
        np.arange(0, max_x, resolution), np.arange(0, max_y, resolution), indexing="xy"
    )
    grid_points = np.column_stack((X.ravel(), Y.ravel()))

    flat_powers = received_power(
        sender, grid_points, room_vector, d0=1, obstacle_loss=10, n=1.5
    )
    flat_probabilities = connection_probability(flat_powers, threshold, sigma)
    Z = flat_probabilities.reshape(X.shape)

    draw_room_on_ax(ax, room_vector)
    im = ax.imshow(
        Z,
        origin="lower",
        extent=[0, max_x, 0, max_y],
        aspect="equal",
        zorder=2,
        alpha=0.95,
        cmap="plasma",
    )
    ax.scatter(sender[0], sender[1], zorder=3, color="r")
    ax.set_title("Q-Function Probability Map")
    return im


def draw_coverage_heatmap_on_ax(
    ax,
    sensors,
    room_vector,
    radius,
    resolution=0.5,
    upscale_factor=10,
    blur_sigma=4.0,
    title="Coverage",
):
    if len(sensors) == 0:
        return None

    max_x = max(np.max(room_vector[:, 0]), np.max(room_vector[:, 2]))
    max_y = max(np.max(room_vector[:, 1]), np.max(room_vector[:, 3]))

    X, Y = np.meshgrid(
        np.arange(0, max_x + resolution, resolution),
        np.arange(0, max_y + resolution, resolution),
        indexing="xy",
    )
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    M_pixels, sensors_arr = len(grid_points), np.array(sensors)
    K_sensors = len(sensors_arr)

    distances = np.linalg.norm(
        sensors_arr[:, None, :] - grid_points[None, :, :], axis=2
    )
    segments = np.hstack(
        (np.repeat(sensors_arr, M_pixels, axis=0), np.tile(grid_points, (K_sensors, 1)))
    )
    collisions = np.array(
        [count_segment_collisions(seg, room_vector) for seg in segments]
    ).reshape(K_sensors, M_pixels)

    penalty = np.where(collisions > 0, 1e-9, 1.0)
    effective_distances = distances / penalty

    Z_low = np.sum(effective_distances <= radius, axis=0).reshape(X.shape).astype(float)
    Z_high = zoom(Z_low, zoom=upscale_factor, order=1)
    Z_smooth = gaussian_filter(Z_high, sigma=blur_sigma)

    draw_room_on_ax(ax, room_vector)
    im = ax.imshow(
        Z_smooth,
        origin="lower",
        extent=[0, max_x, 0, max_y],
        aspect="equal",
        zorder=2,
        alpha=0.90,
        cmap="viridis",
    )
    ax.scatter(
        sensors_arr[:, 0],
        sensors_arr[:, 1],
        zorder=3,
        color="red",
        marker="o",
        edgecolors="black",
        s=50,
    )
    ax.set_title(title)
    ax.set_xlim(-0.5, max_x + 0.5)
    ax.set_ylim(-0.5, max_y + 0.5)
    return im
