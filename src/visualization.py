import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from src.core import IndoorEnvironment, PropagationModel


class EnvironmentVisualizer:
    def __init__(self, env: IndoorEnvironment, physics: PropagationModel):
        self.env = env
        self.physics = physics

    def draw_room_on_ax(self, ax):
        """Draws the room layout directly onto a provided matplotlib axis."""
        for segment in self.env.segments:
            x = [segment[0], segment[2]]
            y = [segment[1], segment[3]]
            ax.plot(x, y, color="black", linewidth=6, zorder=5)
        ax.set_aspect("equal", adjustable="box")

    def draw_rssi_heatmap_on_ax(self, ax, sender, resolution=0.2):
        max_x = np.max(self.env.segments[:, [0, 2]])
        max_y = np.max(self.env.segments[:, [1, 3]])
        X, Y = np.meshgrid(
            np.arange(0, max_x, resolution), np.arange(0, max_y, resolution), indexing="xy"
        )
        grid_points = np.column_stack((X.ravel(), Y.ravel()))

        # Calculate distances and collisions using the class environment
        distances = np.linalg.norm(grid_points - sender, axis=1)
        senders_repeated = np.tile(sender, (len(grid_points), 1))
        segments = np.hstack((senders_repeated, grid_points))
        collisions = np.array(
            [self.env.count_collisions(seg, self.env.segments) for seg in segments]
        )

        flat_powers = self.physics.received_power(distances, collisions)
        Z = flat_powers.reshape(X.shape)

        self.draw_room_on_ax(ax)
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

    def draw_prob_heatmap_on_ax(self, ax, sender, resolution=0.2):
        max_x = np.max(self.env.segments[:, [0, 2]])
        max_y = np.max(self.env.segments[:, [1, 3]])
        X, Y = np.meshgrid(
            np.arange(0, max_x, resolution), np.arange(0, max_y, resolution), indexing="xy"
        )
        grid_points = np.column_stack((X.ravel(), Y.ravel()))

        # Calculate distances and collisions using the class environment
        distances = np.linalg.norm(grid_points - sender, axis=1)
        senders_repeated = np.tile(sender, (len(grid_points), 1))
        segments = np.hstack((senders_repeated, grid_points))
        collisions = np.array(
            [self.env.count_collisions(seg, self.env.segments) for seg in segments]
        )

        flat_powers = self.physics.received_power(distances, collisions)
        flat_probabilities = self.physics.connection_probability(flat_powers)
        Z = flat_probabilities.reshape(X.shape)

        self.draw_room_on_ax(ax)
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
        self,
        ax,
        sensors,
        radius,
        resolution=0.5,
        upscale_factor=10,
        blur_sigma=4.0,
        title="Coverage",
    ):
        if len(sensors) == 0:
            return None

        max_x = max(np.max(self.env.segments[:, 0]), np.max(self.env.segments[:, 2]))
        max_y = max(np.max(self.env.segments[:, 1]), np.max(self.env.segments[:, 3]))

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
            [self.env.count_collisions(seg, self.env.segments) for seg in segments]
        ).reshape(K_sensors, M_pixels)

        penalty = np.where(collisions > 0, 1e-9, 1.0)
        effective_distances = distances / penalty

        Z_low = np.sum(effective_distances <= radius, axis=0).reshape(X.shape).astype(float)
        Z_high = zoom(Z_low, zoom=upscale_factor, order=1)
        Z_smooth = gaussian_filter(Z_high, sigma=blur_sigma)

        self.draw_room_on_ax(ax)
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
