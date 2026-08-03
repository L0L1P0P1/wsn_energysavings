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
            np.arange(0, max_x, resolution),
            np.arange(0, max_y, resolution),
            indexing="xy",
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
            np.arange(0, max_x, resolution),
            np.arange(0, max_y, resolution),
            indexing="xy",
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
        relays=None,
        bs_pos=None,
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
            (
                np.repeat(sensors_arr, M_pixels, axis=0),
                np.tile(grid_points, (K_sensors, 1)),
            )
        )
        collisions = np.array(
            [self.env.count_collisions(seg, self.env.segments) for seg in segments]
        ).reshape(K_sensors, M_pixels)

        penalty = np.where(collisions > 0, 1e-9, 1.0)
        effective_distances = distances / penalty

        Z_low = (
            np.sum(effective_distances <= radius, axis=0).reshape(X.shape).astype(float)
        )
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

        # ---------------------------------------------------------
        # Topological Connections (Lines)
        # ---------------------------------------------------------
        if relays is not None and len(relays) > 0:
            relays_arr = np.array(relays)

            # Connect Relays to Base Station (White dashed lines)
            if bs_pos is not None and len(bs_pos) > 0:
                for r in relays_arr:
                    ax.plot(
                        [r[0], bs_pos[0]],
                        [r[1], bs_pos[1]],
                        color="white",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                        zorder=3,
                    )

            # Connect Sensors to Nearest Relay (Black dotted lines)
            for s in sensors_arr:
                dists = np.linalg.norm(relays_arr - s, axis=1)
                nearest_r = relays_arr[np.argmin(dists)]
                ax.plot(
                    [s[0], nearest_r[0]],
                    [s[1], nearest_r[1]],
                    color="black",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.6,
                    zorder=3,
                )

        # ---------------------------------------------------------
        # Hardware Nodes (Markers)
        # ---------------------------------------------------------
        ax.scatter(
            sensors_arr[:, 0],
            sensors_arr[:, 1],
            zorder=4,
            color="red",
            marker="o",
            edgecolors="black",
            s=50,
            label="Sensors",
        )

        if relays is not None and len(relays) > 0:
            ax.scatter(
                relays_arr[:, 0],
                relays_arr[:, 1],
                zorder=5,
                color="orange",
                marker="^",
                edgecolors="black",
                s=90,
                label="Relays",
            )

        if bs_pos is not None and len(bs_pos) > 0:
            ax.scatter(
                bs_pos[0],
                bs_pos[1],
                zorder=6,
                color="cyan",
                marker="s",
                edgecolors="black",
                s=130,
                label="Base Station",
            )

        ax.set_title(title)
        ax.set_xlim(-0.5, max_x + 0.5)
        ax.set_ylim(-0.5, max_y + 0.5)

        # Add a legend if we are plotting a multi-tier layout
        if (relays is not None and len(relays) > 0) or (
            bs_pos is not None and len(bs_pos) > 0
        ):
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

        return im
