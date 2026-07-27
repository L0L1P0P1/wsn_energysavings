import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from src.core import IndoorEnvironment, PropagationModel


class EnvironmentVisualizer:
    def __init__(self, env: IndoorEnvironment, physics: PropagationModel):
        self.env = env
        self.physics = physics

    def draw_room(self, ax):
        for segment in self.env.segments:
            x, y = [segment[0], segment[2]], [segment[1], segment[3]]
            ax.plot(x, y, color="black", linewidth=6, zorder=5)
        ax.set_aspect("equal", adjustable="box")

    def draw_coverage_heatmap(self, ax, sensors: np.ndarray, radius: float = 3.0):
        if len(sensors) == 0:
            return

        max_x = max(np.max(self.env.segments[:, 0]), np.max(self.env.segments[:, 2]))
        max_y = max(np.max(self.env.segments[:, 1]), np.max(self.env.segments[:, 3]))

        # Setup standard grid mapping
        X, Y = np.meshgrid(
            np.arange(0, max_x + 0.5, 0.5),
            np.arange(0, max_y + 0.5, 0.5),
            indexing="xy",
        )
        grid_points = np.column_stack((X.ravel(), Y.ravel()))

        distances = np.linalg.norm(
            sensors[:, None, :] - grid_points[None, :, :], axis=2
        )
        segments = np.hstack(
            (
                np.repeat(sensors, len(grid_points), axis=0),
                np.tile(grid_points, (len(sensors), 1)),
            )
        )

        # We can dynamically use the physics model or the environment for evaluation here
        collisions = np.array(
            [self.env.count_collisions(seg, self.env.segments) for seg in segments]
        ).reshape(len(sensors), len(grid_points))

        penalty = np.where(collisions > 0, 1e-9, 1.0)
        Z_low = (
            np.sum((distances / penalty) <= radius, axis=0)
            .reshape(X.shape)
            .astype(float)
        )

        Z_smooth = gaussian_filter(zoom(Z_low, zoom=10, order=1), sigma=4.0)

        self.draw_room(ax)
        ax.imshow(
            Z_smooth,
            origin="lower",
            extent=[0, max_x, 0, max_y],
            zorder=2,
            alpha=0.90,
            cmap="viridis",
        )
        ax.scatter(
            sensors[:, 0],
            sensors[:, 1],
            zorder=3,
            color="red",
            edgecolors="black",
            s=50,
        )
