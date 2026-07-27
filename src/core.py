import numpy as np
import scipy.stats
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PropagationModel:
    """Encapsulates the Log-Normal Shadowing physics parameters."""

    ptx: float = -42.0
    d0: float = 1.0
    n: float = 1.5
    obstacle_loss: float = 10.0
    threshold: float = -53.0
    sigma: float = 2.0

    def received_power(
        self, distances: np.ndarray, collisions: np.ndarray
    ) -> np.ndarray:
        d_safe = np.where(distances == 0, 1e-9, distances)
        p_rec = self.ptx - 10 * self.n * np.log10(d_safe / self.d0)
        p_rec -= collisions * self.obstacle_loss

        # Handle self-interference (distance = 0)
        if isinstance(p_rec, np.ndarray) and p_rec.ndim > 0:
            p_rec[distances == 0] = self.ptx
        return p_rec

    def connection_probability(self, powers: np.ndarray) -> np.ndarray:
        return 1 - scipy.stats.norm.cdf(
            (self.threshold - powers) / self.sigma, scale=self.sigma
        )


class IndoorEnvironment:
    """Manages the grid, vector geometry, and spatial precomputations."""

    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.segments = self._extract_segments()
        self.valid_cells = self._extract_valid_cells()

        # Cache matrices
        M = len(self.valid_cells)
        self.dist_mat = np.zeros((M, M), dtype=np.float64)
        self.col_mat = np.zeros((M, M), dtype=np.int32)
        self._precompute_matrices()

    def _extract_segments(self) -> np.ndarray:
        segments = []
        rows, cols = self.grid.shape
        # Horizontal segments
        for i in range(rows):
            start = None
            for j in range(cols):
                if self.grid[i, j] == -1:
                    if start is None:
                        start = j
                else:
                    if start is not None and j - start >= 2:
                        segments.append([i, start, i, j - 1])
                    start = None
            if start is not None and cols - start >= 2:
                segments.append([i, start, i, cols - 1])

        # Vertical segments
        for j in range(cols):
            start = None
            for i in range(rows):
                if self.grid[i, j] == -1:
                    if start is None:
                        start = i
                else:
                    if start is not None and i - start >= 2:
                        segments.append([start, j, i - 1, j])
                    start = None
            if start is not None and rows - start >= 2:
                segments.append([start, j, rows - 1, j])
        return np.array(segments)

    def _extract_valid_cells(self) -> np.ndarray:
        x_indices, y_indices = np.where(self.grid == 0)
        return np.column_stack((x_indices, y_indices))

    def count_collisions(self, seg1: np.ndarray, seg2: np.ndarray) -> int:
        def orientation(p, q, r):
            val = (p[..., 1] - r[..., 1]) * (q[..., 0] - r[..., 0]) - (
                p[..., 0] - r[..., 0]
            ) * (q[..., 1] - r[..., 1])
            val[val == 0] = 0
            val[val > 0] = 1
            val[val < 0] = -1
            return val

        def on_segment(p, q, r):
            pq_max = np.maximum(p, q)
            pq_min = np.minimum(p, q)
            return (
                (r[..., 0] <= pq_max[..., 0])
                & (r[..., 0] >= pq_min[..., 0])
                & (r[..., 1] <= pq_max[..., 1])
                & (r[..., 1] >= pq_min[..., 1])
            )

        o1 = orientation(seg1[..., :2], seg1[..., 2:], seg2[..., :2])
        o2 = orientation(seg1[..., :2], seg1[..., 2:], seg2[..., 2:])
        o3 = orientation(seg2[..., :2], seg2[..., 2:], seg1[..., :2])
        o4 = orientation(seg2[..., :2], seg2[..., 2:], seg1[..., 2:])

        cols = (o1 != o2) & (o3 != o4)
        cols = cols | (o1 == 0) & on_segment(
            seg1[..., :2], seg1[..., 2:], seg2[..., :2]
        )
        cols = cols | (o2 == 0) & on_segment(
            seg1[..., :2], seg1[..., 2:], seg2[..., 2:]
        )
        cols = cols | (o3 == 0) & on_segment(
            seg2[..., :2], seg2[..., 2:], seg1[..., :2]
        )
        cols = cols | (o4 == 0) & on_segment(
            seg2[..., :2], seg2[..., 2:], seg1[..., 2:]
        )
        return np.count_nonzero(cols)

    def _precompute_matrices(self):
        M = len(self.valid_cells)
        for i in range(M):
            p1 = self.valid_cells[i]
            diff = self.valid_cells - p1
            self.dist_mat[i, :] = np.linalg.norm(diff, axis=1)
            for j in range(i + 1, M):
                p2 = self.valid_cells[j]
                segment = np.hstack((p1, p2))
                collisions = self.count_collisions(segment, self.segments)
                self.col_mat[i, j] = self.col_mat[j, i] = collisions
