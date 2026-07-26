import numpy as np
import scipy
from src.geometry import count_segment_collisions


def received_power(
    sender, receivers, room_vector, n=1.5, ptx=-42, d0=1, obstacle_loss=10
):
    """Calculates continuous received power."""
    d = np.linalg.norm(receivers - sender, axis=1)
    d_safe = np.where(d == 0, 1e-9, d)
    senders_repeated = np.tile(sender, (len(receivers), 1))
    segments = np.hstack((senders_repeated, receivers))
    collisions = np.array(
        [count_segment_collisions(seg, room_vector) for seg in segments]
    )

    p_rec = ptx - 10 * n * np.log10(d_safe / d0)
    p_rec -= collisions * obstacle_loss
    p_rec[d == 0] = ptx
    return p_rec


def connection_probability(power, threshold, sigma):
    return 1 - scipy.stats.norm.cdf((threshold - power) / sigma, scale=sigma)


def precompute_room_environment(room, room_vector):
    x_indices, y_indices = np.where(room == 0)
    valid_cells = np.column_stack((x_indices, y_indices))
    M = len(valid_cells)
    dist_mat = np.zeros((M, M), dtype=np.float64)
    col_mat = np.zeros((M, M), dtype=np.int32)

    for i in range(M):
        p1 = valid_cells[i]
        diff = valid_cells - p1
        dist_mat[i, :] = np.linalg.norm(diff, axis=1)
        for j in range(i + 1, M):
            p2 = valid_cells[j]
            segment = np.hstack((p1, p2))
            collisions = count_segment_collisions(segment, room_vector)
            col_mat[i, j] = col_mat[j, i] = collisions

    return valid_cells, dist_mat, col_mat


def precompute_connection_probabilities(
    dist_mat, col_mat, ptx=-42, d0=1, n=1.5, obstacle_loss=15, threshold=-53, sigma=2
):
    d_safe = np.where(dist_mat == 0, 1e-9, dist_mat)
    p_rec = ptx - 10 * n * np.log10(d_safe / d0)
    p_rec -= col_mat * obstacle_loss
    np.fill_diagonal(p_rec, ptx)
    return 1 - scipy.stats.norm.cdf((threshold - p_rec) / sigma, scale=sigma)
