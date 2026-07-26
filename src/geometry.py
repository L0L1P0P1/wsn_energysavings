import numpy as np


def get_room_vector(room: np.ndarray):
    segments = []
    rows, cols = room.shape
    for i in range(rows):
        start = None
        for j in range(cols):
            if room[i, j] == -1:
                if start is None:
                    start = j
            else:
                if start is not None and j - start >= 2:
                    segments.append([i, start, i, j - 1])
                start = None
        if start is not None and cols - start >= 2:
            segments.append([i, start, i, cols - 1])

    for j in range(cols):
        start = None
        for i in range(rows):
            if room[i, j] == -1:
                if start is None:
                    start = i
            else:
                if start is not None and i - start >= 2:
                    segments.append([start, j, i - 1, j])
                start = None
        if start is not None and rows - start >= 2:
            segments.append([start, j, rows - 1, j])
    return np.array(segments)


def on_segment(p, q, r):
    pq_max_x = np.max(np.stack([p[..., 0], q[..., 0]]), axis=0)
    pq_min_x = np.min(np.stack([p[..., 0], q[..., 0]]), axis=0)
    pq_max_y = np.max(np.stack([p[..., 1], q[..., 1]]), axis=0)
    pq_min_y = np.min(np.stack([p[..., 1], q[..., 1]]), axis=0)
    return (
        (r[..., 0] <= pq_max_x)
        & (r[..., 0] >= pq_min_x)
        & (r[..., 1] <= pq_max_y)
        & (r[..., 1] >= pq_min_y)
    )


def orientation(p, q, r):
    val = (p[..., 1] - r[..., 1]) * (q[..., 0] - r[..., 0]) - (
        p[..., 0] - r[..., 0]
    ) * (q[..., 1] - r[..., 1])
    val[val == 0] = 0
    val[val > 0] = 1
    val[val < 0] = -1
    return val


def count_segment_collisions(seg1, seg2):
    o1 = orientation(seg1[..., :2], seg1[..., 2:], seg2[..., :2])
    o2 = orientation(seg1[..., :2], seg1[..., 2:], seg2[..., 2:])
    o3 = orientation(seg2[..., :2], seg2[..., 2:], seg1[..., :2])
    o4 = orientation(seg2[..., :2], seg2[..., 2:], seg1[..., 2:])

    collisions = (o1 != o2) & (o3 != o4)
    collisions = collisions | np.logical_and(
        o1 == 0, on_segment(seg1[..., :2], seg1[..., 2:], seg2[..., :2])
    )
    collisions = collisions | np.logical_and(
        o2 == 0, on_segment(seg1[..., :2], seg1[..., 2:], seg2[..., 2:])
    )
    collisions = collisions | np.logical_and(
        o3 == 0, on_segment(seg2[..., :2], seg2[..., 2:], seg1[..., :2])
    )
    collisions = collisions | np.logical_and(
        o4 == 0, on_segment(seg2[..., :2], seg2[..., 2:], seg1[..., 2:])
    )
    return np.count_nonzero(collisions)
