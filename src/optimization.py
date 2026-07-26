import numpy as np


def calculate_coverage_rate(sensor_coords, valid_cells, dist_mat, col_mat, radius):
    if len(sensor_coords) == 0:
        return 0.0
    dists = np.linalg.norm(sensor_coords[:, None, :] - valid_cells[None, :, :], axis=2)
    s_idx = np.argmin(dists, axis=1)
    eff_dists = dist_mat[s_idx, :] / np.where(col_mat[s_idx, :] > 0, 1e-9, 1.0)
    return np.sum(np.any(eff_dists <= radius, axis=0)) / dist_mat.shape[1]


def calculate_deployment_cost(sensor_coords, unit_cost=1.0):
    return len(sensor_coords) * unit_cost


MOVES = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])


def random_empty_cell(room, occupied):
    max_x, max_y = room.shape

    while True:
        x = np.random.randint(max_x)
        y = np.random.randint(max_y)

        if room[x, y] == 0 and (x, y) not in occupied:
            return np.array([x, y])


def move_sensor(X, room):
    X_new = X.copy()

    sensor_idx = np.random.randint(len(X))
    sensor = X_new[sensor_idx].astype(int)

    for move in np.random.permutation(MOVES):
        candidate = sensor + move
        candidate = candidate.astype(int)

        x, y = int(candidate[0]), int(candidate[1])

        if 0 <= x < room.shape[0] and 0 <= y < room.shape[1] and room[x, y] == 0:
            X_new = np.asarray(X_new, dtype=np.int64).reshape(-1, 2)
            occupied = set(map(tuple, X_new))
            occupied.remove(tuple(sensor))

            if (x, y) not in occupied:
                X_new[sensor_idx] = [x, y]
                return X_new

    return X_new


def relocate_sensor(X, room):
    X_new = X.copy()

    sensor_idx = np.random.randint(len(X))

    occupied = set(map(tuple, X_new))
    occupied.remove(tuple(X_new[sensor_idx]))

    X_new[sensor_idx] = random_empty_cell(room, occupied)

    return X_new


def move_multiple_sensors(X, room, k=3):
    X_new = X.copy()

    indices = np.random.choice(len(X), size=min(k, len(X)), replace=False)

    occupied = set(map(tuple, X_new))

    for idx in indices:
        occupied.remove(tuple(X_new[idx]))

        X_new[idx] = random_empty_cell(room, occupied)

        occupied.add(tuple(X_new[idx]))

    return X_new


def geometry_aware_neighbor(X, room):
    r = np.random.rand()

    if r < 0.75:
        return move_sensor(X, room)

    elif r < 0.925:
        return relocate_sensor(X, room)

    else:
        return move_multiple_sensors(X, room)


def geometry_aware_SA(X, room, valid_cells, dist_mat, col_mat, alpha, temp, max_iter):
    args = dict(valid_cells=valid_cells, dist_mat=dist_mat, col_mat=col_mat, radius=3)
    f1 = calculate_coverage_rate(X, **args)
    best_X, best_f, history = X.copy(), f1, [f1]

    for _ in range(max_iter):
        new_X = geometry_aware_neighbor(X, room)
        f2 = calculate_coverage_rate(new_X, **args)

        if f2 > f1 or np.random.rand() < np.exp((f2 - f1) / temp):
            X, f1 = new_X, f2
        if f1 > best_f:
            best_f, best_X = f1, X.copy()

        history.append(f1)
        temp *= alpha

    return best_X, np.array(history)


def random_sensors(room, k):
    max_x, max_y, sensors = room.shape[0], room.shape[1], set()
    while len(sensors) < k:
        x, y = np.random.randint(0, max_x), np.random.randint(0, max_y)
        if room[x, y] == 0:
            sensors.add((x, y))
    return np.array(list(sensors), dtype=np.int64)


def crossover(p1, p2, room):
    child, mask = np.empty_like(p1), np.random.rand(len(p1)) < 0.5
    child[mask], child[~mask] = p1[mask], p2[~mask]
    occupied = set()
    for i in range(len(child)):
        if tuple(child[i]) in occupied:
            child[i] = random_empty_cell(room, occupied)
        occupied.add(tuple(child[i]))
    return child


def genetic_algorithm(
    room, valid_cells, dist_mat, col_mat, k, pop_size=40, gens=100, mut_rate=0.3
):
    args = dict(valid_cells=valid_cells, dist_mat=dist_mat, col_mat=col_mat, radius=3)
    pop = [random_sensors(room, k) for _ in range(pop_size)]
    best_X, best_f, history = None, -1, []

    for _ in range(gens):
        fits = np.array([calculate_coverage_rate(ind, **args) for ind in pop])
        max_idx = np.argmax(fits)
        if fits[max_idx] > best_f:
            best_f, best_X = fits[max_idx], pop[max_idx].copy()
        history.append(best_f)
        new_pop = [best_X.copy()]

        while len(new_pop) < pop_size:
            t1, t2 = (
                np.random.choice(pop_size, 2, replace=False),
                np.random.choice(pop_size, 2, replace=False),
            )
            parent1 = pop[t1[0]] if fits[t1[0]] > fits[t1[1]] else pop[t1[1]]
            parent2 = pop[t2[0]] if fits[t2[0]] > fits[t2[1]] else pop[t2[1]]
            child = crossover(parent1, parent2, room)
            if np.random.rand() < mut_rate:
                child = move_sensor(child, room)
            new_pop.append(child)
        pop = new_pop
    return best_X, np.array(history)
