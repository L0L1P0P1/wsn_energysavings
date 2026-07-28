import numpy as np
from src.core import IndoorEnvironment

class SpatialOperators:
    MOVES = np.array(
        [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    )

    @staticmethod
    def random_empty_cell(env: IndoorEnvironment, occupied: set) -> np.ndarray:
        max_x, max_y = env.grid.shape
        while True:
            x, y = np.random.randint(max_x), np.random.randint(max_y)
            if env.grid[x, y] == 0 and (x, y) not in occupied:
                return np.array([x, y])

    @staticmethod
    def random_nodes(
        env: IndoorEnvironment, k: int, existing_occupied: set = None
    ) -> np.ndarray:
        nodes = set()
        occupied = existing_occupied if existing_occupied else set()
        while len(nodes) < k:
            cell = tuple(SpatialOperators.random_empty_cell(env, occupied | nodes))
            nodes.add(cell)
        return np.array(list(nodes), dtype=np.int64)

    @staticmethod
    def move_sensor(X: np.ndarray, env: IndoorEnvironment) -> np.ndarray:
        if len(X) == 0:
            return X
        X_new = X.copy()
        sensor_idx = np.random.randint(len(X))
        sensor = X_new[sensor_idx]

        occupied = set(map(tuple, X_new))
        occupied.remove(tuple(sensor))

        for move in np.random.permutation(SpatialOperators.MOVES):
            cand = (sensor + move).astype(int)
            x, y = cand[0], cand[1]
            if (
                0 <= x < env.grid.shape[0]
                and 0 <= y < env.grid.shape[1]
                and env.grid[x, y] == 0
            ):
                if (x, y) not in occupied:
                    X_new[sensor_idx] = [x, y]
                    return X_new
        return X_new

    @staticmethod
    def relocate_sensor(X: np.ndarray, env: IndoorEnvironment) -> np.ndarray:
        if len(X) == 0:
            return X
        X_new = X.copy()
        sensor_idx = np.random.randint(len(X))

        occupied = set(map(tuple, X_new))
        occupied.remove(tuple(X_new[sensor_idx]))

        X_new[sensor_idx] = SpatialOperators.random_empty_cell(env, occupied)
        return X_new

    @staticmethod
    def move_multiple_sensors(
        X: np.ndarray, env: IndoorEnvironment, k: int = 3
    ) -> np.ndarray:
        if len(X) == 0:
            return X
        X_new = X.copy()
        indices = np.random.choice(len(X), size=min(k, len(X)), replace=False)

        occupied = set(map(tuple, X_new))

        for idx in indices:
            occupied.remove(tuple(X_new[idx]))
            X_new[idx] = SpatialOperators.random_empty_cell(env, occupied)
            occupied.add(tuple(X_new[idx]))

        return X_new

    @staticmethod
    def geometry_aware_neighbor(X: np.ndarray, env: IndoorEnvironment) -> np.ndarray:
        r = np.random.rand()
        if r < 0.75:
            return SpatialOperators.move_sensor(X, env)
        elif r < 0.925:
            return SpatialOperators.relocate_sensor(X, env)
        else:
            return SpatialOperators.move_multiple_sensors(X, env)

    @staticmethod
    def crossover(
        p1: np.ndarray, p2: np.ndarray, env: IndoorEnvironment, avoid: set = None
    ) -> np.ndarray:
        target_len = (len(p1) + len(p2)) // 2
        if target_len == 0:
            target_len = 1

        pool = (
            np.vstack([p1, p2])
            if len(p1) > 0 and len(p2) > 0
            else (p1 if len(p1) > 0 else p2)
        )
        np.random.shuffle(pool)

        child, occupied = [], avoid if avoid else set()
        for pt in pool:
            tup = tuple(pt)
            if tup not in occupied:
                child.append(pt)
                occupied.add(tup)
            if len(child) == target_len:
                break
        return np.array(child)


