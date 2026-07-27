import numpy as np
from src.core import IndoorEnvironment


class DeploymentSolution:
    """Represents a single chromosome/solution in the population."""

    def __init__(self, sensors: np.ndarray):
        self.sensors = sensors
        self.coverage_rate: float = 0.0
        self.cost: float = 0.0
        self.rank: int = -1
        self.crowding_distance: float = 0.0

    @property
    def objectives(self) -> np.ndarray:
        return np.array([-self.coverage_rate, self.cost])


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
    def random_sensors(env: IndoorEnvironment, k: int) -> np.ndarray:
        sensors = set()
        while len(sensors) < k:
            sensors.add(tuple(SpatialOperators.random_empty_cell(env, sensors)))
        return np.array(list(sensors), dtype=np.int64)

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
    def crossover(p1: np.ndarray, p2: np.ndarray, env: IndoorEnvironment) -> np.ndarray:
        target_len = (len(p1) + len(p2)) // 2
        if target_len == 0:
            target_len = 1

        pool = np.vstack([p1, p2])
        np.random.shuffle(pool)

        child, occupied = [], set()
        for pt in pool:
            tup = tuple(pt)
            if tup not in occupied:
                child.append(pt)
                occupied.add(tup)
            if len(child) == target_len:
                break
        return np.array(child)


class BaseOptimizer:
    def __init__(self, env: IndoorEnvironment, radius: float = 3.0):
        self.env = env
        self.radius = radius

    def evaluate(self, sol: DeploymentSolution):
        sol.cost = float(len(sol.sensors))
        if len(sol.sensors) == 0:
            sol.coverage_rate = 0.0
            return

        dists = np.linalg.norm(
            sol.sensors[:, None, :] - self.env.valid_cells[None, :, :], axis=2
        )
        s_idx = np.argmin(dists, axis=1)
        eff_dists = self.env.dist_mat[s_idx, :] / np.where(
            self.env.col_mat[s_idx, :] > 0, 1e-9, 1.0
        )
        sol.coverage_rate = (
            np.sum(np.any(eff_dists <= self.radius, axis=0))
            / self.env.dist_mat.shape[1]
        )


class SimulatedAnnealing(BaseOptimizer):
    def __init__(
        self,
        env: IndoorEnvironment,
        radius: float = 3.0,
        temp: float = 0.001,
        alpha: float = 0.99995,
        max_iter: int = 10000,
    ):
        super().__init__(env, radius)
        self.temp = temp
        self.alpha = alpha
        self.max_iter = max_iter

    def run(self, init_k: int) -> tuple[DeploymentSolution, list[float]]:
        current_sol = DeploymentSolution(
            SpatialOperators.random_sensors(self.env, init_k)
        )
        self.evaluate(current_sol)

        best_sol = DeploymentSolution(current_sol.sensors.copy())
        best_sol.coverage_rate = current_sol.coverage_rate
        best_sol.cost = current_sol.cost
        history = [current_sol.coverage_rate]

        temp = self.temp

        for _ in range(self.max_iter):
            # Geometry-aware neighbor selection
            r = np.random.rand()
            if r < 0.75:
                new_sensors = SpatialOperators.move_sensor(
                    current_sol.sensors, self.env
                )
            else:
                new_sensors = SpatialOperators.relocate_sensor(
                    current_sol.sensors, self.env
                )

            candidate = DeploymentSolution(new_sensors)
            self.evaluate(candidate)

            # Acceptance criteria
            if (
                candidate.coverage_rate > current_sol.coverage_rate
                or np.random.rand()
                < np.exp((candidate.coverage_rate - current_sol.coverage_rate) / temp)
            ):
                current_sol = candidate

            if current_sol.coverage_rate > best_sol.coverage_rate:
                best_sol = DeploymentSolution(current_sol.sensors.copy())
                best_sol.coverage_rate = current_sol.coverage_rate

            history.append(current_sol.coverage_rate)
            temp *= self.alpha

        return best_sol, history


class GeneticAlgorithm(BaseOptimizer):
    def __init__(
        self,
        env: IndoorEnvironment,
        radius: float = 3.0,
        pop_size: int = 40,
        gens: int = 100,
        mut_rate: float = 0.3,
    ):
        super().__init__(env, radius)
        self.pop_size = pop_size
        self.gens = gens
        self.mut_rate = mut_rate

    def run(self, init_k: int) -> tuple[DeploymentSolution, list[float]]:
        pop = [
            DeploymentSolution(SpatialOperators.random_sensors(self.env, init_k))
            for _ in range(self.pop_size)
        ]
        for ind in pop:
            self.evaluate(ind)

        best_sol = None
        best_f = -1.0
        history = []

        for _ in range(self.gens):
            pop.sort(key=lambda x: x.coverage_rate, reverse=True)

            if pop[0].coverage_rate > best_f:
                best_f = pop[0].coverage_rate
                best_sol = DeploymentSolution(pop[0].sensors.copy())
                best_sol.coverage_rate = best_f
                best_sol.cost = pop[0].cost

            history.append(best_f)

            new_pop = [DeploymentSolution(best_sol.sensors.copy())]  # Elitism

            while len(new_pop) < self.pop_size:
                # Tournament Selection
                t1 = np.random.choice(self.pop_size, 2, replace=False)
                t2 = np.random.choice(self.pop_size, 2, replace=False)

                p1 = (
                    pop[t1[0]]
                    if pop[t1[0]].coverage_rate > pop[t1[1]].coverage_rate
                    else pop[t1[1]]
                )
                p2 = (
                    pop[t2[0]]
                    if pop[t2[0]].coverage_rate > pop[t2[1]].coverage_rate
                    else pop[t2[1]]
                )

                child_sensors = SpatialOperators.crossover(
                    p1.sensors, p2.sensors, self.env
                )

                if np.random.rand() < self.mut_rate:
                    child_sensors = SpatialOperators.move_sensor(
                        child_sensors, self.env
                    )

                child = DeploymentSolution(child_sensors)
                self.evaluate(child)
                new_pop.append(child)

            pop = new_pop

        return best_sol, history


class NSGA2:
    """Multi-Objective Optimizer utilizing Pareto Dominance."""

    def __init__(
        self,
        env: IndoorEnvironment,
        radius: float = 3.0,
        pop_size: int = 40,
        gens: int = 100,
    ):
        self.env = env
        self.radius = radius
        self.pop_size = pop_size
        self.gens = gens
        self.moves = np.array(
            [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
        )

    def evaluate(self, sol: DeploymentSolution):
        """Calculates both objective functions for a given solution."""
        sol.cost = float(len(sol.sensors))
        if len(sol.sensors) == 0:
            sol.coverage_rate = 0.0
            return

        dists = np.linalg.norm(
            sol.sensors[:, None, :] - self.env.valid_cells[None, :, :], axis=2
        )
        s_idx = np.argmin(dists, axis=1)
        eff_dists = self.env.dist_mat[s_idx, :] / np.where(
            self.env.col_mat[s_idx, :] > 0, 1e-9, 1.0
        )
        sol.coverage_rate = (
            np.sum(np.any(eff_dists <= self.radius, axis=0))
            / self.env.dist_mat.shape[1]
        )

    def _fast_non_dominated_sort(
        self, population: list[DeploymentSolution]
    ) -> list[list[int]]:
        N = len(population)
        S = [[] for _ in range(N)]
        fronts = [[]]
        n = np.zeros(N, dtype=int)

        for p in range(N):
            for q in range(N):
                obj_p = population[p].objectives
                obj_q = population[q].objectives

                # p dominates q if p is strictly better in at least one, and no worse in others
                p_dom_q = np.all(obj_p <= obj_q) and np.any(obj_p < obj_q)
                q_dom_p = np.all(obj_q <= obj_p) and np.any(obj_q < obj_p)

                if p_dom_q:
                    S[p].append(q)
                elif q_dom_p:
                    n[p] += 1

            if n[p] == 0:
                population[p].rank = 0
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        population[q].rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]
