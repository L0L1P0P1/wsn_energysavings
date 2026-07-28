import numpy as np
from src.core import IndoorEnvironment
from src.optimizer.operators import SpatialOperators


class SingleTierSolution:
    """Represents a flat, single-tier sensor network for PoC algorithms."""

    def __init__(self, sensors: np.ndarray):
        self.sensors = sensors
        self.coverage_rate: float = 0.0
        self.cost: float = 0.0


class SingleTierOptimizer:
    """Base evaluator for single-objective PoC algorithms."""

    def __init__(self, env: IndoorEnvironment, radius: float = 3.0):
        self.env = env
        self.radius = radius

    def evaluate(self, sol: SingleTierSolution):
        """Evaluates only cost and coverage. Ignores connectivity and relays."""
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


class SimulatedAnnealing(SingleTierOptimizer):
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

    def run(self, init_k: int) -> tuple[SingleTierSolution, list[float]]:
        current_sol = SingleTierSolution(
            SpatialOperators.random_nodes(self.env, init_k)
        )
        self.evaluate(current_sol)

        best_sol = SingleTierSolution(current_sol.sensors.copy())
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
            elif r < 0.925:
                new_sensors = SpatialOperators.relocate_sensor(
                    current_sol.sensors, self.env
                )
            else:
                new_sensors = SpatialOperators.move_multiple_sensors(
                    current_sol.sensors, self.env
                )

            candidate = SingleTierSolution(new_sensors)
            self.evaluate(candidate)

            # Acceptance criteria
            if (
                candidate.coverage_rate > current_sol.coverage_rate
                or np.random.rand()
                < np.exp((candidate.coverage_rate - current_sol.coverage_rate) / temp)
            ):
                current_sol = candidate

            # Track Global Best
            if current_sol.coverage_rate > best_sol.coverage_rate:
                best_sol = SingleTierSolution(current_sol.sensors.copy())
                best_sol.coverage_rate = current_sol.coverage_rate
                best_sol.cost = current_sol.cost

            history.append(current_sol.coverage_rate)
            temp *= self.alpha

        return best_sol, history


class GeneticAlgorithm(SingleTierOptimizer):
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

    def run(self, init_k: int) -> tuple[SingleTierSolution, list[float]]:
        pop = [
            SingleTierSolution(SpatialOperators.random_nodes(self.env, init_k))
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
                best_sol = SingleTierSolution(pop[0].sensors.copy())
                best_sol.coverage_rate = best_f
                best_sol.cost = pop[0].cost

            history.append(best_f)

            new_pop = [SingleTierSolution(best_sol.sensors.copy())]  # Elitism

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

                if len(child_sensors) > 0:
                    child = SingleTierSolution(child_sensors)
                    self.evaluate(child)
                    new_pop.append(child)

            pop = new_pop

        return best_sol, history
