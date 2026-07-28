import numpy as np
from src.core import IndoorEnvironment, PropagationModel
from src.optimizer.operators import SpatialOperators

class DeploymentSolution:
    """Represents a two-tier network topology."""

    def __init__(self, sensors: np.ndarray, relays: np.ndarray):
        self.sensors = sensors
        self.relays = relays

        # Objectives
        self.coverage_rate: float = 0.0
        self.reliability: float = 0.0
        self.cost: float = 0.0

        # Constraint Tracking
        self.penalty: float = 0.0

        # NSGA-II Metadata
        self.rank: int = -1
        self.crowding_distance: float = 0.0

    @property
    def objectives(self) -> np.ndarray:
        # Maximize Coverage, Maximize Reliability, Minimize Cost
        # We negate the maximizing objectives for a standard minimum-sort.
        # The penalty is heavily applied to push invalid topologies out of the Pareto front.
        return np.array(
            [
                -self.coverage_rate + self.penalty,
                -self.reliability + self.penalty,
                self.cost + self.penalty,
            ]
        )

class BaseOptimizer:
    def __init__(
        self,
        env: IndoorEnvironment,
        physics: PropagationModel,
        bs_pos: np.ndarray,
        radius: float = 3.0,
    ):
        self.env = env
        self.physics = physics
        self.bs_pos = bs_pos
        self.radius = radius

        # Hyperparameters based on paper Eq 6 & 7
        self.cost_sensor = 1.0
        self.cost_relay = 5.0
        self.connection_threshold = (
            0.85  # Minimum probability to consider a link "active"
        )

    def evaluate(self, sol: DeploymentSolution):
        """Evaluates Cost, Coverage, Reliability, and Connectivity Constraints."""

        # 1. Cost Objective (Eq 6 & 7)
        sol.cost = (len(sol.sensors) * self.cost_sensor) + (
            len(sol.relays) * self.cost_relay
        )

        if len(sol.sensors) == 0 or len(sol.relays) == 0:
            sol.coverage_rate = 0.0
            sol.reliability = 0.0
            sol.penalty = 10000.0
            return

        # 2. Coverage Objective (Eq 11)
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

        # 3. Graph Connectivity Constraints & Reliability (Eq 12, 13, 22-25)
        # Check Relays -> Base Station
        dist_r_bs = np.linalg.norm(sol.relays - self.bs_pos, axis=1)
        segments_r_bs = np.hstack(
            (sol.relays, np.tile(self.bs_pos, (len(sol.relays), 1)))
        )
        cols_r_bs = np.array(
            [self.env.count_collisions(seg, self.env.segments) for seg in segments_r_bs]
        )

        powers_r_bs = self.physics.received_power(dist_r_bs, cols_r_bs)
        prob_r_bs = self.physics.connection_probability(powers_r_bs)

        disconnected_relays = np.sum(prob_r_bs < self.connection_threshold)

        # Check Sensors -> Relays
        dist_s_r = np.linalg.norm(
            sol.sensors[:, None, :] - sol.relays[None, :, :], axis=2
        )
        segments_s_r = np.hstack(
            (
                np.repeat(sol.sensors, len(sol.relays), axis=0),
                np.tile(sol.relays, (len(sol.sensors), 1)),
            )
        )
        cols_s_r = np.array(
            [self.env.count_collisions(seg, self.env.segments) for seg in segments_s_r]
        ).reshape(len(sol.sensors), len(sol.relays))

        powers_s_r = self.physics.received_power(dist_s_r, cols_s_r)
        prob_s_r = self.physics.connection_probability(powers_s_r)

        # Link matrix (1 if connected, 0 if not)
        links = (prob_s_r >= self.connection_threshold).astype(int)
        k_j = np.sum(links, axis=1)  # Links per sensor (Eq 22)

        disconnected_sensors = np.sum(k_j == 0)

        # Apply strict constraint penalties
        sol.penalty = (disconnected_relays + disconnected_sensors) * 1000.0

        # Objective 3: Reliability Calculation (Eq 23, 24, 25)
        if len(k_j) > 0 and disconnected_sensors == 0:
            k_mean = np.mean(k_j)
            k_var = np.var(k_j)
            sol.reliability = np.sum(k_mean - k_var)
        else:
            sol.reliability = 0.0

class NSGA2(BaseOptimizer):
    """Multi-Objective Optimizer utilizing Pareto Dominance."""

    def __init__(
        self,
        env: IndoorEnvironment,
        physics: PropagationModel,
        bs_pos: np.ndarray,
        radius: float = 3.0,
        pop_size: int = 40,
        gens: int = 100,
    ):
        super().__init__(env, physics, bs_pos, radius)
        self.pop_size = pop_size
        self.gens = gens

    # ... (Keep your _fast_non_dominated_sort here. It natively supports 3 objectives) ...

    def _crowding_distance_assignment(
        self, front: list[int], population: list[DeploymentSolution]
    ):
        l = len(front)
        for i in front:
            population[i].crowding_distance = 0.0
        if l <= 2:
            for i in front:
                population[i].crowding_distance = float("inf")
            return

        # Loop through all 3 objectives now
        for m in range(3):
            front.sort(key=lambda x: population[x].objectives[m])
            population[front[0]].crowding_distance = float("inf")
            population[front[-1]].crowding_distance = float("inf")

            obj_min = population[front[0]].objectives[m]
            obj_max = population[front[-1]].objectives[m]
            if obj_max - obj_min == 0:
                continue

            for i in range(1, l - 1):
                population[front[i]].crowding_distance += (
                    population[front[i + 1]].objectives[m]
                    - population[front[i - 1]].objectives[m]
                ) / (obj_max - obj_min)

    def run(self, init_s: int, init_r: int) -> list[DeploymentSolution]:
        pop = []
        for _ in range(self.pop_size):
            s = SpatialOperators.random_nodes(self.env, init_s)
            r = SpatialOperators.random_nodes(
                self.env, init_r, existing_occupied=set(map(tuple, s))
            )
            pop.append(DeploymentSolution(s, r))

        for ind in pop:
            self.evaluate(ind)

        for gen in range(self.gens):
            self._fast_non_dominated_sort(pop)

            offspring = []
            while len(offspring) < self.pop_size:
                # Tournament Selection
                t1, t2 = np.random.choice(len(pop), 2, replace=False)
                p1 = (
                    pop[t1]
                    if (
                        pop[t1].rank < pop[t2].rank
                        or (
                            pop[t1].rank == pop[t2].rank
                            and pop[t1].crowding_distance > pop[t2].crowding_distance
                        )
                    )
                    else pop[t2]
                )

                t3, t4 = np.random.choice(len(pop), 2, replace=False)
                p2 = (
                    pop[t3]
                    if (
                        pop[t3].rank < pop[t4].rank
                        or (
                            pop[t3].rank == pop[t4].rank
                            and pop[t3].crowding_distance > pop[t4].crowding_distance
                        )
                    )
                    else pop[t4]
                )

                # Crossover (Independently for sensors and relays)
                child_s = SpatialOperators.crossover(
                    p1.sensors, p2.sensors, self.env, avoid=set()
                )
                child_r = SpatialOperators.crossover(
                    p1.relays, p2.relays, self.env, avoid=set(map(tuple, child_s))
                )

                # Dynamic Mutation (Simplified for brevity, apply to both arrays)
                if np.random.rand() < 0.4:
                    child_s = SpatialOperators.move_sensor(child_s, self.env)
                if np.random.rand() < 0.4:
                    child_r = SpatialOperators.move_sensor(child_r, self.env)

                if len(child_s) > 0 and len(child_r) > 0:
                    child = DeploymentSolution(child_s, child_r)
                    self.evaluate(child)
                    offspring.append(child)

            merged_pop = pop + offspring
            merged_fronts = self._fast_non_dominated_sort(merged_pop)

            next_pop = []
            for front in merged_fronts:
                self._crowding_distance_assignment(front, merged_pop)
                if len(next_pop) + len(front) <= self.pop_size:
                    next_pop.extend([merged_pop[i] for i in front])
                else:
                    front.sort(
                        key=lambda x: merged_pop[x].crowding_distance, reverse=True
                    )
                    rem = self.pop_size - len(next_pop)
                    next_pop.extend([merged_pop[i] for i in front[:rem]])
                    break

            pop = next_pop

        return pop

    def _add_sensor(self, sensors: np.ndarray) -> np.ndarray:
        occupied = set(map(tuple, sensors))
        new_sensor = SpatialOperators.random_empty_cell(self.env, occupied)
        if len(sensors) > 0:
            return np.vstack([sensors, new_sensor])
        return np.array([new_sensor])

    def _remove_sensor(self, sensors: np.ndarray) -> np.ndarray:
        if len(sensors) <= 1:
            return sensors
        idx = np.random.randint(len(sensors))
        return np.delete(sensors, idx, axis=0)

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
