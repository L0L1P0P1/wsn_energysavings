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
        return np.array(
            [
                -self.coverage_rate,
                -self.reliability,
                self.cost,
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

        # Map spatial coordinates to their exact indices in the precomputed matrices
        s_idx = np.argmin(
            np.linalg.norm(
                sol.sensors[:, None, :] - self.env.valid_cells[None, :, :], axis=2
            ),
            axis=1,
        )
        r_idx = np.argmin(
            np.linalg.norm(
                sol.relays[:, None, :] - self.env.valid_cells[None, :, :], axis=2
            ),
            axis=1,
        )
        bs_idx = np.argmin(np.linalg.norm(self.bs_pos - self.env.valid_cells, axis=1))

        # 2. Coverage Objective (Eq 11)
        eff_dists = self.env.dist_mat[s_idx, :] / np.where(
            self.env.col_mat[s_idx, :] > 0, 1e-9, 1.0
        )
        sol.coverage_rate = (
            np.sum(np.any(eff_dists <= self.radius, axis=0))
            / self.env.dist_mat.shape[1]
        )

        # 3. Graph Connectivity Constraints & Reliability (Eq 12, 13, 22-25)
        # Check Relays -> Base Station using O(1) Matrix Lookups
        dist_r_bs = self.env.dist_mat[r_idx, bs_idx]
        cols_r_bs = self.env.col_mat[r_idx, bs_idx]

        powers_r_bs = self.physics.received_power(dist_r_bs, cols_r_bs)
        prob_r_bs = self.physics.connection_probability(powers_r_bs)
        disconnected_relays = np.sum(prob_r_bs < self.connection_threshold)

        # Check Sensors -> Relays using O(1) Matrix Lookups
        dist_s_r = self.env.dist_mat[s_idx[:, None], r_idx]
        cols_s_r = self.env.col_mat[s_idx[:, None], r_idx]

        powers_s_r = self.physics.received_power(dist_s_r, cols_s_r)
        prob_s_r = self.physics.connection_probability(powers_s_r)

        # Link matrix (1 if connected, 0 if not)
        links = (prob_s_r >= self.connection_threshold).astype(int)
        k_j = np.sum(links, axis=1)  # Links per sensor (Eq 22)

        disconnected_sensors = np.sum(k_j == 0)

        # Apply strict constraint penalties
        sol.penalty = float(disconnected_relays + disconnected_sensors)

        # Objective 3: Reliability Calculation (Eq 23, 24, 25)
        if len(k_j) > 0 and disconnected_sensors == 0:
            k_mean = np.mean(k_j)
            k_var = np.var(k_j)
            sol.reliability = k_mean - k_var
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

    def run(self, init_s: int, init_r: int) -> list[DeploymentSolution]:
        pop = []
        for _ in range(self.pop_size):
            # USE SMART INITIALIZATION HERE
            sol = self._smart_initialize(init_s, init_r)
            pop.append(sol)

        for ind in pop:
            self.evaluate(ind)

        for gen in range(self.gens):
            self._fast_non_dominated_sort(pop)

            offspring = []
            while len(offspring) < self.pop_size:
                # Tournament Selection
                t1, t2 = np.random.choice(len(pop), 2, replace=False)
                p1 = self._tournament(pop[t1], pop[t2])

                t3, t4 = np.random.choice(len(pop), 2, replace=False)
                p2 = self._tournament(pop[t3], pop[t4])

                # Crossover (Independently for sensors and relays)
                child_s = SpatialOperators.crossover(
                    p1.sensors, p2.sensors, self.env, avoid=set()
                )
                child_r = SpatialOperators.crossover(
                    p1.relays, p2.relays, self.env, avoid=set(map(tuple, child_s))
                )

                # ---------------------------------------------------------
                # Dynamic Structural Mutation (Add, Remove, Move)
                # ---------------------------------------------------------

                # Mutate Sensor Tier
                mut_prob_s = np.random.rand()
                if mut_prob_s < 0.2:
                    child_s = self._add_sensor(child_s)
                elif mut_prob_s < 0.4:
                    child_s = self._remove_sensor(child_s)
                elif mut_prob_s < 0.7:
                    child_s = SpatialOperators.move_sensor(child_s, self.env)

                # Mutate Relay Tier
                mut_prob_r = np.random.rand()
                if mut_prob_r < 0.2:
                    child_r = self._add_sensor(child_r)
                elif mut_prob_r < 0.4:
                    child_r = self._remove_sensor(child_r)
                elif mut_prob_r < 0.7:
                    child_r = SpatialOperators.move_sensor(child_r, self.env)

                # Ensure we don't accidentally evaluate empty network topologies
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

    def _dominates(self, p: DeploymentSolution, q: DeploymentSolution) -> bool:
        """Applies Deb's Constrained Domination Rules."""

        if p.penalty < q.penalty:
            return True
        if p.penalty > q.penalty:
            return False

        return bool(
            np.all(p.objectives <= q.objectives) and np.any(p.objectives < q.objectives)
        )

    def _tournament(
        self, p1: DeploymentSolution, p2: DeploymentSolution
    ) -> DeploymentSolution:
        """Selects the best parent respecting feasibility (penalty), rank, and crowding."""
        # 1. Feasibility rules (lower penalty always wins)
        if p1.penalty < p2.penalty:
            return p1
        if p2.penalty < p1.penalty:
            return p2

        # 2. If equally feasible (or equally infeasible), check Pareto rank
        if p1.rank < p2.rank:
            return p1
        if p2.rank < p1.rank:
            return p2

        # 3. If same rank, prefer less crowded space for better diversity
        return p1 if p1.crowding_distance > p2.crowding_distance else p2

    def _smart_initialize(self, init_s: int, init_r: int) -> DeploymentSolution:
        """Deterministically spawns a fully connected topology to bypass the Constraint Cliff."""
        # 1. Find Base Station Index
        bs_idx = np.argmin(np.linalg.norm(self.bs_pos - self.env.valid_cells, axis=1))

        # 2. Find Valid Relay Cells (must have prob >= threshold to BS)
        dist_to_bs = self.env.dist_mat[:, bs_idx]
        cols_to_bs = self.env.col_mat[:, bs_idx]
        prob_to_bs = self.physics.connection_probability(
            self.physics.received_power(dist_to_bs, cols_to_bs)
        )

        valid_r_idx = np.where(prob_to_bs >= self.connection_threshold)[0]
        if len(valid_r_idx) < init_r:
            if len(valid_r_idx) == 0:
                raise ValueError(
                    "CRITICAL: Base Station is completely isolated. Move it or lower the threshold."
                )
            chosen_r_idx = np.random.choice(
                valid_r_idx, size=len(valid_r_idx), replace=False
            )
        else:
            chosen_r_idx = np.random.choice(valid_r_idx, size=init_r, replace=False)

        relays = self.env.valid_cells[chosen_r_idx]

        # 3. Find Valid Sensor Cells (must have prob >= threshold to ANY chosen Relay)
        dist_to_relays = self.env.dist_mat[:, chosen_r_idx]
        cols_to_relays = self.env.col_mat[:, chosen_r_idx]
        prob_to_relays = self.physics.connection_probability(
            self.physics.received_power(dist_to_relays, cols_to_relays)
        )

        # Get the best connection probability to any of the placed relays
        max_prob_to_r = np.max(prob_to_relays, axis=1)
        valid_s_idx = np.where(max_prob_to_r >= self.connection_threshold)[0]

        # Prevent sensors from spawning exactly on top of relays
        valid_s_idx = np.setdiff1d(valid_s_idx, chosen_r_idx)

        if len(valid_s_idx) < init_s:
            chosen_s_idx = np.random.choice(
                valid_s_idx, size=len(valid_s_idx), replace=False
            )
        else:
            chosen_s_idx = np.random.choice(valid_s_idx, size=init_s, replace=False)

        sensors = self.env.valid_cells[chosen_s_idx]

        return DeploymentSolution(sensors, relays)

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

    def _fast_non_dominated_sort(
        self, population: list[DeploymentSolution]
    ) -> list[list[int]]:
        N = len(population)
        S = [[] for _ in range(N)]
        fronts = [[]]
        n = np.zeros(N, dtype=int)

        for p in range(N):
            for q in range(N):
                if self._dominates(population[p], population[q]):
                    S[p].append(q)
                elif self._dominates(population[q], population[p]):
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
