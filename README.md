# 📡 Multi-Objective WSN Deployment Optimizer

A Python-based optimization engine for deploying indoor two-tier Wireless Sensor Networks (WSNs). This project uses evolutionary algorithms to discover optimal network topologies by balancing **Cost**, **Coverage Rate**, and network **Reliability** across complex indoor environments with physical obstacles.

## ✨ Key Features

* **Two-Tier Architecture:** Evaluates edge sensors and relay (sink) nodes routing data back to a central Base Station.
* **Physics-Aware:** Utilizes a log-normal shadowing propagation model to calculate connection probabilities, heavily penalizing signals that pass through walls/doors.
* **Vectorized Spatial Lookups:** Bypasses slow line-segment intersection loops by mapping dynamic coordinates to precomputed $\mathcal{O}(1)$ collision and distance matrices.
* **NSGA-II Engine:** Maps the 3D Pareto front of conflicting objectives using Deb's Constrained Domination rules to strictly enforce fully connected graph topologies.
* **PoC Baselines:** Includes lightweight Simulated Annealing (SA) and Genetic Algorithm (GA) engines for single-tier, single-objective (Coverage) benchmarking.

## 🗂️ Project Structure

* `src/core.py`: Contains the `IndoorEnvironment` (spatial matrix handling) and `PropagationModel` (physics and signal attenuation).
* `src/optimizer/nsga2.py`: The multi-objective NSGA-II implementation.
* `src/optimizer/baselines.py`: Single-objective `SimulatedAnnealing` and `GeneticAlgorithm` classes.
* `src/optimizer/operators.py`: The `SpatialOperators` class containing geometry-aware mutations (move, add, remove, crossover).

## 🚀 Quick Start

Here is a minimum reproducible example to run the 3D multi-objective optimizer and generate a Pareto front of valid network layouts.

```python
import numpy as np
from src.core import IndoorEnvironment, PropagationModel
from src.optimizer.nsga2 import NSGA2

# 1. Initialize Environment and Physics
env = IndoorEnvironment(...) # Load your room layout/matrices here
physics = PropagationModel(...)

# 2. Define Base Station Position
base_station_pos = np.array([2, 2])

# 3. Configure the NSGA-II Optimizer
optimizer = NSGA2(
    env=env, 
    physics=physics, 
    bs_pos=base_station_pos, 
    radius=3.0, 
    pop_size=20, 
    gens=500
)

# 4. Tune the physics thresholds (0.0 to 1.0)
optimizer.connection_threshold = 0.3

# 5. Run the optimizer (Initialize with 10 sensors, 4 relays)
pareto_population = optimizer.run(init_s=10, init_r=4)

# 6. Filter for mathematically valid, fully-connected networks
valid_layouts = [sol for sol in pareto_population if sol.penalty == 0]

print(f"Discovered {len(valid_layouts)} valid layouts.")
for idx, sol in enumerate(valid_layouts):
    print(f"Layout {idx}: Cost={sol.cost}, Coverage={sol.coverage_rate:.2f}, Reliability={sol.reliability:.2f}")

```

## 🧠 Optimization Objectives

1. **Minimize Cost:** Calculates infrastructure cost based on the number of deployed nodes (Sensors cost 1x, Relays cost 5x).
2. **Maximize Coverage:** The percentage of valid room space falling within the `radius` of at least one sensor with an unblocked line-of-sight.
3. **Maximize Reliability:** For fully connected networks, this calculates the mean minus the variance of the sensor-to-relay communication links, rewarding networks that provide redundant, balanced routing paths.
