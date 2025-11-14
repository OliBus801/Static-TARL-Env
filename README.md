# Static-TARL-Env

**Static-TARL-Env** is a lightweight experimental environment designed to evaluate reinforcement learning (RL) algorithms on their ability to solve the **Static Assignment Problem (SAP) of Traffic** under the **System Equilibrium** formulation proposed by Warwick.

It provides a standardized framework to:

* run static traffic assignment simulations;
* store and process simulation outputs;
* evaluate the performance of RL methods on this problem.

---

## 🚀 Goals

* Provide an environment compatible with RL frameworks to simplify the evaluation of RL algorithms on the Static Traffic Assignment Problem.
* Facilitate fair comparison between different approaches.

---

## ⚡ Quick Usage

```python
import numpy as np
import torch

from simulation import build_env_from_json

# Instantiate the environment from JSON definitions
_, _, env = build_env_from_json(
    "examples/networks/small_network.json",
    "examples/demands/small_demand.json",
    k_paths=3,
)

observation, info = env.reset(seed=0)

# Build a dummy action (logits per path) that keeps all paths equally likely
dummy_action = np.zeros(env.action_space.shape, dtype=np.float32)
observation, reward, terminated, truncated, info = env.step(dummy_action)
```

The snippet mirrors the flow exercised in `pytest tests/test_env.py`, ensuring that the documented usage corresponds to a valid execution path.

---

## 📁 JSON Assumptions

* **Network files** must expose two top-level arrays, `nodes` and `edges`. Each node entry requires an integer `id` and may optionally provide `x`/`y` coordinates (defaults to `0.0`). Each edge requires `source` and `target` ids that reference the declared nodes, plus capacity descriptors: `capacity` (vehicles/hour) and `freeflow_travel_time` (time units). Missing nodes, edges, or required attributes raise `ValueError`s when calling `TapScenario.load_network_from_json`.
* **Demand files** must contain a `matrix` field describing the OD matrix as a rectangular list of lists. The loader converts this field to a NumPy array and derives the total number of agents from its sum.

These structures allow `build_env_from_json` to construct the `TapScenario`, derive a `PathSet`, and feed consistent tensors into `StaticTapEnv`.

---
