# API Reference

## OffshoreMaintenanceEnv

The primary Gymnasium environment class representing the offshore wind farm maintenance scenario.
- **`step(action)`**: Advances the simulation by one month.
- **`reset()`**: Resets the environment to its initial state.

## EnvironmentConfig

The configuration dataclass, customizable via `get_default_config()`, defining the environment's parameters (e.g., `n_turbines`, multi-objective weights).

## WeatherOracle

Provides maintenance window accessibility probability hooks. Can be used as a statistical weather predictor for top-level Hierarchical RL (HRL) agents.

## TrajectoryLogger

Logs simulation metrics and events over time, useful for benchmarking the multi-objective performances such as $J_{damage}$ and $J_{spatial}$.
