# API Reference

## OffshoreMaintenanceEnv

The primary Gymnasium environment class representing the offshore wind farm maintenance scenario.
- **`step(action)`**: Advances the simulation by one decision interval (`decision_interval_months`, default 6 months; each month is simulated with stochastic wind sampling).
- **`reset()`**: Resets the environment to its initial state.

## EnvironmentConfig

The configuration dataclass, customizable via `get_default_config()`, defining the environment's parameters (e.g., `n_turbines`, multi-objective weights).

## WeatherOracle

Provides maintenance window accessibility probability hooks. Can be used as a statistical weather predictor for top-level Hierarchical RL (HRL) agents.

## LoggingModel

Simulation logger (`whisper_env.LoggingModel`, enabled with `enable_logging=True`; `history` / `dataframe()`). Logs per-month states and per-decision metrics, useful for benchmarking the multi-objective performances such as $J_{damage}$ and $J_{spatial}$. (There is no `TrajectoryLogger` class.)

## WindClimate

Stochastic per-month Weibull wind-speed sampler (i.i.d. draws truncated to [3, 25] m/s via the env RNG). Wind direction is not modeled (wake neglected; PyWake removed).
