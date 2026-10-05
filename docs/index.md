# Welcome to WHISPER-RL

**WHISPER-RL** (Wind turbine Health-Index & Spatial Policy Environment for Reinforcement Learning) is a modular Gymnasium environment for modeling Offshore Wind Turbine Blade Maintenance.

## The Green Paradox

The core scientific scope of WHISPER-RL is addressing the offshore wind "Green Paradox": the need to balance composite blade structural fatigue (a Health-Index proxy) against the carbon-intensive marine vessel logistics (Spatial Policy) required to perform maintenance. While extensive maintenance maximizes energy capture and minimizes turbine fatigue, the logistical overhead of marine vessels introduces significant carbon emissions and spatial routing costs.

## Architectural Design

WHISPER-RL is structured to provide a robust, scientifically grounded framework for multi-objective Reinforcement Learning (MORL). It incorporates:

- **Stochastic Wind Sampling:** Each simulated month draws `n_slices` (default 30) i.i.d. Weibull wind speeds (truncated to [3, 25] m/s) via the environment RNG; expected damage is estimated from the slice mean and scaled to the decision interval. Wind direction is not modeled (wake neglected; PyWake removed).
- **Multi-Objective Formulation:** Dimensionless trade-off between blade degradation ($J_{damage}$) and spatial vessel routing compactness ($J_{spatial}$).
- **Pluggable Climate Presets:** Support for distinct operating conditions such as the North Sea, US Atlantic, and Taiwan Strait.
- **Weather Oracle:** Maintenance window accessibility probability hooks designed for future Hierarchical RL (HRL) expansion.
