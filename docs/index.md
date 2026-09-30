# Welcome to WHISPER-RL

**WHISPER-RL** (Wind turbine Health-Index & Spatial Policy Environment for Reinforcement Learning) is a modular Gymnasium environment for modeling Offshore Wind Turbine Blade Maintenance.

## The Green Paradox

The core scientific scope of WHISPER-RL is addressing the offshore wind "Green Paradox": the need to balance composite blade structural fatigue (a Health-Index proxy) against the carbon-intensive marine vessel logistics (Spatial Policy) required to perform maintenance. While extensive maintenance maximizes energy capture and minimizes turbine fatigue, the logistical overhead of marine vessels introduces significant carbon emissions and spatial routing costs.

## Architectural Design

WHISPER-RL is structured to provide a robust, scientifically grounded framework for multi-objective Reinforcement Learning (MORL). It incorporates:

- **Deterministic Stratified Sampling:** A Quasi-Monte Carlo numerical scheme ($m \approx 10$) solving Jensen's inequality bias in blade fatigue.
- **Multi-Objective Formulation:** Dimensionless trade-off between blade degradation ($J_{damage}$) and spatial vessel routing compactness ($J_{spatial}$).
- **Pluggable Climate Presets:** Support for distinct operating conditions such as the North Sea, US Atlantic, and Taiwan Strait.
- **Weather Oracle:** Maintenance window accessibility probability hooks designed for future Hierarchical RL (HRL) expansion.
