# Limitations

The WHISPER-RL framework is built with explicit modeling assumptions and technical limitations to ensure scientific objectivity and transparency.

## Linear Scalarization Convexity Assumption

The multi-objective formulation employs linear scalarization to balance blade degradation ($J_{damage}$) and spatial logistics ($J_{spatial}$). This approach fundamentally assumes a convex Pareto front. Non-convex regions of the Pareto front cannot be fully explored or resolved using this method alone.

## Euclidean Proxy Nature of Vessel Distances

Vessel spatial routing compactness ($J_{spatial}$) is calculated using a Euclidean distance proxy between serviced turbines. This is a simplification that abstracts away realistic maritime pathfinding, wake-induced navigational constraints, and graph-based logistical routing limits.

## Sampling Noise of Stochastic Wind Slices

Damage is estimated from `n_slices` (default 30) i.i.d. random Weibull wind draws per month, so it carries Monte Carlo sampling noise; increasing `wind_time_slices_per_month` reduces it at higher compute cost. Wind direction is not modeled and wake effects are neglected.
