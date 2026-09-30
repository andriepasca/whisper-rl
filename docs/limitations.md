# Limitations

The WHISPER-RL framework is built with explicit modeling assumptions and technical limitations to ensure scientific objectivity and transparency.

## Linear Scalarization Convexity Assumption

The multi-objective formulation employs linear scalarization to balance blade degradation ($J_{damage}$) and spatial logistics ($J_{spatial}$). This approach fundamentally assumes a convex Pareto front. Non-convex regions of the Pareto front cannot be fully explored or resolved using this method alone.

## Euclidean Proxy Nature of Vessel Distances

Vessel spatial routing compactness ($J_{spatial}$) is calculated using a Euclidean distance proxy between serviced turbines. This is a simplification that abstracts away realistic maritime pathfinding, wake-induced navigational constraints, and graph-based logistical routing limits.

## Computational Trade-off of Numerical Stratification

The use of a deterministic Stratified Sampling scheme (Quasi-Monte Carlo) ensures numerical stability and eliminates Jensen's inequality bias. However, this mathematical rigor comes at a computational cost, leading to slower step execution times compared to environments that use purely random sampling approximations.
