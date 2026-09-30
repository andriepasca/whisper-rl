# Physics & Mechanics

The WHISPER-RL framework relies on a mathematically grounded physics engine to simulate turbine blade dynamics and maintenance mechanics.

## Palmgren-Miner Composite Fatigue Mechanics

The accumulation of structural fatigue is modeled using the Palmgren-Miner linear damage hypothesis. Blade degradation is accumulated linearly over time, proportional to the wind loads and structural stress cycles.

## Deterministic Stratified Sampling

To eliminate the bias introduced by Jensen's inequality when aggregating non-linear fatigue over varying wind conditions, WHISPER-RL uses deterministic joint-probability Stratified Sampling. This Quasi-Monte Carlo numerical scheme ($m \approx 10$) strictly avoids random sampling to guarantee mathematical stability and ensure that accumulated damage and power calculations remain deterministic and unbiased.

## Leading Edge Protection (LEP) Repair Dynamics

Maintenance actions primarily involve the application of Leading Edge Protection (LEP). The application of LEP resets or shields the underlying composite material from further fatigue accumulation for a specific duration or until the protective layer degrades. The environment models the protective remaining life of LEP and its subsequent degradation.
