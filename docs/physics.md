# Physics & Mechanics

The WHISPER-RL framework relies on a mathematically grounded physics engine to simulate turbine blade dynamics and maintenance mechanics.

## Palmgren-Miner Composite Fatigue Mechanics

The accumulation of structural fatigue is modeled using the Palmgren-Miner linear damage hypothesis. Blade degradation is accumulated linearly over time, proportional to the wind loads and structural stress cycles.

## Stochastic Wind Sampling

For each simulated month, the environment draws `n_slices` (`wind_time_slices_per_month`, default 30) i.i.d. wind speeds from the month's Weibull distribution, truncated to [3, 25] m/s by rejection sampling, using the environment RNG (seeded via `reset(seed=...)`). The damage rate of each slice is evaluated through the response surface, the mean over slices estimates the expected 10-minute damage, and this is scaled to the whole month (and thus the decision interval). Damage is therefore stochastic, not deterministic; averaging over slices reduces, but does not eliminate, Jensen's-inequality sampling noise. Wind direction is not modeled: wake effects are neglected (PyWake was removed) and a fixed turbulence intensity of 0.10 is used.

## Leading Edge Protection (LEP) Repair Dynamics

Maintenance actions primarily involve the application of Leading Edge Protection (LEP). The application of LEP resets or shields the underlying composite material from further fatigue accumulation for a specific duration or until the protective layer degrades. The environment models the protective remaining life of LEP and its subsequent degradation.
