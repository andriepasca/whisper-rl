# Experiments & Pareto

WHISPER-RL provides a reproducible end-to-end experimental pipeline to evaluate Multi-Objective Reinforcement Learning (MORL) agents on the trade-off between structural fatigue ($J_{damage}$) and vessel spatial routing ($J_{spatial}$).

## Full Reproduction Guide

The experimental workflow consists of three distinct steps:

### 1. Training Agents

Train PPO agents across different scalarization weights for the Pareto front:

```bash
python scripts/train_ppo_pareto.py --timesteps 50000 --seed 42
```

### 2. Evaluating Benchmarks

Evaluate the trained agents using the benchmark script. This script records the multi-objective metrics for each agent:

```bash
python scripts/benchmark_pareto.py --seed 42
```

### 3. Generating Publication Plots

Generate 300 DPI, publication-ready plots of the Pareto front and agent performances:

```bash
python scripts/plot_publication.py --format both
```
