import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from .config import WindClimateConfig, RewardConfig
from .maintenance import MaintenanceResult

class WindClimate:
    """
    Models the stochastic ambient wind climate for the offshore environment.

    This class handles the probability distributions of wind speed and direction,
    providing methods for both random sampling and deterministic stratified sampling.
    """

    def __init__(self, config: WindClimateConfig):
        """
        Initializes the WindClimate model with the given configuration.

        Args:
            config (WindClimateConfig): Configuration containing Weibull parameters and
                                        wind direction probabilities.
        """
        self.config = config

        # --------------------------------------------------
        # Wind-direction probability
        # --------------------------------------------------
        if config.wind_direction_probability is None:

            # von Misses distribution
            dominant_direction = 270.0
            kappa = 2.0
            theta = np.deg2rad(config.wind_direction)
            mu = np.deg2rad(dominant_direction)
            wind_direction_probability = np.exp(kappa * np.cos(theta - mu))
            wind_direction_probability /= wind_direction_probability.sum()

            self.wind_direction_probability = wind_direction_probability

        else:

            probability = np.asarray(
                config.wind_direction_probability,
                dtype=float,
            )

            if len(probability) != len(config.wind_direction):
                raise ValueError(
                    "wind_direction_probability must have "
                    "the same length as wind_direction."
                )

            if np.any(probability < 0):
                raise ValueError(
                    "wind_direction_probability cannot contain "
                    "negative values."
                )

            probability_sum = probability.sum()

            if probability_sum <= 0:
                raise ValueError(
                    "wind_direction_probability must have "
                    "a positive sum."
                )

            # Normalize automatically
            self.wind_direction_probability = (
                probability / probability_sum
            )

    def sample(self, month, rng, size=None):
        """
        Stochastically samples ambient wind speed for a given month.

        Args:
            month (int): The current month (1-12) used to select Weibull parameters.
            rng (np.random.Generator): Random number generator instance.
            size (Optional[int]): Number of samples to draw. If None, returns scalars.

        Returns:
            If size is None, returns float for wind speed.
            If size is not None, returns np.ndarray of shape (size,).
        """

        # --------------------------------------------------
        # Ambient wind speed
        # --------------------------------------------------
        weibull = self.config.monthly_weibull[month]

        k = weibull["k"]
        c = weibull["c"]

        if size is None:
            while True:
                ambient_u = c * rng.weibull(k)
                if 3.0 <= ambient_u <= 25.0:
                    break

            return float(ambient_u)
        else:
            ambient_u = np.zeros(size)
            needed = np.ones(size, dtype=bool)
            
            while needed.any():
                n_needed = needed.sum()
                u_samples = c * rng.weibull(k, size=n_needed)
                valid = (u_samples >= 3.0) & (u_samples <= 25.0)
                
                valid_indices = np.where(needed)[0][valid]
                ambient_u[valid_indices] = u_samples[valid]
                needed[valid_indices] = False

            return ambient_u

@dataclass
class TransitionResult:
    HI: float
    damage_multiplier: float
    protection_remaining: int


class TransitionModel:
    """
    Turbine health transition model.

    This model computes the new health state of a turbine given its current state,
    accumulated fatigue damage, and any applied maintenance interventions.
    """

    def solve(
        self,
        hi: float,
        delta_damage: float,
        damage_multiplier: float,
        protection_remaining: int,
        maintenance: Optional[MaintenanceResult] = None,
    ) -> TransitionResult:
        """
        Computes the next state of the turbine health index.

        Args:
            hi (float): Current Health Index (0.0 to 1.0).
            delta_damage (float): Accumulated fatigue damage in the current time step.
            damage_multiplier (float): Current multiplier applied to damage.
            protection_remaining (int): Remaining months of protection.
            maintenance (Optional[MaintenanceResult]): Applied maintenance intervention.

        Returns:
            TransitionResult: The resulting health state of the turbine.
        """

        # ----------------------------------------------------------
        # Convert Health Index to degradation
        # ----------------------------------------------------------
        degradation_new = 1.0 - hi

        multiplier_new = damage_multiplier
        protection_new = protection_remaining

        # ----------------------------------------------------------
        # Apply maintenance at beginning of month
        # ----------------------------------------------------------
        if (
            maintenance is not None
            and maintenance.maintenance_type is not None
        ):
            maintenance_type = maintenance.maintenance_type

            # ------------------------------------------------------
            # Replacement restores the blade completely
            # ------------------------------------------------------
            if maintenance_type.is_replacement:
                degradation_new = 0.0

            # ------------------------------------------------------
            # Apply maintenance effect to future damage
            # ------------------------------------------------------
            multiplier_new = maintenance_type.damage_multiplier
            protection_new = maintenance_type.duration_months

        # ----------------------------------------------------------
        # Monthly fatigue accumulation
        # ----------------------------------------------------------
        degradation_new += (
            multiplier_new * delta_damage
        )

        degradation_new = min(
            1.0,
            degradation_new,
        )

        # ----------------------------------------------------------
        # Update protection duration
        # ----------------------------------------------------------
        if protection_new > 0:
            protection_new -= 1

            if protection_new == 0:
                multiplier_new = 1.0

        # ----------------------------------------------------------
        # Convert back to Health Index
        # ----------------------------------------------------------
        hi_new = 1.0 - degradation_new

        return TransitionResult(
            HI=hi_new,
            damage_multiplier=multiplier_new,
            protection_remaining=protection_new,
        )

class SpatialGroupingObjective:
    """
    Spatial grouping objective based on the layout of selected maintenance turbines.

    Lower values indicate a more spatially compact maintenance group, which is
    often desirable to minimize vessel routing and logistical costs.
    """

    def __init__(self, layout_config=None):
        """
        Initializes the spatial grouping objective.

        Args:
            layout_config (Optional[LayoutConfig]): The wind farm layout configuration.
        """
        if layout_config is None:
            from .config import LayoutConfig
            layout_config = LayoutConfig()

        self.D = getattr(layout_config, "D", 126.0)

        self.x = np.asarray(layout_config.x, dtype=float)
        self.y = np.asarray(layout_config.y, dtype=float)

        if len(self.x) != len(self.y):
            raise ValueError(
                "layout_config.x and layout_config.y must have the same length."
            )

        self.n_turbines = len(self.x)

        dx = self.x[:, None] - self.x[None, :]
        dy = self.y[:, None] - self.y[None, :]
        distances = np.sqrt(dx**2 + dy**2)
        self.max_distance = float(np.max(distances))
        if self.max_distance == 0.0:
            self.max_distance = 1.0

    def solve(self, maintenance_indices):
        """
        Calculates the spatial grouping metric for the selected turbines.

        Args:
            maintenance_indices (List[int]): Indices of turbines undergoing maintenance.

        Returns:
            float: The mean pairwise distance between selected turbines plus a base mobilization distance,
                   or 0.0 if no turbines are selected.
        """
        indices = np.asarray(
            maintenance_indices,
            dtype=int,
        )

        if indices.ndim != 1:
            raise ValueError(
                "maintenance_indices must be one-dimensional."
            )

        if len(indices) == 0:
            return 0.0

        if np.any(indices < 0) or np.any(indices >= self.n_turbines):
            raise IndexError(
                "maintenance_indices contains an invalid turbine index."
            )

        if len(np.unique(indices)) != len(indices):
            raise ValueError(
                "maintenance_indices must contain unique turbine indices."
            )

        base_distance = self.D * 2.0

        if len(indices) == 1:
            return float(base_distance)

        selected_x = self.x[indices]
        selected_y = self.y[indices]

        dx = selected_x[:, None] - selected_x[None, :]
        dy = selected_y[:, None] - selected_y[None, :]

        distances = np.sqrt(dx**2 + dy**2)

        upper = np.triu_indices(
            len(indices),
            k=1,
        )

        return float(base_distance + np.mean(distances[upper]))

@dataclass(frozen=True)
class RewardResult:
    reward: float
    objectives: dict[str, float]

class RewardModel:
    """
    Converts multi-objective maintenance metrics into an RL-compatible scalar reward.
    """

    def __init__(
        self,
        config: RewardConfig,
    ):
        """
        Initializes the RewardModel.

        Args:
            config (RewardConfig): Configuration containing reward objectives and weights.
        """
        self.config = config

    def solve(
        self,
        metrics: dict[str, float],
    ) -> RewardResult:
        """
        Computes the scalar reward from the given metrics.

        Args:
            metrics (dict[str, float]): Dictionary of evaluated metrics (e.g., damage burden).

        Returns:
            RewardResult: A dataclass containing the final scalar reward and individual objectives.
        """

        reward = 0.0
        objectives = {}

        for name, cfg in self.config.objectives.items():

            if name not in metrics:
                raise KeyError(
                    f"Objective metric '{name}' not found."
                )

            value = float(metrics[name])

            objectives[name] = value

            if cfg.normalize and cfg.scale > 0:
                value = value / cfg.scale

            if cfg.direction == "min":
                contribution = -value

            elif cfg.direction == "max":
                contribution = value

            else:
                raise ValueError(
                    f"Unknown objective direction: "
                    f"{cfg.direction}. "
                    f"Use 'min' or 'max'."
                )

            reward += (
                cfg.weight
                * contribution
            )

        return RewardResult(
            reward=reward,
            objectives=objectives,
        )

@dataclass
class LoggingModel:
    """
    Centralized simulation logger for tracking state transitions and decisions.
    """

    history: List[Dict[str, Any]] = field(default_factory=list)

    def reset(self):
        """Clears the simulation history."""
        self.history.clear()

    def log_state(
        self,
        *,
        year,
        month,
        elapsed_month,
        HI,
        power,
        ambient_u,
        u_eff,
        ti_eff,
        repair_params: Dict[str, Any],
        repair_count,
        replacement_count
    ):
        """
        Logs the state of the environment at a given time step.
        """

        self.history.append(
            {
                "elapsed_month": int(elapsed_month),
                "year": int(year),
                "month": int(month),
                "HI_mean": float(np.mean(HI)),
                "HI_min": float(np.min(HI)),
                "HI_max": float(np.max(HI)),
                "HI_turbine": HI.copy().tolist(),
                "ambient_u": float(ambient_u),
                "u_eff": u_eff.copy().tolist(),
                "ti_eff": ti_eff.copy().tolist(),
                "power": power.copy().tolist(),
                "repair_params": {
                    "threshold": float(repair_params["threshold"]),
                    "multiplier": float(repair_params["multiplier"]),
                    "protection_duration": int(
                        repair_params["duration"]
                    ),
                },
                "repair_count": repair_count.copy().tolist(),
                "replacement_count": replacement_count.copy().tolist(),
                "decision_event": False,
                "action": None,
                "maintenance_type": None,
                "reward": None,
                "interval_damage": None,
                "interval_damage_burden": None,
                "spatial_grouping": None,
                "protection_remaining": None,
            }
        )

    def log_decision(
        self,
        *,
        reward,
        objectives,
        interval_damage,
        interval_damage_burden,
        spatial_grouping,
        maintenance_results,
        action,
        protection_remaining,
    ):
        """
        Logs the decision and resulting metrics at an intervention step.
        """
        if len(self.history) == 0:
            return

        row = self.history[-1]
        row["decision_event"] = True
        row["action"] = action.copy().tolist()
        row["interval_damage_burden"] = float(interval_damage_burden)
        row["spatial_grouping"] = float(spatial_grouping)
        row["interval_damage"] = float(interval_damage)
        row["reward"] = float(reward)
        row["maintenance_type"] = [
            None
            if m.maintenance_type is None
            else m.maintenance_type.name.title()
            for m in maintenance_results
        ]
        row["protection_remaining"] = protection_remaining.copy().tolist()

        for k, v in objectives.items():
            row[k] = float(v)

    def dataframe(self):
        """
        Returns the logged history as a pandas DataFrame.
        """
        return pd.DataFrame(self.history)

class RandomScatteredLayout:
    """
    Generate a reproducible random-scattered wind farm layout
    with a minimum center-to-center turbine spacing constraint.
    """

    def __init__(
        self,
        D=126.0,
        n_turbines=25,
        min_spacing_D=7.0,
        farm_scale=1.5,
        seed=42,
        max_attempts=1_000_000
    ):
        """
        Initializes the random scattered layout generator.

        Args:
            D (float): Rotor diameter in meters.
            n_turbines (int): Number of turbines to place.
            min_spacing_D (float): Minimum spacing between turbines in rotor diameters.
            farm_scale (float): Scaling factor for the overall farm area.
            seed (int): Random seed for reproducibility.
            max_attempts (int): Maximum number of placement attempts.
        """

        self.D = D
        self.n_turbines = n_turbines
        self.min_spacing_D = min_spacing_D
        self.min_spacing = min_spacing_D * D

        self.farm_scale = farm_scale
        self.seed = seed
        self.max_attempts = max_attempts

        self.farm_size = (
            farm_scale
            * min_spacing_D
            * self.min_spacing
        )

        self.x_min = 0.0
        self.x_max = self.farm_size
        self.y_min = 0.0
        self.y_max = self.farm_size

        self.coordinates = self._generate()
        self.x = self.coordinates[:, 0]
        self.y = self.coordinates[:, 1]
        self.info = self._calculate_info()

    def _generate(self):
        rng = np.random.default_rng(self.seed)
        coordinates = []
        attempt = 0

        while (
            len(coordinates) < self.n_turbines
            and attempt < self.max_attempts
        ):
            attempt += 1
            candidate = np.array([
                rng.uniform(self.x_min, self.x_max),
                rng.uniform(self.y_min, self.y_max)
            ])

            if len(coordinates) == 0:
                coordinates.append(candidate)
                continue

            distances = np.linalg.norm(
                np.asarray(coordinates) - candidate,
                axis=1
            )

            if np.all(distances >= self.min_spacing):
                coordinates.append(candidate)

        if len(coordinates) < self.n_turbines:
            raise RuntimeError(
                f"Could only place {len(coordinates)} "
                f"out of {self.n_turbines} turbines "
                f"after {self.max_attempts:,} attempts."
            )

        coordinates = np.asarray(coordinates)
        coordinates -= coordinates.mean(axis=0)

        return coordinates

    def _calculate_info(self):
        """
        Calculates geometric and spatial info about the generated layout.
        """
        if self.n_turbines == 1:
            return {
                "number_of_turbines": 1,
                "rotor_diameter_m": self.D,
                "minimum_spacing_m": 0.0,
                "minimum_spacing_D": 0.0,
                "mean_pair_distance_m": 0.0,
                "farm_area_m2": 0.0
            }

        pairwise_distances = []
        for i in range(self.n_turbines):
            for j in range(i + 1, self.n_turbines):
                distance = np.linalg.norm(
                    self.coordinates[i]
                    - self.coordinates[j]
                )
                pairwise_distances.append(distance)
        pairwise_distances = np.asarray(pairwise_distances)

        width = self.x.max() - self.x.min()
        height = self.y.max() - self.y.min()
        area = width * height
        area_per_turbine = area / self.n_turbines

        return {
            "number_of_turbines": self.n_turbines,
            "rotor_diameter_m": self.D,
            "minimum_spacing_m": pairwise_distances.min(),
            "minimum_spacing_D": pairwise_distances.min() / self.D,
            "mean_pair_distance_m": pairwise_distances.mean(),
            "farm_width_m": width,
            "farm_height_m": height,
            "farm_area_km2": area / 1e6,
            "area_per_turbine_km2": area_per_turbine / 1e6,
            "seed": self.seed
        }

    def print_info(self):
        """
        Prints information about the generated layout.
        """
        info = self.info
        print(f"Number of turbines : {info['number_of_turbines']}")
        print(f"Rotor diameter     : {info['rotor_diameter_m']:.1f} m")
        print(f"Minimum spacing    : {info['minimum_spacing_m']:.2f} m")
        print(f"Minimum spacing    : {info['minimum_spacing_D']:.2f}D")
        print(f"Mean pair distance : {info['mean_pair_distance_m']:.2f} m")
        print(f"Wind farm area     : {info['farm_area_km2']:.2f} km²")
        print(f"Area per turbine   : {info['area_per_turbine_km2']:.2f} km²/turbine")
        print(f"Random seed        : {info['seed']}")

    def plot(self, turbine_labels=True):
        """
        Plots the generated layout.

        Args:
            turbine_labels (bool): Whether to label the turbines.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(self.x, self.y, s=100)
        if turbine_labels:
            for i, (x, y) in enumerate(zip(self.x, self.y), start=1):
                ax.text(x + 20, y + 20, f"T{i}", fontsize=8)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Random Scattered Layout (minimum spacing = {self.min_spacing_D:.0f}D)")
        ax.set_aspect("equal")
        ax.grid(True)
        plt.show()
