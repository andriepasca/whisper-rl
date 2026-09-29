%%capture
#pip install gymnasium
!pip install py_wake

import numpy as np
import gymnasium as gym
import pandas as pd
import xarray as xr

from dataclasses import dataclass, field, asdict

from scipy.interpolate import RegularGridInterpolator

from py_wake.site import XRSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_farm_models import PropagateDownwind

from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation

from typing import Callable, Dict, List, Optional, Any, Literal

def minmax_normalize(x, xmin, xmax):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, xmin, xmax)
    return (x - xmin) / (xmax - xmin)

def cyclic_encode(x, period):
    angle = 2 * np.pi * x / period
    return np.array(
        [np.sin(angle), np.cos(angle)],
        dtype=np.float32
    )

@dataclass
class WindClimateConfig:
    monthly_weibull: dict
    wind_direction: np.ndarray
    wind_direction_probability: Optional[np.ndarray] = None

class WindClimate:

    def __init__(self, config):
        self.config = config

        # --------------------------------------------------
        # Wind-direction probability
        # --------------------------------------------------
        if config.wind_direction_probability is None:

            # Default: uniform synthetic wind rose
            self.wind_direction_probability = (
                np.ones(len(config.wind_direction))
                / len(config.wind_direction)
            )

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

    def sample(self, month, rng):

        # --------------------------------------------------
        # Ambient wind speed
        # --------------------------------------------------
        weibull = self.config.monthly_weibull[month]

        k = weibull["k"]
        c = weibull["c"]

        while True:

            ambient_u = c * rng.weibull(k)

            if 3.0 <= ambient_u <= 25.0:
                break

        # --------------------------------------------------
        # Ambient wind direction
        # --------------------------------------------------
        ambient_wd = rng.choice(
            self.config.wind_direction,
            p=self.wind_direction_probability,
        )

        return (
            float(ambient_u),
            float(ambient_wd),
        )

@dataclass
class LayoutConfig:
    x: np.ndarray
    y: np.ndarray

@dataclass
class TurbineConfig:
    name: str
    rotor_diameter: float
    hub_height: float
    csv_path: str # source from https://github.com/NatLabRockies/turbine-models$0
    power_unit: str = "kW"

@dataclass
class WakeSolverConfig:
    layout: LayoutConfig
    turbine: TurbineConfig
    wake_deficit_model: Any
    superposition_model: Any = None
    turbulence_model: Any = None

class WakeSolver:
    def __init__(self, config):
        self.config = config
        self.layout = config.layout
        self.layout_x = config.layout.x
        self.layout_y = config.layout.y
        self.n_turbines = len(self.layout_x)

        self.turbine_config = config.turbine
        self.turbine_df = pd.read_csv(self.turbine_config.csv_path)

        self.wake_deficit_model = config.wake_deficit_model
        self.superposition_model = config.superposition_model
        self.turbulence_model = config.turbulence_model

        self.wind_turbines = None
        self.wind_farm_model = None

        self._build_turbine()
        self._build_site()
        self._build_model()

    @staticmethod
    def ambient_ti(ws, z=90.0):
        # ============================================================
        # Ambient turbulence intensity
        #
        # Extended ISO turbulence-intensity relationship
        # (Jeans, 2024), evaluated using the ENOW neutral
        # offshore coefficient set:
        # Reference: https://doi.org/10.5194/wes-9-2001-2024
        #
        # ============================================================

        # ENOW neutral offshore coefficient set
        a1 = 0.025
        a2 = 0.0450
        a3 = 0.030

        # Reference quantities
        U_ref = 10.0   # [m/s]
        z_ref = 10.0   # [m]

        ws = np.asarray(ws, dtype=float)

        # Normalized wind speed
        U_norm = ws / U_ref

        # Ambient turbulence intensity
        ti = (
            a1 * U_norm
            + a2
            + a3 * U_norm**(-1)
        ) * (z / z_ref)**(-0.22)

        return ti

    def _build_turbine(self):
        turbine_df = self.turbine_df

        power_ct = PowerCtTabular(
            ws=turbine_df.iloc[:, 0].to_numpy(dtype=float),
            power=turbine_df.iloc[:, 1].to_numpy(dtype=float),
            power_unit=self.turbine_config.power_unit,
            ct=turbine_df.iloc[:, 4].to_numpy(dtype=float)
        )

        self.wind_turbines = WindTurbine(
            name=self.turbine_config.name,
            diameter=self.turbine_config.rotor_diameter,
            hub_height=self.turbine_config.hub_height,
            powerCtFunction=power_ct
        )

    def _build_site(self):

        # Turbine hub height [m]
        z_hub = 90.0

        # Wind-speed grid [m/s]
        ws = np.arange(3.0, 26.1, 0.1)

        # Ambient turbulence intensity
        ti = self.ambient_ti(ws, z=z_hub)

        # One wind-direction sector with probability = 1
        wd = np.array([0.0])
        p_wd = np.array([1.0])

        self.site = XRSite(
            xr.Dataset(
                data_vars={
                    'P': ('wd', p_wd),
                    'TI': ('ws', ti),
                },
                coords={
                    'wd': wd,
                    'ws': ws,
                }
            ),
            interp_method='linear'
        )

    def _build_model(self):
        kwargs = {}

        if self.superposition_model is not None:
            kwargs["superpositionModel"] = self.superposition_model

        if self.turbulence_model is not None:
            kwargs["turbulenceModel"] = self.turbulence_model

        self.wind_farm_model = PropagateDownwind(
            site=self.site,
            windTurbines=self.wind_turbines,
            wake_deficitModel=self.wake_deficit_model,
            **kwargs
        )

    def solve(
        self,
        ambient_u,
        ambient_wd,
    ):

        simulation = self.wind_farm_model(
            x=self.layout_x,
            y=self.layout_y,
            ws=np.atleast_1d(ambient_u),
            wd=np.atleast_1d(ambient_wd)
        )

        u_eff = (
            simulation.WS_eff
            .isel(
                wt=range(self.n_turbines),
                wd=0,
                ws=0
            )
            .values
            .astype("float32")
        )

        ti_eff = (
            simulation.TI_eff
            .isel(
                wt=range(self.n_turbines),
                wd=0,
                ws=0
            )
            .values
            .astype("float32")
        )

        power_kw = (
            simulation.Power
            .isel(
                wt=range(self.n_turbines),
                wd=0,
                ws=0
            )
            .values
            .astype("float32")
        )

        power = power_kw / 1e6  # MW

        return {
            "u_eff": u_eff,
            "ti_eff": ti_eff,
            "power": power,
        }

    def solve_complete(self, ambient_u, ambient_wd):
        simulation = self.wind_farm_model(
            x=self.layout_x,
            y=self.layout_y,
            ws=np.atleast_1d(ambient_u),
            wd=np.atleast_1d(ambient_wd)
        )

        u_eff = simulation.WS_eff.isel(wt=range(self.n_turbines), wd=0, ws=0).values.astype("float32")
        ti_eff = simulation.TI_eff.isel(wt=range(self.n_turbines), wd=0, ws=0).values.astype("float32")
        power_kw = simulation.Power.isel(wt=range(self.n_turbines), wd=0, ws=0).values.astype("float32")

        ambient_u = float(ambient_u)
        ambient_wd = float(ambient_wd)
        ambient_ti = float(self.ambient_ti(
            np.array([ambient_u]),
            z=self.turbine_config.hub_height
        )[0])

        velocity_ratio = (u_eff / ambient_u).astype("float32")
        velocity_deficit = (1.0 - velocity_ratio).astype("float32")
        ti_added = (ti_eff - ambient_ti).astype("float32")
        ti_ratio = (ti_eff / ambient_ti).astype("float32")

        return {
            "ambient_u": ambient_u,
            "ambient_ti": ambient_ti,
            "wind_direction": ambient_wd,
            "u_eff": u_eff,
            "ti_eff": ti_eff,
            "velocity_ratio": velocity_ratio,
            "velocity_deficit": velocity_deficit,
            "ti_added": ti_added,
            "ti_ratio": ti_ratio,
            "power_kw": power_kw,
            "power": power_kw / 1e6,
            "x": np.asarray(self.layout_x, dtype="float32"),
            "y": np.asarray(self.layout_y, dtype="float32"),
        }

    def flow_map(
        self,
        ambient_u,
        ambient_wd
    ):

        simulation = self.wind_farm_model(
            x=self.layout_x,
            y=self.layout_y,
            ws=ambient_u,
            wd=ambient_wd
        )

        return simulation.flow_map()

@dataclass
class DamageSolverConfig:
    csv_path: str
    u_column: str = "u"
    ti_column: str = "ti"
    del_flap_column: str = "del_flap"
    del_edge_column: str = "del_edge"
    m_coef: float = 10.0
    del_flap_ref: float = 2803.716141751299
    del_edge_ref: float = 5588.786717232858
    design_life_years: float = 20.0

class DamageSolver:
    def __init__(self, config):
        self.config = config
        self.response_surface_df = pd.read_csv(self.config.csv_path)
        self.u_grid = np.sort(self.response_surface_df["u"].unique())
        self.ti_grid = np.sort(self.response_surface_df["ti"].unique())
        self._build_interpolators()

    def _create_interpolator(self, value_column):
        surface = (
            self.response_surface_df
            .pivot(
                index=self.config.u_column,
                columns=self.config.ti_column,
                values=value_column
            )
            .values
        )
        return RegularGridInterpolator(
            (self.u_grid, self.ti_grid),
            surface,
            method="linear",
            bounds_error=False,
            fill_value=None
        )

    def _build_interpolators(self):
        self.interpolators = {
            "del_flap": self._create_interpolator(self.config.del_flap_column),
            "del_edge": self._create_interpolator(self.config.del_edge_column),
        }

    def predict_del(
        self,
        u_eff,
        ti_eff,
    ):
        points = np.column_stack((u_eff, ti_eff))

        del_flap = self.interpolators["del_flap"](points).astype(np.float32)
        del_edge = self.interpolators["del_edge"](points).astype(np.float32)

        return {
            "del_flap": del_flap,
            "del_edge": del_edge,
        }

    def solve(
        self,
        u_eff,
        ti_eff,
        duration_minutes=10.0
    ):
        n_ref = self.config.design_life_years*365*24*60/10 # number of cycles allowed per 10 minutes block
        del_pred = self.predict_del(u_eff, ti_eff)
        m = self.config.m_coef

        damage_flap_10min = ((del_pred["del_flap"]/self.config.del_flap_ref)**m)/n_ref
        damage_edge_10min = ((del_pred["del_edge"]/self.config.del_edge_ref)**m)/n_ref
        damage_10min = (damage_flap_10min + damage_edge_10min)/2

        scale = duration_minutes / 10.0
        damage = damage_10min * scale
        return damage


@dataclass
class MaintenanceType:
    name: str
    threshold: float

    # Maintenance consequences
    cost: float
    downtime_hours: float
    carbon_emission: float
    damage_multiplier: float

    # Duration of damage multiplier effectiveness.
    duration_months: int

    # Replacement resets degradation to zero.
    is_replacement: bool = False

@dataclass
class MaintenanceConfig:
    maintenance_types: tuple[MaintenanceType, ...]

    @property
    def replacement(self) -> MaintenanceType | None:
        return next(
            (
                maintenance
                for maintenance in self.maintenance_types
                if maintenance.is_replacement
            ),
            None,
        )

@dataclass
class MaintenanceResult:
    """
    Result of MaintenancePolicy.
    """
    maintenance_type: Optional[MaintenanceType]

    cost: float
    downtime_hours: float
    carbon_emission: float

    # Applied to future fatigue damage.
    damage_multiplier: float

    # Remaining duration of damage protection.
    protection_duration: int

class MaintenancePolicy:

    def __init__(self, config: MaintenanceConfig):
        self.config = config

    def solve(
        self,
        hi: float,
        action: int,
        protection_remaining: int,
    ) -> MaintenanceResult:

        if action not in (0, 1):
            raise ValueError(
                f"Unknown action: {action}. "
                "Expected 0 (No intervention) or 1 (Intervention)."
            )

        if not (0.0 <= hi <= 1.0):
            raise ValueError(
                f"Health Index must be within [0, 1]. "
                f"Received {hi}."
            )

        # ----------------------------------------------------------
        # No intervention requested
        # ----------------------------------------------------------
        if action == 0:
            return self._no_maintenance()

        # ----------------------------------------------------------
        # Select maintenance based on HI
        # ----------------------------------------------------------
        maintenance = self._select_type(hi)

        # ----------------------------------------------------------
        # No maintenance type is applicable
        # ----------------------------------------------------------
        if maintenance is None:
            return self._no_maintenance()

        # ----------------------------------------------------------
        # Protection constraint
        # ----------------------------------------------------------
        if protection_remaining > 0:
            return self._no_maintenance()

        # ----------------------------------------------------------
        # Maintenance approved
        # ----------------------------------------------------------
        return MaintenanceResult(
            maintenance_type=maintenance,

            cost=maintenance.cost,
            downtime_hours=maintenance.downtime_hours,
            carbon_emission=maintenance.carbon_emission,

            damage_multiplier=maintenance.damage_multiplier,
            protection_duration=maintenance.duration_months,
        )

    def _select_type(
        self,
        hi: float,
    ) -> MaintenanceType | None:

        applicable = [
            maintenance
            for maintenance in self.config.maintenance_types
            if hi <= maintenance.threshold
        ]

        if not applicable:
            return None

        return min(
            applicable,
            key=lambda maintenance: maintenance.threshold,
        )

    def _no_maintenance(self) -> MaintenanceResult:
        return MaintenanceResult(
            maintenance_type=None,

            cost=0.0,
            downtime_hours=0.0,
            carbon_emission=0.0,

            damage_multiplier=1.0,
            protection_duration=0,
        )

@dataclass
class TransitionResult:
    HI: float
    damage_multiplier: float
    protection_remaining: int

class TransitionModel:
    """
    Turbine health transition model.

    This model updates Health Index (HI) after:
    1. maintenance intervention
    2. monthly fatigue accumulation

    Notes
    -----
    - Stateless.
    - Does not modify Environment variables.
    - One TransitionModel handles one turbine.
    - Does not know which maintenance types exist.
    """

    def solve(
        self,
        hi: float,
        delta_damage: float,
        damage_multiplier: float,
        protection_remaining: int,
        maintenance: MaintenanceResult | None = None,
    ) -> TransitionResult:
        """
        Parameters
        ----------
        hi
            Current Health Index.

        delta_damage
            Monthly damage increment predicted by DamageSolver.

        damage_multiplier
            Current maintenance protection multiplier.

        protection_remaining
            Remaining protection duration in months.

        maintenance
            Maintenance decision generated by MaintenancePolicy.

        Returns
        -------
        TransitionResult
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

@dataclass
class ElectricityPriceModel:
    def __init__(
        self,
        monthly_mean,
        monthly_std,
    ):
        self.monthly_mean = np.asarray(monthly_mean, dtype=float)
        self.monthly_std = np.asarray(monthly_std, dtype=float)

    def sample(self, month, rng):
        """
        Parameters
        ----------
        month : int
            Month index (1-12)

        Returns
        -------
        float
            Electricity price (€/MWh)
        """
        mu = self.monthly_mean[month - 1]
        sigma = self.monthly_std[month - 1]

        price = rng.normal(mu, sigma)

        # Prevent negative prices (optional)
        return max(price, 0.0)

class SpatialGroupingObjective:
    """
    Spatial grouping objective based on the layout of selected
    maintenance turbines.

    Lower values indicate a more spatially compact maintenance group.
    """

    def __init__(self, layout_config):
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
        indices = np.asarray(
            maintenance_indices,
            dtype=int,
        )

        if indices.ndim != 1:
            raise ValueError(
                "maintenance_indices must be one-dimensional."
            )

        if len(indices) < 2:
            return 0.0

        if np.any(indices < 0) or np.any(indices >= self.n_turbines):
            raise IndexError(
                "maintenance_indices contains an invalid turbine index."
            )

        if len(np.unique(indices)) != len(indices):
            raise ValueError(
                "maintenance_indices must contain unique turbine indices."
            )

        selected_x = self.x[indices]
        selected_y = self.y[indices]

        dx = selected_x[:, None] - selected_x[None, :]
        dy = selected_y[:, None] - selected_y[None, :]

        distances = np.sqrt(dx**2 + dy**2)

        upper = np.triu_indices(
            len(indices),
            k=1,
        )

        return float(np.mean(distances[upper]))

@dataclass(frozen=True)
class RewardObjective:
    """
    Configuration of one optimization objective.
    """
    weight: float = 1.0
    direction: str = "min"     # "min" or "max"
    normalize: bool = False
    scale: float = 1.0

@dataclass(frozen=True)
class RewardConfig:
    """
    Configuration of reward scalarization.
    """
    objectives: dict[str, RewardObjective] = field(
        default_factory=dict
    )

@dataclass(frozen=True)
class RewardResult:
    reward: float
    objectives: dict[str, float]

class RewardModel:
    """
    Converts multi-objective maintenance metrics into
    an RL-compatible scalar reward.

    The Paper 3 formulation considers:
    1. fatigue-related degradation,
    2. economic maintenance consequences,
    3. environmental impact.

    All objectives are represented as minimization objectives.
    """

    def __init__(
        self,
        config: "RewardConfig",
    ):
        self.config = config

    def solve(
        self,
        metrics: dict[str, float],
    ) -> "RewardResult":

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
    Centralized simulation logger.

    Philosophy
    ----------
    One record = one simulated month.

    log_state()
        Creates a new monthly record.

    log_decision()
        Updates the most recent monthly record if that month
        corresponds to a maintenance decision.
    """

    history: List[Dict[str, Any]] = field(default_factory=list)

    def reset(self):
        self.history.clear()

    def log_state(
        self,
        *,
        year,
        month,
        elapsed_month,
        HI,
        power,
        electricity_price,
        ambient_u,
        ambient_wd,
        u_eff,
        ti_eff,
        repair_params: Dict[str, Any],
        repair_count,
        replacement_count
    ):

        self.history.append(
            {
                # --------------------------------------------------
                # Time
                # --------------------------------------------------
                "elapsed_month": int(elapsed_month),
                "year": int(year),
                "month": int(month),

                # --------------------------------------------------
                # State
                # --------------------------------------------------
                "HI_mean": float(np.mean(HI)),
                "HI_min": float(np.min(HI)),
                "HI_max": float(np.max(HI)),
                "HI_turbine": HI.copy().tolist(),

                "ambient_u": float(ambient_u),
                "ambient_wd": float(ambient_wd),

                "u_eff": u_eff.copy().tolist(),
                "ti_eff": ti_eff.copy().tolist(),

                "power": power.copy().tolist(),
                "mean_farm_power": float(np.mean(power)),
                "electricity_price": float(electricity_price),

                # --------------------------------------------------
                # State
                # --------------------------------------------------
                "repair_params": {
                    "threshold": float(repair_params["threshold"]),
                    "multiplier": float(repair_params["multiplier"]),
                    "protection_duration": int(
                        repair_params["duration"]
                    ),
                },

                # --------------------------------------------------
                # Cumulative maintenance counts
                # --------------------------------------------------
                "repair_count": repair_count.copy().tolist(),
                "replacement_count": replacement_count.copy().tolist(),

                # --------------------------------------------------
                # Decision information
                # Filled later by log_decision()
                # --------------------------------------------------
                "decision_event": False,
                "action": None,
                "maintenance_type": None,
                "reward": None,
                "energy": None,
                "maintenance_cost": None,
                "revenue": None,
                "interval_damage": None,
                "interval_damage_burden": None,
                "spatial_grouping": None,
                "interval_damage": None,
                "protection_remaining": None,
            }
        )

    def log_decision(
        self,
        *,
        reward,
        objectives,
        energy,
        revenue,
        maintenance_cost,
        interval_damage,
        interval_damage_burden,
        spatial_grouping,
        maintenance_results,
        action,
        protection_remaining,
    ):
        """
        Update the most recent mmonthly record with
        decision-related information.
        """

        if len(self.history) == 0:
            return

        row = self.history[-1]
        row["decision_event"] = True
        row["action"] = action.copy().tolist()
        row["energy"] = float(energy)
        row["revenue"] = float(revenue)
        row["maintenance_cost"] = float(maintenance_cost)
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
        return pd.DataFrame(self.history)

@dataclass
class EnergyResult:
    """
    Energy production result.
    """
    energy: float

class EnergySolver:
    """
    Convert turbine power into generated energy.
    """
    def solve(
        power: float,
        operating_hours: float,
    ) -> EnergyResult:
        energy = (
            power
            * operating_hours
        )

        return EnergyResult(
            energy=energy,
        )

import numpy as np
import matplotlib.pyplot as plt


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

        self.D = D
        self.n_turbines = n_turbines
        self.min_spacing_D = min_spacing_D
        self.min_spacing = min_spacing_D * D

        self.farm_scale = farm_scale
        self.seed = seed
        self.max_attempts = max_attempts

        # Farm size
        #
        # Original concept:
        # 1.5 × 7 × minimum_spacing
        #
        self.farm_size = (
            farm_scale
            * min_spacing_D
            * self.min_spacing
        )

        self.x_min = 0.0
        self.x_max = self.farm_size

        self.y_min = 0.0
        self.y_max = self.farm_size

        # Generate layout
        self.coordinates = self._generate()

        # Extract x and y
        self.x = self.coordinates[:, 0]
        self.y = self.coordinates[:, 1]

        # Calculate information
        self.info = self._calculate_info()


    # ========================================================
    # Generate random layout
    # ========================================================

    def _generate(self):

        rng = np.random.default_rng(self.seed)

        coordinates = []

        attempt = 0

        while (
            len(coordinates) < self.n_turbines
            and attempt < self.max_attempts
        ):

            attempt += 1

            # Random candidate
            candidate = np.array([
                rng.uniform(self.x_min, self.x_max),
                rng.uniform(self.y_min, self.y_max)
            ])

            # First turbine
            if len(coordinates) == 0:
                coordinates.append(candidate)
                continue

            # Distance to existing turbines
            distances = np.linalg.norm(
                np.asarray(coordinates) - candidate,
                axis=1
            )

            # Minimum-spacing constraint
            if np.all(distances >= self.min_spacing):
                coordinates.append(candidate)

        # Check whether generation succeeded
        if len(coordinates) < self.n_turbines:

            raise RuntimeError(
                f"Could only place {len(coordinates)} "
                f"out of {self.n_turbines} turbines "
                f"after {self.max_attempts:,} attempts.\n"
                f"Try increasing farm_scale or "
                f"reducing min_spacing_D."
            )

        coordinates = np.asarray(coordinates)

        # Center layout at (0, 0)
        coordinates -= coordinates.mean(axis=0)

        return coordinates


    # ========================================================
    # Calculate layout information
    # ========================================================

    def _calculate_info(self):

        # ----------------------------------------------------
        # Pairwise distances
        # ----------------------------------------------------

        pairwise_distances = []

        for i in range(self.n_turbines):

            for j in range(i + 1, self.n_turbines):

                distance = np.linalg.norm(
                    self.coordinates[i]
                    - self.coordinates[j]
                )

                pairwise_distances.append(distance)

        pairwise_distances = np.asarray(pairwise_distances)

        # ----------------------------------------------------
        # Bounding-box farm area
        # ----------------------------------------------------

        width = self.x.max() - self.x.min()
        height = self.y.max() - self.y.min()

        area = width * height

        area_per_turbine = area / self.n_turbines

        # ----------------------------------------------------
        # Information dictionary
        # ----------------------------------------------------

        return {
            "number_of_turbines": self.n_turbines,

            "rotor_diameter_m": self.D,

            "minimum_spacing_m":
                pairwise_distances.min(),

            "minimum_spacing_D":
                pairwise_distances.min() / self.D,

            "mean_pair_distance_m":
                pairwise_distances.mean(),

            "farm_width_m":
                width,

            "farm_height_m":
                height,

            "farm_area_km2":
                area / 1e6,

            "area_per_turbine_km2":
                area_per_turbine / 1e6,

            "seed":
                self.seed
        }


    # ========================================================
    # Print information
    # ========================================================

    def print_info(self):

        info = self.info

        print(
            f"Number of turbines : "
            f"{info['number_of_turbines']}"
        )

        print(
            f"Rotor diameter     : "
            f"{info['rotor_diameter_m']:.1f} m"
        )

        print(
            f"Minimum spacing    : "
            f"{info['minimum_spacing_m']:.2f} m"
        )

        print(
            f"Minimum spacing    : "
            f"{info['minimum_spacing_D']:.2f}D"
        )

        print(
            f"Mean pair distance : "
            f"{info['mean_pair_distance_m']:.2f} m"
        )

        print(
            f"Wind farm area     : "
            f"{info['farm_area_km2']:.2f} km²"
        )

        print(
            f"Area per turbine   : "
            f"{info['area_per_turbine_km2']:.2f} "
            f"km²/turbine"
        )

        print(
            f"Random seed        : "
            f"{info['seed']}"
        )


    # ========================================================
    # Plot layout
    # ========================================================

    def plot(self, turbine_labels=True):

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(
            self.x,
            self.y,
            s=100
        )

        if turbine_labels:

            for i, (x, y) in enumerate(
                zip(self.x, self.y),
                start=1
            ):

                ax.text(
                    x + 20,
                    y + 20,
                    f"T{i}",
                    fontsize=8
                )

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        ax.set_title(
            f"Random Scattered Layout "
            f"(minimum spacing = "
            f"{self.min_spacing_D:.0f}D)"
        )

        ax.set_aspect("equal")
        ax.grid(True)

        plt.show()

@dataclass(frozen=True)
class EnvironmentConfig:
    wind_climate: WindClimate
    wake_solver: WakeSolver
    damage_solver: DamageSolver
    maintenance_policy: MaintenancePolicy
    transition_model: TransitionModel
    electricity_price_model: ElectricityPriceModel
    spatial_grouping_objective: SpatialGroupingObjective
    reward_model: RewardModel
    energy_solver: EnergySolver
    logger: LoggingModel
    max_protection_duration: int
    initial_hi: float | np.ndarray
    wind_time_slices_per_month: int
    hours_per_month: float = 730.5
    decision_interval_months: int = 1
    max_simulation_years: int = 20
    u_min: float = 0.0
    u_max: float = 30.0
    ti_min: float = 0.03
    ti_max: float = 0.30
    seed: int = 42

    def get_initial_hi(self, n_turbines: int) -> np.ndarray:
        if np.isscalar(self.initial_hi):
            return np.full(
                n_turbines,
                self.initial_hi,
                dtype=np.float32,
            )
        hi = np.asarray(self.initial_hi, dtype=np.float32)
        if len(hi) != n_turbines:
            raise ValueError(
                "initial_hi length must equal number of turbines."
            )
        return hi.copy()

class OffshoreMaintenanceEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.wind_climate = config.wind_climate
        self.wake_solver = config.wake_solver
        self.damage_solver = config.damage_solver
        self.maintenance_policy = config.maintenance_policy
        self.transition_model = config.transition_model
        self.electricity_price_model = config.electricity_price_model
        self.spatial_grouping_objective = config.spatial_grouping_objective
        self.reward_model = config.reward_model
        self.logger = config.logger
        self.decision_interval = config.decision_interval_months
        self.n_turbines = self.wake_solver.n_turbines
        self.rng = np.random.default_rng(config.seed)

        # Weather
        self.ambient_u = None
        self.ambient_wd = None

        # Turbine
        self.u_eff = None
        self.ti_eff = None
        self.power = None

        # Damage/Maintenance
        self.HI = None
        self.delta_damage = None
        self.damage_multiplier = None
        self.protection_remaining = None
        self.repair_params = None

        # Intervention counters
        self.repair_count = None
        self.replacement_count = None

        # Reward
        self.electricity_price = None

        # Episode
        self.current_year = None
        self.current_month = None
        self.elapsed_month = None

        # Action
        self.last_action = None

        # History for Plotting
        self.history = None

        self.action_meaning = {
            0: "Continue",
            1: "Intervention"
        }

        self.observation_space = spaces.Dict({
            "farm": spaces.Dict({
                "month": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            }),

            "turbines": spaces.Dict({
                "HI": spaces.Box(low=0, high=1, shape=(self.n_turbines,), dtype=np.float32),
                "protection_remaining": spaces.Box(low=0, high=1, shape=(self.n_turbines,), dtype=np.float32)
            })
        })

        self.action_space = spaces.MultiDiscrete([2] * self.n_turbines)

    @property
    def max_steps(self):
        return int(
            np.ceil(
                self.config.max_simulation_years * 12
                / self.decision_interval
            )
        )

    def _get_obs(self):
        month = cyclic_encode(self.current_month - 1, period=12)
        protection_remaining_norm = (self.protection_remaining / self.config.max_protection_duration)
        return {
            "farm": {
                "month": month,
            },
            "turbines": {
                "HI": self.HI.astype(np.float32),
                "protection_remaining": protection_remaining_norm.astype(np.float32)
            }
        }

    def _get_info(self):
        return {
            "damage_multiplier": self.damage_multiplier.copy(),
            "last_action": self.last_action,
        }

    def reset(
        self,
        *,
        seed=None,
        options=None
    ):
        super().reset(seed=seed)

        # ==========================================================
        # Random number generator
        # ==========================================================
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # ----------------------------------------------------------
        # Maintenance parameter randomization
        # ----------------------------------------------------------
        randomize_maintenance_params = (
            options is not None
            and options.get("randomize_maintenance_params", False)
        )

        # ==========================================================
        # Episode-specific maintenance parameters
        # ==========================================================
        # Find the repair maintenance type from the existing config.
        # The sampled values will remain fixed throughout this episode
        # and will be resampled at the next reset().
        if randomize_maintenance_params:
            for maintenance in self.maintenance_policy.config.maintenance_types:

                if maintenance.name.lower() == "repair":

                    # Repair Threshold:
                    # 0.75 = late repair
                    # 0.90 = early repair
                    maintenance.threshold = self.rng.uniform(
                        0.75,
                        0.90
                    )

                    # Damage multiplier:
                    # 0.5 = stronger repair effect
                    # 0.9 = weaker repair effect
                    maintenance.damage_multiplier = self.rng.uniform(
                        0.5,
                        0.9
                    )

                    # Protection duration:
                    # Integer number of months.
                    # 3 to 12 months, inclusive.
                    maintenance.duration_months = int(
                        self.rng.integers(
                            3,
                            13
                        )
                    )

                    break

        self.repair_params = {
            "threshold": maintenance.threshold,
            "multiplier": maintenance.damage_multiplier,
            "duration": maintenance.duration_months,
        }

        # ==========================================================
        # Episode state
        # ==========================================================
        self.current_year = 1
        self.current_month = 1
        self.elapsed_month = 0

        # ==========================================================
        # Damage / Maintenance state
        # ==========================================================
        self.delta_damage = np.zeros(
            self.n_turbines,
            dtype=np.float32
        )

        self.damage_multiplier = np.ones(
            self.n_turbines,
            dtype=np.float32
        )

        self.protection_remaining = np.zeros(
            self.n_turbines,
            dtype=np.int32
        )

        self.repair_count = np.zeros(
            self.n_turbines,
            dtype=np.int32,
        )

        self.replacement_count = np.zeros(
            self.n_turbines,
            dtype=np.int32,
        )

        # ==========================================================
        # Action
        # ==========================================================
        self.last_action = np.zeros(
            self.n_turbines,
            dtype=np.int32
        )

        # ==========================================================
        # Turbine state
        # ==========================================================
        self.HI = self.config.get_initial_hi(
            self.n_turbines
        )

        # ==========================================================
        # Sample initial weather
        # ==========================================================
        (
            self.ambient_u,
            self.ambient_wd
        ) = self.wind_climate.sample(
            month=self.current_month,
            rng=self.rng
        )

        # ==========================================================
        # Sample initial electricity price
        # ==========================================================
        self.electricity_price = (
            self.electricity_price_model.sample(
                month=self.current_month,
                rng=self.rng
            )
        )

        # ==========================================================
        # Logger
        # ==========================================================
        self.logger.reset()

        # ==========================================================
        # Wake calculation
        # ==========================================================
        wake = self.wake_solver.solve(
            ambient_u=self.ambient_u,
            ambient_wd=self.ambient_wd,
        )

        self.u_eff = wake["u_eff"]
        self.ti_eff = wake["ti_eff"]
        self.power = wake["power"]

        # ==========================================================
        # Observation & info
        # ==========================================================
        observation = self._get_obs()
        info = self._get_info()

        # Useful for checking which maintenance realization
        # was sampled for this episode.
        for maintenance in self.maintenance_policy.config.maintenance_types:
            if maintenance.name.lower() == "repair":
                info["repair_damage_multiplier"] = (
                    maintenance.damage_multiplier
                )
                info["repair_duration_months"] = (
                    maintenance.duration_months
                )
                break

        return observation, info

    def step(self, action):

        # ----------------------------------------------------------
        # Validate action
        # ----------------------------------------------------------
        action = np.asarray(action, dtype=np.int32)

        if action.shape != (self.n_turbines,):
            raise ValueError(
                f"Expected action shape {(self.n_turbines,)}, got {action.shape}."
            )

        if np.any((action != 0) & (action != 1)):
            raise ValueError("Action values must be 0 or 1.")

        # ----------------------------------------------------------
        # Maintenance decisions (once per decision interval)
        # ----------------------------------------------------------
        maintenance_results = [
            self.maintenance_policy.solve(
                hi=self.HI[i],
                action=action[i],
                protection_remaining=self.protection_remaining[i],
            )
            for i in range(self.n_turbines)
        ]

        for i, maintenance in enumerate(maintenance_results):
            maintenance_type = maintenance.maintenance_type
            if maintenance_type is None:
                continue
            if maintenance_type.is_replacement:
                self.replacement_count[i] += 1
            else:
                self.repair_count[i] += 1

        mean_farm_power = 0.0
        interval_cost = 0.0
        interval_energy = 0.0
        interval_revenue = 0.0
        interval_damage = 0.0
        interval_damage_burden = 0.0

        terminated = False
        truncated = False

        hours_per_month = self.config.hours_per_month
        duration_minutes = hours_per_month * 60.0

        HI_before = self.HI.copy()

        # ----------------------------------------------------------
        # Simulate one decision interval
        # ----------------------------------------------------------
        months_simulated = 0

        for simulation_month in range(self.decision_interval):

            months_simulated += 1

            # ------------------------------------------------------
            # Monthly damage accumulator
            # ------------------------------------------------------
            monthly_damage = np.zeros(self.n_turbines)

            monthly_cost = 0.0
            monthly_energy = 0.0

            # ------------------------------------------------------
            # Sample multiple wind conditions within CURRENT month
            # ------------------------------------------------------
            n_slices = self.config.wind_time_slices_per_month

            slice_minutes = (
                duration_minutes / n_slices
            )

            for _ in range(n_slices):

                self.ambient_u, self.ambient_wd = (
                    self.wind_climate.sample(
                        month=self.current_month,
                        rng=self.rng,
                    )
                )

                # --------------------------------------------------
                # PyWake
                # --------------------------------------------------
                wake = self.wake_solver.solve(
                    ambient_u=self.ambient_u,
                    ambient_wd=self.ambient_wd,
                )

                self.u_eff = wake["u_eff"]
                self.ti_eff = wake["ti_eff"]
                self.power = wake["power"]

                mean_farm_power += np.sum(self.power)

                # --------------------------------------------------
                # Damage for this wind time slice
                # --------------------------------------------------
                damage_10min = self.damage_solver.solve(
                    u_eff=self.u_eff,
                    ti_eff=self.ti_eff,
                    duration_minutes=10.0,
                )

                # Damage accumulated over this time slice
                slice_damage = (
                    damage_10min
                    * slice_minutes
                    / 10.0
                )

                monthly_damage += slice_damage

            # ------------------------------------------------------
            # Monthly damage increment
            # ------------------------------------------------------
            self.delta_damage = monthly_damage

            # ------------------------------------------------------
            # Accumulate damage over decision interval
            # ------------------------------------------------------
            interval_damage += np.sum(self.delta_damage)

            # ------------------------------------------------------
            # Monthly state transition
            # ------------------------------------------------------
            for i in range(self.n_turbines):

                maintenance = None

                if simulation_month == 0:
                    maintenance = maintenance_results[i]

                transition = self.transition_model.solve(
                    hi=self.HI[i],
                    delta_damage=self.delta_damage[i],
                    damage_multiplier=self.damage_multiplier[i],
                    protection_remaining=self.protection_remaining[i],
                    maintenance=maintenance,
                )

                self.HI[i] = transition.HI
                self.damage_multiplier[i] = transition.damage_multiplier
                self.protection_remaining[i] = transition.protection_remaining

                availability = 1.0

                if simulation_month == 0 and maintenance is not None:
                    availability = max(
                        0.0,
                        (hours_per_month - maintenance.downtime_hours)
                        / hours_per_month,
                    )

                    monthly_cost += maintenance.cost

                monthly_energy += (
                    self.power[i]
                    * availability
                )

            interval_cost += monthly_cost
            interval_energy += monthly_energy
            interval_revenue += monthly_energy * self.electricity_price
            interval_damage_burden += np.sum(1.0 - self.HI)

            # ----------------------------------------------------------
            # Log CURRENT month state
            # (before advancing to next month)
            # ----------------------------------------------------------
            self.logger.log_state(
                year=self.current_year,
                month=self.current_month,
                elapsed_month=self.elapsed_month + 1,
                HI=self.HI.copy(),
                power=self.power,
                electricity_price=self.electricity_price,
                ambient_u=self.ambient_u,
                ambient_wd=self.ambient_wd,
                u_eff=self.u_eff,
                ti_eff=self.ti_eff,
                repair_params=self.repair_params,
                repair_count=self.repair_count,
                replacement_count=self.replacement_count,
            )

            # ----------------------------------------------------------
            # Advance calendar
            # ----------------------------------------------------------
            self.elapsed_month += 1

            self.current_month += 1

            if self.current_month > 12:
                self.current_month = 1
                self.current_year += 1

            if self.elapsed_month >= self.config.max_simulation_years * 12:
                truncated = True
                break

            # ----------------------------------------------------------
            # Prepare next month's operating condition
            # ----------------------------------------------------------
            self.electricity_price = self.electricity_price_model.sample(
                month=self.current_month,
                rng=self.rng
            )

        mean_farm_power /= max(months_simulated, 1)

        maintenance_indices = [i for i, result in enumerate(maintenance_results) if result.maintenance_type is not None]
        spatial_grouping = self.spatial_grouping_objective.solve(maintenance_indices)

        metrics = {
            "damage_burden": interval_damage_burden,
            "spatial_grouping": spatial_grouping,
        }

        reward_result = self.reward_model.solve(metrics)

        self.logger.log_decision(
            reward=reward_result.reward,
            objectives=reward_result.objectives,
            energy=interval_energy,
            revenue=interval_revenue,
            interval_damage_burden=interval_damage_burden,
            spatial_grouping=spatial_grouping,
            maintenance_cost=interval_cost,
            interval_damage=interval_damage,
            maintenance_results=maintenance_results,
            action=action,
            protection_remaining=self.protection_remaining,
        )

        observation = self._get_obs()
        info = self._get_info()
        info["maintenance_results"] = maintenance_results

        return (
            observation,
            float(reward_result.reward),
            terminated,
            truncated,
            info,
        )

'''
Configuration
'''
from py_wake.deficit_models import BastankhahGaussianDeficit, NoWakeDeficit
from py_wake.superposition_models import LinearSum, SqrMaxSum
from py_wake.turbulence_models import STF2017TurbulenceModel

# Config
## Climate Config
wind_monthly_weibull = {
        1:  {"k": 2.2, "c": 13.0},
        2:  {"k": 2.2, "c": 12.5},
        3:  {"k": 2.2, "c": 11.5},
        4:  {"k": 2.2, "c": 10.5},
        5:  {"k": 2.2, "c": 9.5},
        6:  {"k": 2.2, "c": 8.5},
        7:  {"k": 2.2, "c": 8.0},
        8:  {"k": 2.2, "c": 8.5},
        9:  {"k": 2.2, "c": 9.5},
        10: {"k": 2.2, "c": 11.0},
        11: {"k": 2.2, "c": 12.0},
        12: {"k": 2.2, "c": 13.0},
    }

wind_direction = np.arange(0.0, 360.0, 10.0)

# von Misses distribution
dominant_direction = 270.0
kappa = 2.0
theta = np.deg2rad(wind_direction)
mu = np.deg2rad(dominant_direction)
wind_direction_probability = np.exp(kappa * np.cos(theta - mu))
wind_direction_probability /= wind_direction_probability.sum()

wind_climate_config = WindClimateConfig(
    monthly_weibull=wind_monthly_weibull,
    wind_direction=wind_direction,
    wind_direction_probability=None,
)

#Site Config
n_turbines=25
layout = RandomScatteredLayout(
    D=126.0,
    n_turbines=n_turbines,
    min_spacing_D=7.0,
    farm_scale=1.0,
    seed=42
)

x_layout = layout.x
y_layout = layout.y

layout_config = LayoutConfig(
    x=np.array(x_layout),
    y=np.array(y_layout)
)
turbine_config = TurbineConfig(
    name="NREL 5-MW",
    rotor_diameter=126.0,
    hub_height=90.0,
    csv_path="/content/drive/MyDrive/1openfast_data/NREL_Reference_5MW_126.csv",
    power_unit="kW"
)
wake_solver_config = WakeSolverConfig(
    layout=layout_config,
    turbine=turbine_config,
    wake_deficit_model=NoWakeDeficit(),
    superposition_model=LinearSum(),
    turbulence_model=STF2017TurbulenceModel(
            addedTurbulenceSuperpositionModel=SqrMaxSum()
        ),
)
spatial_grouping_objective = SpatialGroupingObjective(
    layout_config=layout_config
)

## Electricity Price Config
monthly_mean = [
    90, 85, 78, 70, 62, 55, 50, 52, 60, 72, 82, 92
]
monthly_std = [
    10, 10, 9, 8, 8, 7, 7, 7, 8, 9, 10, 10
]

price_model = ElectricityPriceModel(
    monthly_mean,
    monthly_std,
)

## Damage Config
damage_solver_config = DamageSolverConfig(
    csv_path="/content/drive/MyDrive/1openfast_data/response_surface_df_200.csv",
    u_column="u",
    ti_column="ti",
    del_flap_column="del_flap",
    del_edge_column="del_edge",
    m_coef=10.0,
    del_flap_ref=2771.0, # Calibrated for n=30 sampling
    del_edge_ref=5523.7, # Calibrated for n=30 sampling
    design_life_years=20
)

repair = MaintenanceType(
    name="repair",
    threshold=0.85,
    cost=2_000,
    downtime_hours=8,
    carbon_emission=200,
    damage_multiplier=0.45,
    duration_months=15,
    is_replacement=False,
)

replacement = MaintenanceType(
    name="replacement",
    threshold=0.10,
    cost=350_000,
    downtime_hours=168,
    carbon_emission=5_000,
    damage_multiplier=1.0,
    duration_months=1,
    is_replacement=True,
)

maintenance_config = MaintenanceConfig(
    maintenance_types=(
        repair,
        replacement,
    ),
)

## Protection Duration
max_protection_duration = max(
    maintenance.duration_months
    for maintenance in maintenance_config.maintenance_types
)

decision_interval_months=6
spatial_reference_scale = np.max(np.sqrt(
    (layout_config.x[:, None] - layout_config.x[None, :])**2 + (layout_config.y[:, None] - layout_config.y[None, :])**2))
reward_config = RewardConfig(
    objectives={
        "damage_burden": RewardObjective(
            weight=0.85,
            direction="min",
            normalize=True,
            scale=n_turbines*decision_interval_months,
        ),
        "spatial_grouping": RewardObjective(
            weight=0.15,
            direction="min",
            normalize=True,
            scale=spatial_reference_scale
        ),
    }
)

# Solver & Model
wind_climate = WindClimate(wind_climate_config)
wake_solver = WakeSolver(wake_solver_config)
damage_solver = DamageSolver(damage_solver_config)
maintenance_policy = MaintenancePolicy(maintenance_config)
transition_model = TransitionModel()
reward_model = RewardModel(reward_config)
energy_solver = EnergySolver()
logger = LoggingModel()

env_config = EnvironmentConfig(
    wind_climate=wind_climate,
    wake_solver=wake_solver,
    damage_solver=damage_solver,
    maintenance_policy=maintenance_policy,
    transition_model=transition_model,
    electricity_price_model=price_model,
    spatial_grouping_objective=spatial_grouping_objective,
    reward_model=reward_model,
    energy_solver=energy_solver,
    logger=logger,
    max_protection_duration=max_protection_duration,
    wind_time_slices_per_month=30,
    max_simulation_years=30,
    decision_interval_months=6,
    initial_hi=1.0,
    seed=42
)

layout.print_info()
layout.plot()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. GENERATE MONTHLY WIND-SPEED SAMPLES
# ============================================================

def generate_monthly_wind_samples(
    monthly_weibull,
    n_samples=10_000,
    wind_min=3.0,
    wind_max=25.0,
    seed=42,
):
    """
    Generate monthly freestream wind-speed samples using
    month-dependent Weibull distributions with rejection
    sampling over the prescribed wind-speed range.
    """

    rng = np.random.default_rng(seed)

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    records = []

    for month in range(1, 13):

        k = monthly_weibull[month]["k"]
        c = monthly_weibull[month]["c"]

        n_accepted = 0

        while n_accepted < n_samples:

            n_remaining = n_samples - n_accepted

            # Generate Weibull samples
            u = c * rng.weibull(
                k,
                size=n_remaining
            )

            # Apply prescribed wind-speed bounds
            u = u[
                (u >= wind_min) &
                (u <= wind_max)
            ]

            records.extend(
                {
                    "month": month,
                    "month_name": month_names[month - 1],
                    "wind_speed": value,
                }
                for value in u
            )

            n_accepted += len(u)

    return pd.DataFrame(records)


# ============================================================
# 2. CALCULATE MONTHLY WIND-SPEED STATISTICS
# ============================================================

def calculate_monthly_wind_statistics(
    samples_df
):
    """
    Calculate descriptive statistics of the generated
    monthly freestream wind-speed samples.
    """

    statistics_df = (
        samples_df
        .groupby(
            ["month", "month_name"]
        )["wind_speed"]
        .agg(
            mean="mean",
            median="median",
            p05=lambda x: np.percentile(x, 5),
            p95=lambda x: np.percentile(x, 95),
        )
        .reset_index()
        .sort_values("month")
    )

    return statistics_df


# ============================================================
# 3. PLOT MONTHLY WIND-SPEED STATISTICS
# ============================================================

def plot_monthly_wind_statistics(
    statistics_df,
    figsize=(10, 5),
):
    """
    Plot monthly freestream wind-speed statistics,
    including the 5th–95th percentile range,
    mean, and median.
    """

    fig, ax = plt.subplots(
        figsize=figsize
    )

    x = np.arange(
        len(statistics_df)
    )

    # 5th–95th percentile range
    ax.fill_between(
        x,
        statistics_df["p05"],
        statistics_df["p95"],
        alpha=0.2,
        label="5th–95th percentile",
    )

    # Mean
    ax.plot(
        x,
        statistics_df["mean"],
        marker="o",
        linewidth=2,
        label="Mean",
    )

    # Median
    ax.plot(
        x,
        statistics_df["median"],
        marker="s",
        linewidth=1.5,
        label="Median",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        statistics_df["month_name"]
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "Freestream wind speed (m/s)"
    )

    ax.set_title(
        "Monthly Wind-Speed Statistics"
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    return fig


# ============================================================
# 4. PLOT FREESTREAM WIND SPEED–TI RELATIONSHIP
# ============================================================

def plot_wind_speed_ti_relationship(
    wind_min=3.0,
    wind_max=25.0,
    hub_height=90.0,
    a1=0.025,
    a2=0.0450,
    a3=0.030,
    figsize=(7, 5),
):
    """
    Plot the deterministic freestream wind speed–TI relationship
    using the extended ISO relationship reported by Jeans (2024)
    with the neutral offshore ENOW coefficient set.

    TI(z) =
        [a1(U/10) + a2 + a3(U/10)^(-1)]
        (z/10)^(-0.22)

    TI is dimensionless internally and is converted to percent
    only for visualization.
    """

    # Wind-speed range
    wind_speed = np.linspace(
        wind_min,
        wind_max,
        500,
    )

    # Normalized wind speed
    wind_ratio = wind_speed / 10.0

    # Jeans relationship with ENOW coefficients
    ti = (
        a1 * wind_ratio
        + a2
        + a3 * wind_ratio**(-1)
    ) * (
        hub_height / 10.0
    )**(-0.22)

    # Convert dimensionless TI to percentage for plotting
    ti_percent = ti * 100.0

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=figsize
    )

    ax.plot(
        wind_speed,
        ti_percent,
        linewidth=2,
    )

    ax.set_xlim(
        wind_min,
        wind_max,
    )

    ax.set_xlabel(
        "Freestream wind speed (m/s)"
    )

    ax.set_ylabel(
        "Freestream turbulence intensity (%)"
    )

    ax.set_title(
        "Freestream Wind Speed–TI Relationship"
    )

    ax.grid(
        axis="both",
        linestyle="--",
        alpha=0.3,
    )

    # Model information
    ax.text(
        0.98,
        0.98,
        (
            f"$z={hub_height:.0f}$ m\n"
            f"$a_1={a1:.3f}$, "
            f"$a_2={a2:.4f}$, "
            f"$a_3={a3:.3f}$"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    fig.tight_layout()

    return fig

wind_monthly_weibull = {
    1:  {"k": 2.2, "c": 13.0},
    2:  {"k": 2.2, "c": 12.5},
    3:  {"k": 2.2, "c": 11.5},
    4:  {"k": 2.2, "c": 10.5},
    5:  {"k": 2.2, "c": 9.5},
    6:  {"k": 2.2, "c": 8.5},
    7:  {"k": 2.2, "c": 8.0},
    8:  {"k": 2.2, "c": 8.5},
    9:  {"k": 2.2, "c": 9.5},
    10: {"k": 2.2, "c": 11.0},
    11: {"k": 2.2, "c": 12.0},
    12: {"k": 2.2, "c": 13.0},
}

samples_df = generate_monthly_wind_samples(
    monthly_weibull=wind_monthly_weibull,
    n_samples=10_000,
    wind_min=3.0,
    wind_max=25.0,
    seed=42,
)

# Calculate monthly statistics
statistics_df = calculate_monthly_wind_statistics(
    samples_df
)

# Plot 1: seasonal wind-speed statistics
fig1 = plot_monthly_wind_statistics(
    statistics_df,
    figsize=(5, 4)
)

plt.show()

# Plot 2: deterministic wind speed–TI relationship
fig2 = plot_wind_speed_ti_relationship(
    wind_min=3.0,
    wind_max=25.0,
    hub_height=90.0,
    a1=0.025,
    a2=0.0450,
    a3=0.030,
    figsize=(5, 4)
)

plt.show()

layout_config

# Solver asli
wake_solver_linear = WakeSolver(
    WakeSolverConfig(
        layout=layout_config,
        turbine=turbine_config,
        wake_deficit_model=BastankhahGaussianDeficit(),
        superposition_model=LinearSum(),
        turbulence_model=STF2017TurbulenceModel(
            addedTurbulenceSuperpositionModel=LinearSum()
        ),
    )
)

# Hanya untuk eksperimen
wake_solver_sqrmax = WakeSolver(
    WakeSolverConfig(
        layout=layout_config,
        turbine=turbine_config,
        wake_deficit_model=BastankhahGaussianDeficit(),
        superposition_model=LinearSum(),
        turbulence_model=STF2017TurbulenceModel(
            addedTurbulenceSuperpositionModel=SqrMaxSum()
        ),
    )
)

# ============================================================
# EXPERIMENT: CHECK TI_eff FROM CURRENT WakeSolver
# 25 turbines, 5x5 square layout, spacing 7D
# No RL environment / no damage / no HI
# ============================================================

import numpy as np

# ------------------------------------------------------------
# 1. Select one representative wind condition
# ------------------------------------------------------------
U_test = 10.0       # wind speed [m/s]

# IMPORTANT:
# Set this according to the orientation of your layout.
# For a first aligned-wake test, choose the WD that makes
# wind direction parallel to one axis of the 5x5 layout.
WD_test = 0.0       # wind direction [deg]


# ------------------------------------------------------------
# 2. Use the EXISTING wake_solver
# ------------------------------------------------------------
# This is your current configuration:
#
# BastankhahGaussianDeficit()
# + LinearSum() wake deficit superposition
# + STF2017TurbulenceModel()
#
# Nothing is changed here.

result = wake_solver.solve(
    ambient_u=U_test,
    ambient_wd=WD_test,
)


# ------------------------------------------------------------
# 3. Extract results
# ------------------------------------------------------------
u_eff = np.asarray(result["u_eff"], dtype=float)
ti_eff = np.asarray(result["ti_eff"], dtype=float)
power = np.asarray(result["power"], dtype=float)


# ------------------------------------------------------------
# 4. Check number of turbines
# ------------------------------------------------------------
print("=" * 75)
print("WAKE SOLVER TI EXPERIMENT")
print("=" * 75)

print(f"Number of turbines : {len(ti_eff)}")
print(f"Ambient wind speed : {U_test:.2f} m/s")
print(f"Wind direction     : {WD_test:.2f} deg")

print()


# ------------------------------------------------------------
# 5. Print turbine-by-turbine results
# ------------------------------------------------------------
print(
    f"{'Turbine':<10}"
    f"{'U_eff [m/s]':>15}"
    f"{'TI_eff':>15}"
    f"{'Power [MW]':>15}"
)

print("-" * 75)

for i in range(len(ti_eff)):
    print(
        f"T{i+1:02d}"
        f"{u_eff[i]:>15.4f}"
        f"{ti_eff[i]:>15.5f}"
        f"{power[i]:>15.4f}"
    )


# ------------------------------------------------------------
# 6. Summary statistics
# ------------------------------------------------------------
print()
print("=" * 75)
print("TI_eff SUMMARY")
print("=" * 75)

print(f"Minimum TI_eff : {np.min(ti_eff):.5f}")
print(f"Mean TI_eff    : {np.mean(ti_eff):.5f}")
print(f"Maximum TI_eff : {np.max(ti_eff):.5f}")


# ------------------------------------------------------------
# 7. Identify turbines with highest TI
# ------------------------------------------------------------
print()
print("=" * 75)
print("HIGHEST TI_eff")
print("=" * 75)

idx_sorted = np.argsort(ti_eff)[::-1]

for rank, i in enumerate(idx_sorted[:10], start=1):
    print(
        f"{rank:02d}. "
        f"T{i+1:02d}  "
        f"TI_eff={ti_eff[i]:.5f}  "
        f"U_eff={u_eff[i]:.4f} m/s"
    )


# ------------------------------------------------------------
# 8. Show layout coordinates together with TI_eff
# ------------------------------------------------------------
print()
print("=" * 75)
print("TI_eff BY LAYOUT POSITION")
print("=" * 75)

for i in range(len(ti_eff)):
    x = wake_solver.layout_x[i]
    y = wake_solver.layout_y[i]

    print(
        f"T{i+1:02d}: "
        f"x={x:8.2f}, "
        f"y={y:8.2f}, "
        f"U_eff={u_eff[i]:7.3f}, "
        f"TI_eff={ti_eff[i]:.5f}"
    )


# ------------------------------------------------------------
# 9. Optional: reshape only if your turbine ordering is
#    confirmed to be 5x5 row-major.
# ------------------------------------------------------------
if len(ti_eff) == 25:

    print()
    print("=" * 75)
    print("TI_eff AS 5x5 GRID")
    print("=" * 75)

    print(
        np.array2string(
            ti_eff.reshape(5, 5),
            formatter={
                "float_kind": lambda x: f"{x:7.4f}"
            }
        )
    )

# ============================================================
# EXPERIMENT: STF TURBULENCE SUPERPOSITION
# LinearSum vs SqrMaxSum
#
# Tujuan:
# Mengecek apakah akumulasi added turbulence STF dengan
# LinearSum menyebabkan TI_eff downstream terlalu tinggi.
#
# Tidak menggunakan RL environment.
# Tidak mengubah env_config.
# Tidak mengubah step().
# ============================================================

import numpy as np

from py_wake.superposition_models import LinearSum, SqrMaxSum
from py_wake.turbulence_models import STF2017TurbulenceModel


# ------------------------------------------------------------
# 1. Test condition
# ------------------------------------------------------------
U_test = 10.0
WD_test = 0.0


# ------------------------------------------------------------
# 2. CASE A — konfigurasi asli
# ------------------------------------------------------------
wake_solver_linear = WakeSolver(
    WakeSolverConfig(
        layout=layout_config,
        turbine=turbine_config,
        wake_deficit_model=BastankhahGaussianDeficit(),
        superposition_model=LinearSum(),
        turbulence_model=STF2017TurbulenceModel(
            addedTurbulenceSuperpositionModel=LinearSum()
        ),
    )
)


# ------------------------------------------------------------
# 3. CASE B — hanya added turbulence STF yang diubah
# ------------------------------------------------------------
wake_solver_sqrmax = WakeSolver(
    WakeSolverConfig(
        layout=layout_config,
        turbine=turbine_config,
        wake_deficit_model=BastankhahGaussianDeficit(),

        # Tetap sama dengan konfigurasi asli
        superposition_model=LinearSum(),

        # HANYA ini yang berubah
        turbulence_model=STF2017TurbulenceModel(
            addedTurbulenceSuperpositionModel=SqrMaxSum()
        ),
    )
)


# ------------------------------------------------------------
# 4. Run CASE A — LinearSum
# ------------------------------------------------------------
result_linear = wake_solver_linear.solve(
    ambient_u=U_test,
    ambient_wd=WD_test,
)

u_linear = np.asarray(result_linear["u_eff"], dtype=float)
ti_linear = np.asarray(result_linear["ti_eff"], dtype=float)


# ------------------------------------------------------------
# 5. Run CASE B — SqrMaxSum
# ------------------------------------------------------------
result_sqrmax = wake_solver_sqrmax.solve(
    ambient_u=U_test,
    ambient_wd=WD_test,
)

u_sqrmax = np.asarray(result_sqrmax["u_eff"], dtype=float)
ti_sqrmax = np.asarray(result_sqrmax["ti_eff"], dtype=float)


# ------------------------------------------------------------
# 6. Basic validation
# ------------------------------------------------------------
assert len(ti_linear) == len(ti_sqrmax), (
    "Number of turbines berbeda antara kedua solver."
)

assert len(ti_linear) == 25, (
    f"Diharapkan 25 turbine, tetapi ditemukan {len(ti_linear)}."
)


# ------------------------------------------------------------
# 7. Print comparison
# ------------------------------------------------------------
print("=" * 85)
print("STF ADDED TURBULENCE EXPERIMENT")
print("=" * 85)

print(f"Ambient wind speed : {U_test:.2f} m/s")
print(f"Wind direction     : {WD_test:.2f} deg")
print(f"Number of turbines : {len(ti_linear)}")

print()
print(
    f"{'Turbine':<10}"
    f"{'U Linear':>12}"
    f"{'TI Linear':>14}"
    f"{'U SqrMax':>12}"
    f"{'TI SqrMax':>14}"
    f"{'Δ TI':>12}"
)

print("-" * 85)

for i in range(len(ti_linear)):

    delta_ti = ti_linear[i] - ti_sqrmax[i]

    print(
        f"T{i+1:02d}"
        f"{u_linear[i]:>12.4f}"
        f"{ti_linear[i]:>14.5f}"
        f"{u_sqrmax[i]:>12.4f}"
        f"{ti_sqrmax[i]:>14.5f}"
        f"{delta_ti:>12.5f}"
    )


# ------------------------------------------------------------
# 8. Summary
# ------------------------------------------------------------
print()
print("=" * 85)
print("SUMMARY")
print("=" * 85)

print("\nLinearSum — added turbulence:")
print(f"  TI min  = {ti_linear.min():.5f}")
print(f"  TI mean = {ti_linear.mean():.5f}")
print(f"  TI max  = {ti_linear.max():.5f}")

print("\nSqrMaxSum — added turbulence:")
print(f"  TI min  = {ti_sqrmax.min():.5f}")
print(f"  TI mean = {ti_sqrmax.mean():.5f}")
print(f"  TI max  = {ti_sqrmax.max():.5f}")


# ------------------------------------------------------------
# 9. Difference
# ------------------------------------------------------------
delta = ti_linear - ti_sqrmax

print()
print("Difference (LinearSum - SqrMaxSum):")
print(f"  ΔTI min  = {delta.min():.5f}")
print(f"  ΔTI mean = {delta.mean():.5f}")
print(f"  ΔTI max  = {delta.max():.5f}")


# ------------------------------------------------------------
# 10. 5x5 layout comparison
# ------------------------------------------------------------
print()
print("=" * 85)
print("TI_eff — LinearSum")
print("=" * 85)

print(
    np.array2string(
        ti_linear.reshape(5, 5),
        formatter={"float_kind": lambda x: f"{x:7.4f}"}
    )
)

print()
print("=" * 85)
print("TI_eff — SqrMaxSum")
print("=" * 85)

print(
    np.array2string(
        ti_sqrmax.reshape(5, 5),
        formatter={"float_kind": lambda x: f"{x:7.4f}"}
    )
)

print()
print("=" * 85)
print("ΔTI — LinearSum minus SqrMaxSum")
print("=" * 85)

print(
    np.array2string(
        delta.reshape(5, 5),
        formatter={"float_kind": lambda x: f"{x:7.4f}"}
    )
)

# ============================================================
# MULTI-CONDITION TEST
# STF Added Turbulence:
# LinearSum vs SqrMaxSum
#
# Tujuan:
# Memastikan apakah perbedaan TI yang terlihat pada
# U=10 m/s, WD=0 deg tetap muncul pada berbagai
# wind speed dan wind direction.
#
# Tidak menggunakan RL environment.
# Tidak mengubah env_config.
# ============================================================

import numpy as np

from py_wake.superposition_models import LinearSum, SqrMaxSum
from py_wake.turbulence_models import STF2017TurbulenceModel


# ============================================================
# 1. TEST CONDITIONS
# ============================================================

wind_speeds = [6.0, 8.0, 10.0, 12.0, 15.0]
wind_directions = [0.0, 15.0, 30.0, 45.0]


# ============================================================
# 2. BUILD CASE A
#    Original:
#    wake deficit superposition = LinearSum
#    STF added turbulence       = LinearSum
# ============================================================

wake_solver_linear = WakeSolver(
    WakeSolverConfig(
        layout=layout_config,
        turbine=turbine_config,

        wake_deficit_model=BastankhahGaussianDeficit(),

        # Keep wake-deficit superposition unchanged
        superposition_model=LinearSum(),

        turbulence_model=STF2017TurbulenceModel(
            addedTurbulenceSuperpositionModel=LinearSum()
        ),
    )
)


# ============================================================
# 3. BUILD CASE B
#    Only STF added turbulence changes:
#    LinearSum -> SqrMaxSum
# ============================================================

wake_solver_sqrmax = WakeSolver(
    WakeSolverConfig(
        layout=layout_config,
        turbine=turbine_config,

        wake_deficit_model=BastankhahGaussianDeficit(),

        # Keep wake-deficit superposition unchanged
        superposition_model=LinearSum(),

        turbulence_model=STF2017TurbulenceModel(
            addedTurbulenceSuperpositionModel=SqrMaxSum()
        ),
    )
)


# ============================================================
# 4. RUN EXPERIMENT
# ============================================================

results = []


for U in wind_speeds:

    for WD in wind_directions:

        # ----------------------------------------------------
        # LinearSum
        # ----------------------------------------------------
        result_linear = wake_solver_linear.solve(
            ambient_u=U,
            ambient_wd=WD,
        )

        ti_linear = np.asarray(
            result_linear["ti_eff"],
            dtype=float
        )

        u_linear = np.asarray(
            result_linear["u_eff"],
            dtype=float
        )


        # ----------------------------------------------------
        # SqrMaxSum
        # ----------------------------------------------------
        result_sqrmax = wake_solver_sqrmax.solve(
            ambient_u=U,
            ambient_wd=WD,
        )

        ti_sqrmax = np.asarray(
            result_sqrmax["ti_eff"],
            dtype=float
        )

        u_sqrmax = np.asarray(
            result_sqrmax["u_eff"],
            dtype=float
        )


        # ----------------------------------------------------
        # Check that wake deficit is unchanged
        # ----------------------------------------------------
        max_u_difference = np.max(
            np.abs(u_linear - u_sqrmax)
        )


        # ----------------------------------------------------
        # Store summary
        # ----------------------------------------------------
        results.append({
            "U": U,
            "WD": WD,

            "TI_linear_min": np.min(ti_linear),
            "TI_linear_mean": np.mean(ti_linear),
            "TI_linear_max": np.max(ti_linear),

            "TI_sqrmax_min": np.min(ti_sqrmax),
            "TI_sqrmax_mean": np.mean(ti_sqrmax),
            "TI_sqrmax_max": np.max(ti_sqrmax),

            "delta_TI_mean": (
                np.mean(ti_linear)
                - np.mean(ti_sqrmax)
            ),

            "delta_TI_max": (
                np.max(ti_linear)
                - np.max(ti_sqrmax)
            ),

            "ratio_TI_max": (
                np.max(ti_linear)
                / np.max(ti_sqrmax)
                if np.max(ti_sqrmax) > 0
                else np.nan
            ),

            "max_U_difference": max_u_difference,
        })


# ============================================================
# 5. PRINT SUMMARY TABLE
# ============================================================

print("=" * 120)
print("MULTI-CONDITION STF TURBULENCE TEST")
print("=" * 120)

print(
    f"{'U':>5}"
    f"{'WD':>6}"
    f"{'Lin Mean':>12}"
    f"{'Lin Max':>12}"
    f"{'Sqr Mean':>12}"
    f"{'Sqr Max':>12}"
    f"{'Δ Mean':>12}"
    f"{'Δ Max':>12}"
    f"{'Max Ratio':>12}"
    f"{'ΔU':>10}"
)

print("-" * 120)


for r in results:

    print(
        f"{r['U']:>5.1f}"
        f"{r['WD']:>6.1f}"
        f"{r['TI_linear_mean']:>12.5f}"
        f"{r['TI_linear_max']:>12.5f}"
        f"{r['TI_sqrmax_mean']:>12.5f}"
        f"{r['TI_sqrmax_max']:>12.5f}"
        f"{r['delta_TI_mean']:>12.5f}"
        f"{r['delta_TI_max']:>12.5f}"
        f"{r['ratio_TI_max']:>12.3f}"
        f"{r['max_U_difference']:>10.6f}"
    )


# ============================================================
# 6. FIND THE LARGEST DIFFERENCE
# ============================================================

largest_difference = max(
    results,
    key=lambda x: x["delta_TI_max"]
)


print()
print("=" * 80)
print("LARGEST TI DIFFERENCE")
print("=" * 80)

print(
    f"U                  : "
    f"{largest_difference['U']:.1f} m/s"
)

print(
    f"WD                 : "
    f"{largest_difference['WD']:.1f} deg"
)

print(
    f"LinearSum TI max   : "
    f"{largest_difference['TI_linear_max']:.5f}"
)

print(
    f"SqrMaxSum TI max   : "
    f"{largest_difference['TI_sqrmax_max']:.5f}"
)

print(
    f"Difference         : "
    f"{largest_difference['delta_TI_max']:.5f}"
)

print(
    f"Linear/SqrMax ratio: "
    f"{largest_difference['ratio_TI_max']:.3f}"
)


# ============================================================
# 7. CHECK WHETHER U_eff IS IDENTICAL
# ============================================================

max_u_difference_overall = max(
    r["max_U_difference"]
    for r in results
)

print()
print("=" * 80)
print("WAKE DEFICIT CONSISTENCY CHECK")
print("=" * 80)

print(
    "Maximum |U_eff LinearSum - U_eff SqrMaxSum| "
    f"across all cases = {max_u_difference_overall:.10f} m/s"
)

if max_u_difference_overall < 1e-6:

    print(
        "RESULT: U_eff is effectively identical. "
        "The TI difference is isolated to the turbulence "
        "superposition."
    )

else:

    print(
        "WARNING: U_eff differs between the two cases. "
        "Check the solver configuration."
    )


# ============================================================
# 8. DETAILED DOWNSTREAM TEST
#    Show the most relevant case:
#    U=10 m/s, WD=0 deg
# ============================================================

U_detail = 10.0
WD_detail = 0.0

detail_linear = wake_solver_linear.solve(
    ambient_u=U_detail,
    ambient_wd=WD_detail,
)

detail_sqrmax = wake_solver_sqrmax.solve(
    ambient_u=U_detail,
    ambient_wd=WD_detail,
)

ti_l = np.asarray(
    detail_linear["ti_eff"],
    dtype=float
)

ti_s = np.asarray(
    detail_sqrmax["ti_eff"],
    dtype=float
)

u_l = np.asarray(
    detail_linear["u_eff"],
    dtype=float
)


print()
print("=" * 80)
print("DETAILED ALIGNED-WAKE CASE")
print(f"U = {U_detail:.1f} m/s, WD = {WD_detail:.1f} deg")
print("=" * 80)

print(
    f"{'Turbine':<10}"
    f"{'U_eff':>12}"
    f"{'TI Linear':>14}"
    f"{'TI SqrMax':>14}"
    f"{'ΔTI':>12}"
)

print("-" * 80)

for i in range(len(ti_l)):

    print(
        f"T{i+1:02d}"
        f"{u_l[i]:>12.4f}"
        f"{ti_l[i]:>14.5f}"
        f"{ti_s[i]:>14.5f}"
        f"{ti_l[i] - ti_s[i]:>12.5f}"
    )


# ============================================================
# 9. FINAL INTERPRETATION HINT
# ============================================================

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)

mean_delta = np.mean([
    r["delta_TI_mean"]
    for r in results
])

max_delta = np.max([
    r["delta_TI_max"]
    for r in results
])

print(
    f"Average difference in mean TI across cases : "
    f"{mean_delta:.5f}"
)

print(
    f"Largest difference in maximum TI           : "
    f"{max_delta:.5f}"
)

print()
print(
    "If LinearSum consistently produces substantially "
    "higher TI than SqrMaxSum while ΔU ≈ 0, the STF "
    "added-turbulence superposition is a major sensitivity "
    "in this model."
)

print(
    "If the difference appears only for aligned wake "
    "conditions, the effect is strongly dependent on "
    "wake overlap/alignment and should not be generalized "
    "from the single U=10, WD=0 test."
)

sm_crossval = pd.read_csv("/content/drive/MyDrive/1openfast_data/results_all.csv")

def parse_features(x):
    """
    Convert feature representation into a canonical tuple.

    Examples
    --------
    '[u, shear, TI]' -> ('u', 'shear', 'TI')
    ['u', 'TI']      -> ('u', 'TI')
    """

    if isinstance(x, (list, tuple)):
        features = list(x)

    elif isinstance(x, str):

        x = x.strip()

        # Try Python literal format first
        try:
            features = ast.literal_eval(x)

            if not isinstance(features, (list, tuple)):
                features = [features]

        except (ValueError, SyntaxError):

            # Handle format such as:
            # [u, shear, TI]
            x = x.strip("[]")
            features = [
                item.strip().strip("'").strip('"')
                for item in x.split(",")
            ]

    else:
        features = [str(x)]

    # Canonical ordering
    feature_order = {
        "u": 0,
        "shear": 1,
        "TI": 2
    }

    features = sorted(
        features,
        key=lambda v: feature_order.get(v, 99)
    )

    return tuple(features)

def select_best_by_feature_set(
    results_df,
    target_col="target",
    feature_col="features",
    r2_col="R2",
    rmse_col="RMSE"
):
    """
    Select the best surrogate configuration for every
    target-feature-set combination.

    Best configuration:
        1. Lowest RMSE
        2. Highest R2 if RMSE is tied
    """

    df = results_df.copy()

    # Parse feature sets
    df["feature_tuple"] = df[feature_col].apply(parse_features)

    # Sort by:
    # target
    # feature set
    # lowest RMSE
    # highest R2
    df = df.sort_values(
        by=[
            target_col,
            "feature_tuple",
            rmse_col,
            r2_col
        ],
        ascending=[
            True,
            True,
            True,
            False
        ]
    )

    # Keep best model for each target-feature combination
    best_df = (
        df.groupby(
            [
                target_col,
                "feature_tuple"
            ],
            as_index=False
        )
        .first()
    )

    # Paper-friendly feature label
    best_df["Feature set"] = best_df[
        "feature_tuple"
    ].apply(
        lambda x: ", ".join(x)
    )

    return best_df

import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", None)

best_df = select_best_by_feature_set(sm_crossval)

best_df

comparison_feature_sets = [
    ("u", "TI"),
    ("u", "shear", "TI"),
]

comparison_df = best_df[
    best_df["feature_tuple"].isin(comparison_feature_sets)
].copy()

display(
    comparison_df[
        [
            "target",
            "Feature set",
            "model",
            "params",
            "R2",
            "RMSE"
        ]
    ]
)

import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_reference_operating_conditions(
    wind_climate,
    wake_solver,
    samples_per_month=10000,
    seed=42,
):
    """
    Generate the reference operating-condition distribution
    (u_eff, ti_eff) using the same WindClimate and WakeSolver
    employed during RL.

    One record is stored for every turbine at every sampled
    operating condition.
    """

    rng = np.random.default_rng(seed)
    records = []
    for month in range(1, 13):

        for _ in tqdm(
            range(samples_per_month),
            desc=f"Month {month}"
        ):

            # Sample ambient operating condition
            ambient_u, ambient_wd = wind_climate.sample(
                month=month,
                rng=rng,
            )

            # Wake simulation
            wake = wake_solver.solve(
                ambient_u=ambient_u,
                ambient_wd=ambient_wd,
            )

            # Store every turbine
            for wt in range(wake_solver.n_turbines):

                records.append({
                    "month": month,
                    "turbine": wt,
                    "ambient_u": float(ambient_u),
                    "ambient_wd": float(ambient_wd),
                    "u_eff": float(wake["u_eff"][wt]),
                    "ti_eff": float(wake["ti_eff"][wt]),
                    "power": float(wake["power"][wt]),
                })

    return pd.DataFrame(records)

# df_wind_ref = generate_reference_operating_conditions(
#     wind_climate,
#     wake_solver,
#     samples_per_month=10000,
# )

df_wind_ref = pd.read_csv("/content/drive/MyDrive/1openfast_data/operation_wind.csv", index_col=0)

df_wind_ref.describe()

pred = damage_solver.predict_del(
    u_eff=df_wind_ref["u_eff"].to_numpy(),
    ti_eff=df_wind_ref["ti_eff"].to_numpy(),
)
df_wind_ref["del_flap"] = pred["del_flap"]
df_wind_ref["del_edge"] = pred["del_edge"]

import numpy as np

def calculate_reference_del(
    del_values,
    m=10,
):
    """
    Calculate reference DEL from Monte Carlo operational samples.

    Parameters
    ----------
    del_values : array-like
        DEL values predicted under operational conditions.
    m : int
        Wöhler exponent.

    Returns
    -------
    float
        Reference DEL.
    """
    del_values = np.asarray(del_values, dtype=float)

    return np.mean(del_values**m)**(1/m)

DEL_REF_FLAP = calculate_reference_del(
    df_wind_ref["del_flap"]
)

DEL_REF_EDGE = calculate_reference_del(
    df_wind_ref["del_edge"]
)

print(DEL_REF_FLAP)
print(DEL_REF_EDGE)

monthly_damage_share = (
    df_wind_ref
    .groupby("month")[["R_flap", "R_edge"]]
    .sum()
)

monthly_damage_share["flap_share"] = (
    monthly_damage_share["R_flap"]
    /
    (
        monthly_damage_share["R_flap"]
        + monthly_damage_share["R_edge"]
    )
)

monthly_damage_share["edge_share"] = (
    monthly_damage_share["R_edge"]
    /
    (
        monthly_damage_share["R_flap"]
        + monthly_damage_share["R_edge"]
    )
)

display(
    monthly_damage_share[
        ["flap_share", "edge_share"]
    ]
)

def calculate_damage_increment(
    DEL,
    DEL_REF,
    m=10,
    design_life=20, # years
    simulation_time=10, # minutes
):
    n_ref = design_life * 365 * 24 * 60 / simulation_time # number of cycles allowed
    damage_increment = ((DEL/DEL_REF)**m)/n_ref
    return damage_increment

df_wind_ref['delta_damage_flap'] = df_wind_ref['del_flap'].apply(lambda x: calculate_damage_increment(x, DEL_REF_FLAP, m=10, design_life=20, simulation_time=10))
df_wind_ref['delta_damage_edge'] = df_wind_ref['del_edge'].apply(lambda x: calculate_damage_increment(x, DEL_REF_EDGE, m=10, design_life=20, simulation_time=10))

import numpy as np

def calculate_damage_weights(
    flap_damage,
    edge_damage,
):
    """
    Calculate fatigue contribution weights from
    Monte Carlo operational samples.
    """

    E_df = np.mean(flap_damage)
    E_de = np.mean(edge_damage)

    w_f = E_df / (E_df + E_de)
    w_e = E_de / (E_df + E_de)

    return {
        "E_df": E_df,
        "E_de": E_de,
        "w_f": w_f,
        "w_e": w_e,
    }

weights = calculate_damage_weights(
    df_wind_ref["delta_damage_flap"],
    df_wind_ref["delta_damage_edge"],
)
weights

base_df = pd.read_csv("/content/drive/MyDrive/1openfast_data/NREL5MW_Sobol_u_IT_alpha/merged_wind_DEL.csv", index_col=0)

u_min, u_max = base_df["u"].min(), base_df["u"].max()
ti_min, ti_max = base_df["TI"].min(), base_df["TI"].max()

print(u_min)
print(u_max)
print(ti_min)
print(ti_max)

u_grid = np.linspace(
    u_min,
    u_max,
    200
)

ti_grid = np.linspace(
    ti_min,
    ti_max,
    200
)

UU, TT = np.meshgrid(
    u_grid,
    ti_grid
)

X_grid = np.column_stack([
    UU.ravel(),
    TT.ravel()
])

print(X_grid.shape)

# Predict DEL on grid
import joblib

loaded_flap_model = joblib.load('/content/drive/MyDrive/1openfast_data/trained_model/flap_model.pkl')
loaded_edge_model = joblib.load('/content/drive/MyDrive/1openfast_data/trained_model/edge_model.pkl')

del_flap_grid = loaded_flap_model["model"].predict(X_grid)
del_edge_grid = loaded_edge_model["model"].predict(X_grid)

print(del_flap_grid.shape)
print(del_edge_grid.shape)

def calculate_damage_increment(
    DEL,
    DEL_REF,
    m=10,
    design_life=20, # years
    simulation_time=10, # minutes
):
    n_ref = design_life * 365 * 24 * 60 / simulation_time # number of cycles allowed
    damage_increment = ((DEL/DEL_REF)**m)/n_ref
    return damage_increment


dmg_flap_grid = calculate_damage_increment(
    del_flap_grid,
    DEL_REF=2803.716141751299,
    m=10,
    design_life=20,
    simulation_time=10
)

dmg_edge_grid = calculate_damage_increment(
    del_edge_grid,
    DEL_REF=5588.786717232858,
    m=10,
    design_life=20,
    simulation_time=10
)

weight = 0.5

total_damage_grid = (
    weight * dmg_flap_grid
    +
    weight * dmg_edge_grid
)

print(total_damage_grid.min())
print(total_damage_grid.max())
print(total_damage_grid.mean())

flap_damage_surface = dmg_flap_grid.reshape(UU.shape)
edge_damage_surface = dmg_edge_grid.reshape(UU.shape)
total_damage_surface = total_damage_grid.reshape(UU.shape)

print(flap_damage_surface.shape)
print(edge_damage_surface.shape)
print(total_damage_surface.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

contour = plt.contourf(
    UU,
    TT,
    np.log10(flap_damage_surface),
    levels=30
)

plt.colorbar(
    contour,
    label="log10 (Damage Increment)"
)

plt.xlabel("Wind Speed (m/s)")
plt.ylabel("TI")
plt.title("Scenario-based Flapwise Damage Characterization")

plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

contour = plt.contourf(
    UU,
    TT,
    np.log10(edge_damage_surface),
    levels=30
)

plt.colorbar(
    contour,
    label="log10 (Damage Increment)"
)

plt.xlabel("Wind Speed (m/s)")
plt.ylabel("TI")
plt.title("Scenario-based Edgewise Damage Characterization")

plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

contour = plt.contourf(
    UU,
    TT,
    np.log10(total_damage_surface),
    levels=30
)

plt.colorbar(
    contour,
    label="log10 (Damage Increment)"
)

plt.xlabel("Wind Speed (m/s)")
plt.ylabel("TI")
plt.title("Scenario-based Total Damage Characterization")

plt.show()

import pandas as pd

response_surface_df = pd.DataFrame({
    "u": UU.ravel(),
    "ti": TT.ravel(),
    "del_flap": del_flap_grid,
    "del_edge": del_edge_grid,
    "damage_flap": dmg_flap_grid,
    "damage_edge": dmg_edge_grid,
    "damage_total": total_damage_grid
})

# response_surface_df.to_csv('/content/drive/MyDrive/1openfast_data/response_surface_df_200.csv', index=False)

import numpy as np
import matplotlib.pyplot as plt


def plot_damage_response_surface(
    df,
    damage_column="damage_total",
    n_levels=30,
    log_scale=True,
    figsize=(8, 5),
):
    surface = df.pivot(
        index="ti",
        columns="u",
        values=damage_column
    )

    u_grid = surface.columns.to_numpy()

    # Convert TI from fraction to percentage
    ti_grid = surface.index.to_numpy() * 100

    damage_grid = surface.to_numpy()

    if log_scale:
        plot_grid = np.log10(np.maximum(damage_grid, 1e-30))
        color_label = r"$\log_{10}(D)$"
    else:
        plot_grid = damage_grid
        color_label = "Fatigue damage"

    U, TI = np.meshgrid(u_grid, ti_grid)

    fig, ax = plt.subplots(figsize=figsize)

    contour = ax.contourf(
        U,
        TI,
        plot_grid,
        levels=n_levels,
    )

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(color_label)

    ax.set_xlabel(r"Wind speed, $u$ [m/s]")
    ax.set_ylabel(r"Turbulence intensity, $TI$ [%]")

    ax.set_title("")

    plt.tight_layout()
    plt.show()

    return fig, ax

response_surface_df = pd.read_csv('/content/drive/MyDrive/1openfast_data/response_surface_df_200.csv')

plot_damage_response_surface(
    response_surface_df,
    damage_column="damage_total",
    figsize=(6, 5),
)

env = OffshoreMaintenanceEnv(
    env_config
)

obs, info = env.reset(
    seed=env_config.seed,
    options={
        "randomize_maintenance_params": True
    }
)

print(obs)
print(info)

import numpy as np
import pandas as pd


def test_wind_sampling_resolution(
    wind_climate,
    config,
    n_samples_list=(1, 3, 6, 12, 24, 30, 60),
    n_repetitions=100,
    seed=42,
):
    """
    Test convergence of monthly wind sampling resolution.

    Each repetition represents one independent realization
    of N wind time-slices within each month.
    """

    records = []

    for n_samples in n_samples_list:

        for month in range(1, 13):

            for repetition in range(n_repetitions):

                rng = np.random.default_rng(
                    seed + repetition
                )

                wind_climate.start_year(rng)

                wind_speeds = []
                wind_directions = []

                for _ in range(n_samples):

                    u, wd = wind_climate.sample(
                        month=month,
                        rng=rng,
                    )

                    wind_speeds.append(u)
                    wind_directions.append(wd)

                records.append({
                    "n_samples": n_samples,
                    "month": month,
                    "repetition": repetition,
                    "mean_u": np.mean(wind_speeds),
                    "std_u": np.std(wind_speeds),
                    "mean_wd": np.mean(wind_directions),
                })

    return pd.DataFrame(records)

df_resolution = test_wind_sampling_resolution(
    wind_climate=wind_climate,
    config=env_config,
    n_samples_list=[1, 3, 6, 12, 24, 30, 60, 120, 30*24, 30*24*6],
    n_repetitions=100,
    seed=42,
)

summary = (
    df_resolution
    .groupby("n_samples")
    [["mean_u", "std_u"]]
    .agg(["mean", "std"])
)

print(summary)

def always_continue_policy(
    observation,
    env,
):
    return np.zeros(
        env.n_turbines,
        dtype=np.int32,
    )

def always_maintenance_policy(
    observation,
    env,
):
    return np.ones(
        env.n_turbines,
        dtype=np.int32,
    )

def random_policy(
    observation,
    env,
    policy_rng=np.random.default_rng(seed=42)
):
    return policy_rng.integers(
        0,
        2,
        size=env.n_turbines,
        dtype=np.int32,
    )

from tqdm.auto import tqdm


def run_episode(
    env,
    policy,
    seed=42,
    stop_condition=None,
    randomize_maintenance_params=False,
    show_progress=False,
):
    observation, info = env.reset(
        seed=seed,
        options={
            "randomize_maintenance_params": randomize_maintenance_params
        }
    )

    terminated = False
    truncated = False

    progress = tqdm(
        total=env.max_steps,
        desc="Running episode",
        unit="step",
        disable=not show_progress,
    )

    while not (terminated or truncated):

        action = policy(
            observation,
            env,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        progress.update(1)

        if (
            stop_condition is not None
            and stop_condition(env)
        ):
            break

    progress.close()

    return env.logger.dataframe()

seed=env_config.seed

history_continue = run_episode(
    env,
    always_continue_policy,
    seed,
    randomize_maintenance_params=True,
    show_progress=True,
)

history_maintenance = run_episode(
    env,
    always_maintenance_policy,
    seed,
    randomize_maintenance_params=True,
    show_progress=True,
)

history_random = run_episode(
    env,
    random_policy,
    seed,
    randomize_maintenance_params=True,
    show_progress=True,
)

history_continue.to_pickle('/content/drive/MyDrive/1openfast_data/log/simulation_test/history_continue_wakeoff.pkl')
history_maintenance.to_pickle('/content/drive/MyDrive/1openfast_data/log/simulation_test/history_maintenance_wakeoff.pkl')
history_random.to_pickle('/content/drive/MyDrive/1openfast_data/log/simulation_test/history_random_wakeoff.pkl')

from collections import Counter

dfs = {
    "history_maintenance": history_maintenance,
    "history_continue": history_continue,
    "history_random": history_random,
}

for df_name, df_data in dfs.items():
    print(f"--- {df_name} ---")
    maintenance_types = df_data[df_data['decision_event'] == True]['maintenance_type']

    # Flatten the list of lists into a single list
    flattened_maintenance_types = [item for sublist in maintenance_types if sublist is not None for item in sublist]

    # Count the occurrences of each maintenance type
    maintenance_counts = Counter(flattened_maintenance_types)

    print(maintenance_counts)
    print("\n")

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, Colormap
from matplotlib.ticker import MultipleLocator, FuncFormatter


def plot_spatiotemporal_evolution(
    data_column,
    x,
    y,
    decision_interval_months=1,
    snapshot_years=(0,5,10,15,20),
    variable_name="Health Index",
    title="None",
    title_prefix="",
    cmap="RdYlGn",
    reverse_cmap=False,
    marker_size=500,
    outside_marker_size=100,
    label_position="inside",
    label_offset=120,
    annotate=True,
    show_axes=True,
    save_snapshot=False,
    snapshot_name="snapshot.png",
):
    array=np.array(data_column.tolist()).T
    series=np.asarray(array)
    x=np.asarray(x)
    y=np.asarray(y)

    n_turbines,n_steps=series.shape

    if len(x)!=n_turbines or len(y)!=n_turbines:
        raise ValueError("x/y must have same length as number of turbines.")

    if label_position not in ["inside","outside"]:
        raise ValueError("label_position must be 'inside' or 'outside'.")

    years=np.arange(n_steps)*decision_interval_months/12
    vmin=np.min(series)
    vmax=np.max(series)

    if reverse_cmap:
        if isinstance(cmap,str):
            cmap=cmap+"_r"
        elif isinstance(cmap,Colormap):
            if isinstance(cmap,LinearSegmentedColormap):
                cmap=LinearSegmentedColormap.from_list(
                    cmap.name+"_r",
                    cmap.colors[::-1]
                )
            else:
                try:
                    cmap=plt.colormaps[cmap.name+"_r"]
                except KeyError:
                    pass

    plot_marker_size=(
        marker_size
        if label_position=="inside"
        else outside_marker_size
    )

    margin=700

    fig,axes=plt.subplots(
        1,
        len(snapshot_years),
        figsize=(4.2*len(snapshot_years),4.8),
        constrained_layout=True,
    )

    fig.suptitle(title,y=0.92)

    if len(snapshot_years)==1:
        axes=[axes]

    for ax,year in zip(axes,snapshot_years):
        idx=np.argmin(np.abs(years-year))

        sc=ax.scatter(
            x,
            y,
            c=series[:,idx],
            cmap=cmap,
            s=plot_marker_size,
            edgecolors="black",
            linewidths=0.8,
            vmin=vmin,
            vmax=vmax,
            zorder=5,
        )

        for i in range(n_turbines):
            if np.isclose(series[i,idx],0.0):
                ax.plot(
                    x[i],
                    y[i],
                    marker="x",
                    color="red",
                    markersize=9,
                    markeredgewidth=2,
                    zorder=12,
                )

        if annotate:
            for i,(xx,yy) in enumerate(zip(x,y)):
                if label_position=="inside":
                    ax.text(
                        xx,
                        yy,
                        str(i+1),
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="regular",
                        color="black",
                        zorder=11,
                    )
                else:
                    ax.text(
                        xx+label_offset,
                        yy+label_offset,
                        str(i+1),
                        ha="left",
                        va="bottom",
                        fontsize=9,
                        fontweight="regular",
                        color="black",
                        zorder=11,
                    )

        ax.set_title(
            f"{title_prefix}\nYear {years[idx]:.1f}"
            if title_prefix
            else f"Year {years[idx]:.1f}"
        )

        ax.set_aspect("equal",adjustable="box")
        ax.set_xlim(x.min()-margin,x.max()+margin)
        ax.set_ylim(y.min()-margin,y.max()+margin)

        if show_axes:
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    cbar=fig.colorbar(
        sc,
        ax=axes,
        shrink=0.66,
    )

    cbar.set_label(variable_name)

    if save_snapshot:
        fig.savefig(
            snapshot_name,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_HI(
    data_columns_dict: dict[str, pd.DataFrame],
    column_name: str,
    turbine_id: int,
    ylim_max: float,
    repair_threshold: float = 0.85,
    replacement_threshold: float = 0.10,
    plot_baseline: bool = True,
    plot_years: float | None = None,
    show_title: bool = True,
    show_thresholds: bool = True,
    figsize: tuple[int, int] = (8, 4),
):
    plt.figure(figsize=figsize)

    first_key = next(iter(data_columns_dict))
    elapsed_months = data_columns_dict[first_key]['elapsed_month']

    if plot_years is None:
        max_plot_months = max(elapsed_months)
    else:
        max_plot_months = plot_years * 12

    if plot_baseline:
        baseline_x_months = np.arange(
            0,
            max_plot_months + 1,
            1
        )

        baseline_years = baseline_x_months / 12

        # Linear degradation from HI = 1 to HI = 0
        # over the 20-year design life.
        # After 20 years, HI remains at 0.
        baseline_hi_values = np.maximum(
            1 - baseline_years / 20,
            0
        )

        plt.plot(
            baseline_x_months,
            baseline_hi_values,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            label='Reference Degradation'
        )

    for label, history_df in data_columns_dict.items():

        array = np.array(
            history_df[column_name].tolist()
        ).T

        # Convert turbine ID (1-based) to array index (0-based)
        turbine_index = turbine_id - 1

        if turbine_index < 0 or turbine_index >= array.shape[0]:
            raise ValueError(
                f"Invalid turbine_id={turbine_id}. "
                f"Available turbine IDs are "
                f"1 to {array.shape[0]}."
            )

        hi_single_turbine = array[turbine_index]

        plt.plot(
            elapsed_months,
            hi_single_turbine,
            linestyle='-',
            linewidth=1.5,
            label=label
        )

    if show_thresholds:

        plt.axhline(
            y=repair_threshold,
            color='tab:cyan',
            linestyle='--',
            linewidth=1.2,
            label=f'Repair Threshold'
        )

        plt.axhline(
            y=replacement_threshold,
            color='tab:red',
            linestyle='--',
            linewidth=1.2,
            label=f'Replacement Threshold'
        )

    if show_title:
        plt.title(
            f'Health Index (HI) over time for Turbine {turbine_id}'
        )

    plt.xlabel('Elapsed Years')
    plt.ylabel('Health Index (HI)')

    plt.ylim(0, ylim_max)
    plt.xlim(0, max_plot_months)

    ax = plt.gca()

    ax.xaxis.set_major_locator(
        MultipleLocator(12)
    )

    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda x, pos: f'{x / 12:.0f}'
        )
    )
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.show()

import pandas as pd

#Load simulated dataset
history_continue = pd.read_pickle('/content/drive/MyDrive/1openfast_data/log/simulation_test/history_continue_wakeoff.pkl')
history_maintenance = pd.read_pickle('/content/drive/MyDrive/1openfast_data/log/simulation_test/history_maintenance_wakeoff.pkl')
history_random = pd.read_pickle('/content/drive/MyDrive/1openfast_data/log/simulation_test/history_random_wakeoff.pkl')

print(history_continue['HI_turbine'].iloc[0])
print(type(history_continue['HI_turbine'].iloc[0]))

from matplotlib.colors import LinearSegmentedColormap

# Define white to gray (or dark gray/light gray)
colors = ["gray", "white"]

# Create the custom colormap
white_gray_cmap = LinearSegmentedColormap.from_list("white_gray", colors)

x_max = env_config.max_simulation_years+1
snapshot_years=np.arange(0,x_max,x_max//5)

plot_spatiotemporal_evolution(
    data_column=history_continue['HI_turbine'],
    x=x_layout,
    y=y_layout,
    snapshot_years=snapshot_years,
    title='No Maintenance Policy',
    cmap='gray',
    label_position='outside'
)
plot_spatiotemporal_evolution(
    data_column=history_maintenance['HI_turbine'],
    x=x_layout,
    y=y_layout,
    snapshot_years=snapshot_years,
    title='Always Maintenance Policy',
    cmap='gray',
    label_position='outside'
)
plot_spatiotemporal_evolution(
    data_column=history_random['HI_turbine'],
    x=x_layout,
    y=y_layout,
    snapshot_years=snapshot_years,
    title='Random Maintenance Policy',
    cmap='gray',
    label_position='outside'
)

data_to_plot = {
    'Random Policy': history_random,
    'Regular Maintenance': history_maintenance,
    'No Maintenance': history_continue
}

plot_HI(
    data_columns_dict=data_to_plot,
    column_name='HI_turbine',
    turbine_id=5,
    ylim_max=1.0,
    plot_years=20,
    repair_threshold=0.85,
    replacement_threshold=0.10,
    plot_baseline=True,
    show_title=True,
    show_thresholds=True,
    figsize=(8, 5)
)

plt.figure(figsize=(10, 6))
history_continue[history_continue['decision_event'] == True]['reward'].cumsum().plot(label='Continue Policy')
history_maintenance[history_maintenance['decision_event'] == True]['reward'].cumsum().plot(label='Always Maintenance Policy')
history_random[history_random['decision_event'] == True]['reward'].cumsum().plot(label='Random Policy')
plt.title('Cumulative Reward Comparison')
plt.xlabel('Decision Events')
plt.ylabel('Cumulative Reward (€)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt


def plot_scalarized_reward(
    history_continue,
    history_random,
    history_maintenance,
    damage_scale,
    spatial_scale,
    damage_weight=1.0,
    spatial_weight=1.0,
):
    histories = {
        "Continue Policy": history_continue,
        "Random Policy": history_random,
        "Always Maintenance Policy": history_maintenance,
    }

    plt.figure(figsize=(10, 6))

    for label, df in histories.items():

        decision_df = df.loc[
            df["decision_event"] == True
        ].copy()

        # ------------------------------------------------------
        # Normalized objectives
        # ------------------------------------------------------
        decision_df["damage_objective"] = (
            decision_df["interval_damage_burden"]
            / damage_scale
        )

        decision_df["spatial_objective"] = (
            decision_df["spatial_grouping"]
            / spatial_scale
        )

        # ------------------------------------------------------
        # Scalar reward
        # ------------------------------------------------------
        decision_df["damage_reward"] = (
            -damage_weight
            * decision_df["damage_objective"]
        )

        decision_df["spatial_reward"] = (
            -spatial_weight
            * decision_df["spatial_objective"]
        )

        decision_df["scalar_reward"] = (
            decision_df["damage_reward"]
            + decision_df["spatial_reward"]
        )

        # ------------------------------------------------------
        # Cumulative reward
        # ------------------------------------------------------
        decision_df["cumulative_reward"] = (
            decision_df["scalar_reward"].cumsum()
        )

        plt.plot(
            decision_df["elapsed_month"],
            decision_df["cumulative_reward"],
            label=label,
        )

    plt.xlabel("Elapsed Month")
    plt.ylabel("Cumulative Reward")
    plt.title(
        f"Scalarized Cumulative Reward "
        f"($w_D$={damage_weight:.2f}, $w_S$={spatial_weight:.2f})"
    )

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

w1 = 0.85
w2 = 1 - w1
plot_scalarized_reward(
    history_continue,
    history_random,
    history_maintenance,
    damage_scale=25*30,
    spatial_scale=7670.692556,
    damage_weight=w1,
    spatial_weight=w2,
)

history_continue.columns

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

rng = np.random.default_rng(42)
seeds = [int(seed) for seed in rng.integers(0, 5000, size=10)]

print("Seeds:", seeds)

output_dir = Path("/content/drive/MyDrive/1openfast_data/log/simulation_test/no_maintenance")
output_dir.mkdir(parents=True, exist_ok=True)

for seed in seeds:
    print(f"\nRunning seed = {seed}")
    history_continue = run_episode(
        env,
        always_continue_policy,
        seed,
        randomize_maintenance_params=True,
        show_progress=True,
    )

    if not isinstance(history_continue, pd.DataFrame):
        raise TypeError(
            f"Expected pandas.DataFrame, "
            f"got {type(history_continue)}"
        )

    output_path = output_dir / f"history_no_maintenance_seed_{seed}.pkl"
    history_continue.to_pickle(output_path)
    print(f"Saved: {output_path}")

from pathlib import Path
import pandas as pd


def load_histories_from_folder(
    folder_path,
    pattern="history_no_maintenance_seed_*.pkl",
):
    folder = Path(folder_path)

    files = sorted(folder.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {folder}"
        )

    histories = {}

    for file_path in files:
        history = pd.read_pickle(file_path)

        # Extract seed from filename
        seed = int(
            file_path.stem.split("_seed_")[-1]
        )

        histories[seed] = history

    print(f"Loaded {len(histories)} trajectories.")

    return histories

history_no_maintenance = load_histories_from_folder(
    "/content/drive/MyDrive/1openfast_data/log/simulation_test/no_maintenance/"
)

plot_HI(
    data_columns_dict=history_no_maintenance,
    column_name='HI_turbine',
    turbine_id=15,
    ylim_max=1.0,
    repair_threshold=0.85,
    replacement_threshold=0.10,
    plot_baseline=True,
    show_title=False,
    show_thresholds=False,
    figsize=(8, 5)
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma


# ============================================================
# Monthly Weibull parameters
# ============================================================

wind_monthly_weibull = {
    1:  {"k": 2.2, "c": 13.0},
    2:  {"k": 2.2, "c": 12.5},
    3:  {"k": 2.2, "c": 11.5},
    4:  {"k": 2.2, "c": 10.5},
    5:  {"k": 2.2, "c": 9.5},
    6:  {"k": 2.2, "c": 8.5},
    7:  {"k": 2.2, "c": 8.0},
    8:  {"k": 2.2, "c": 8.5},
    9:  {"k": 2.2, "c": 9.5},
    10: {"k": 2.2, "c": 11.0},
    11: {"k": 2.2, "c": 12.0},
    12: {"k": 2.2, "c": 13.0},
}


# ============================================================
# Settings
# ============================================================

samples_per_month_list = [10, 30, 100, 300, 1000, 4320]
seeds = range(1, 21)   # 20 independent seeds


# ============================================================
# Theoretical Weibull statistics
# ============================================================

def weibull_theoretical_statistics(k, c):
    """
    Analytical statistics for Weibull(k, c).

    NumPy parameterization:
        X = c * Weibull(k)
    """

    mean = c * gamma(1.0 + 1.0 / k)

    variance = (
        c**2
        * (
            gamma(1.0 + 2.0 / k)
            - gamma(1.0 + 1.0 / k)**2
        )
    )

    p95 = c * (-np.log(0.05))**(1.0 / k)

    return {
        "mean_u": mean,
        "var_u": variance,
        "p95_u": p95,
    }


# ============================================================
# Convergence experiment
# ============================================================

results = []

for seed in seeds:

    rng = np.random.default_rng(seed)

    for n_s in samples_per_month_list:

        for month in range(1, 13):

            k = wind_monthly_weibull[month]["k"]
            c = wind_monthly_weibull[month]["c"]

            # ------------------------------------------------
            # Generate samples
            # ------------------------------------------------
            u = c * rng.weibull(k, size=n_s)

            # ------------------------------------------------
            # Sample statistics
            # ------------------------------------------------
            sample_mean = np.mean(u)

            sample_variance = np.var(
                u,
                ddof=1
            )

            sample_p95 = np.percentile(
                u,
                95
            )

            # ------------------------------------------------
            # Theoretical statistics
            # ------------------------------------------------
            theoretical = weibull_theoretical_statistics(
                k,
                c
            )

            # ------------------------------------------------
            # Relative errors
            # ------------------------------------------------
            mean_error = (
                abs(sample_mean - theoretical["mean_u"])
                / theoretical["mean_u"]
                * 100
            )

            variance_error = (
                abs(sample_variance - theoretical["var_u"])
                / theoretical["var_u"]
                * 100
            )

            p95_error = (
                abs(sample_p95 - theoretical["p95_u"])
                / theoretical["p95_u"]
                * 100
            )

            results.append({
                "seed": seed,
                "n_s": n_s,
                "month": month,

                "sample_mean_u": sample_mean,
                "sample_variance_u": sample_variance,
                "sample_p95_u": sample_p95,

                "theoretical_mean_u":
                    theoretical["mean_u"],

                "theoretical_variance_u":
                    theoretical["var_u"],

                "theoretical_p95_u":
                    theoretical["p95_u"],

                "mean_error_pct": mean_error,
                "variance_error_pct": variance_error,
                "p95_error_pct": p95_error,
            })


convergence_df = pd.DataFrame(results)

convergence_df.head()

plt.figure(figsize=(9, 6))

data = [
    convergence_df.loc[
        convergence_df["n_s"] == n_s,
        "variance_error_pct"
    ]
    for n_s in samples_per_month_list
]

plt.boxplot(
    data,
    labels=samples_per_month_list
)

plt.xlabel("Samples per month")
plt.ylabel("Absolute relative error of variance [%]")
plt.title("Sampling convergence of wind-speed variance")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()

summary = (
    convergence_df
    .groupby("n_s")
    .agg(
        mean_mean_error_pct=(
            "mean_error_pct",
            "mean"
        ),
        median_mean_error_pct=(
            "mean_error_pct",
            "median"
        ),
        max_mean_error_pct=(
            "mean_error_pct",
            "max"
        ),

        mean_variance_error_pct=(
            "variance_error_pct",
            "mean"
        ),
        median_variance_error_pct=(
            "variance_error_pct",
            "median"
        ),
        max_variance_error_pct=(
            "variance_error_pct",
            "max"
        ),

        mean_p95_error_pct=(
            "p95_error_pct",
            "mean"
        ),
        median_p95_error_pct=(
            "p95_error_pct",
            "median"
        ),
        max_p95_error_pct=(
            "p95_error_pct",
            "max"
        ),
    )
    .reset_index()
)

summary[
    [
        "n_s",
        "median_mean_error_pct",
        "median_variance_error_pct",
        "median_p95_error_pct",
    ]
]

from copy import deepcopy
from dataclasses import replace


def build_sensitivity_config(
    base_config,
    *,
    repair_threshold=0.80,
    repair_damage_multiplier=0.50,
    repair_duration_months=15,
):
    """
    Build deterministic configuration for sensitivity analysis.

    The complete farm/environment configuration is inherited from
    base_config.

    Only maintenance behavior is changed:
        - repair becomes deterministic
        - replacement is inherited unchanged

    Farm layout, number of turbines, wake solver, weather,
    physical parameters, reward configuration, etc. are preserved.
    """

    config = deepcopy(base_config)

    # ========================================================
    # Build deterministic repair
    # ========================================================

    repair = MaintenanceType(
        name="repair",
        threshold=repair_threshold,
        cost=2000,
        downtime_hours=8,
        carbon_emission=200,
        damage_multiplier=repair_damage_multiplier,
        duration_months=repair_duration_months,
        is_replacement=False,
    )

    # ========================================================
    # Inherit replacement from base_config
    # ========================================================

    base_maintenance_types = config.maintenance_policy.config.maintenance_types

    replacement_types = [
        m
        for m in base_maintenance_types
        if m.is_replacement
    ]

    if len(replacement_types) != 1:
        raise ValueError(
            "Expected exactly one replacement "
            "maintenance type in base_config."
        )

    replacement = replacement_types[0]

    # ========================================================
    # Deterministic SA maintenance policy
    # ========================================================

    maintenance_config = MaintenanceConfig(
        maintenance_types=(
            repair,
            replacement,
        )
    )

    maintenance_policy = MaintenancePolicy(
        maintenance_config
    )

    # ========================================================
    # Maximum protection duration
    # ========================================================

    max_protection_duration = max(
        m.duration_months
        for m in maintenance_config.maintenance_types
    )

    # ========================================================
    # Create SA config
    # ========================================================

    return replace(
        config,
        maintenance_policy=maintenance_policy,
        max_protection_duration=max_protection_duration,
        wind_time_slices_per_month=1,
        max_simulation_years=30,
    )

from copy import deepcopy
from dataclasses import replace
from itertools import product

import pandas as pd
from tqdm.auto import tqdm


def _is_replacement_event(value):
    if value is None:
        return False

    if isinstance(value, list):
        return any(
            m is not None and str(m).lower() == "replacement"
            for m in value
        )

    return str(value).lower() == "replacement"


def sensitivity_analysis_2d(
    base_config,
    policy,
    baseline,
    thresholds,
    damage_multipliers,
    durations,
    decision_interval_months,
    seeds,
    parameter_1,
    parameter_2,
):
    """
    Run a 2D sensitivity analysis.

    Only parameter_1 and parameter_2 are varied.
    All remaining parameters use baseline values.

    Health metrics are evaluated only over the first turbine
    lifetime, i.e. before the first replacement.

    If no replacement occurs, the complete simulation history
    is used and the observation is treated as right-censored.
    """

    # ========================================================
    # Parameter definitions
    # ========================================================

    parameter_values = {
        "threshold": list(thresholds),
        "multiplier": list(damage_multipliers),
        "duration": list(durations),
        "decision_interval_months": list(
            decision_interval_months
        ),
    }

    # ========================================================
    # Validate parameters
    # ========================================================

    if parameter_1 not in parameter_values:
        raise ValueError(
            f"Invalid parameter_1='{parameter_1}'. "
            f"Choose from: {list(parameter_values)}"
        )

    if parameter_2 not in parameter_values:
        raise ValueError(
            f"Invalid parameter_2='{parameter_2}'. "
            f"Choose from: {list(parameter_values)}"
        )

    if parameter_1 == parameter_2:
        raise ValueError(
            "parameter_1 and parameter_2 must be different."
        )

    required_baseline = set(parameter_values)

    missing_baseline = (
        required_baseline
        - set(baseline.keys())
    )

    if missing_baseline:
        raise ValueError(
            f"Missing baseline parameters: "
            f"{sorted(missing_baseline)}"
        )

    seeds = list(seeds)

    if not seeds:
        raise ValueError("seeds cannot be empty.")

    # ========================================================
    # Generate selected 2D scenarios
    # ========================================================

    values_1 = parameter_values[parameter_1]
    values_2 = parameter_values[parameter_2]

    scenarios = product(values_1, values_2)

    total_runs = (
        len(values_1)
        * len(values_2)
        * len(seeds)
    )

    results = []

    # ========================================================
    # Run experiment
    # ========================================================

    with tqdm(
        total=total_runs,
        desc=f"{parameter_1} × {parameter_2}",
        unit="episode",
    ) as pbar:

        for value_1, value_2 in scenarios:

            for seed in seeds:

                # ------------------------------------------------
                # Scenario parameters
                # ------------------------------------------------

                params = baseline.copy()
                params[parameter_1] = value_1
                params[parameter_2] = value_2

                # ------------------------------------------------
                # Copy base configuration
                # ------------------------------------------------

                config = deepcopy(base_config)

                # ------------------------------------------------
                # Override repair parameters
                # ------------------------------------------------

                maintenance_types = []
                repair_found = False

                for maintenance in (
                    config
                    .maintenance_policy
                    .config
                    .maintenance_types
                ):

                    if (
                        maintenance.name.lower()
                        == "repair"
                    ):

                        maintenance = replace(
                            maintenance,
                            threshold=params["threshold"],
                            damage_multiplier=params["multiplier"],
                            duration_months=params["duration"],
                        )

                        repair_found = True

                    maintenance_types.append(
                        maintenance
                    )

                if not repair_found:
                    raise ValueError(
                        "Repair maintenance type "
                        "was not found in base_config."
                    )

                # ------------------------------------------------
                # Rebuild maintenance policy
                # ------------------------------------------------

                maintenance_config = replace(
                    config
                    .maintenance_policy
                    .config,
                    maintenance_types=tuple(
                        maintenance_types
                    ),
                )

                maintenance_policy = MaintenancePolicy(
                    maintenance_config
                )

                # ------------------------------------------------
                # Update configuration
                # ------------------------------------------------

                max_protection_duration = max(
                    m.duration_months
                    for m in maintenance_types
                )

                config = replace(
                    config,
                    maintenance_policy=maintenance_policy,
                    max_protection_duration=(
                        max_protection_duration
                    ),
                    decision_interval_months=(
                        params[
                            "decision_interval_months"
                        ]
                    ),
                    seed=seed,
                )

                # ------------------------------------------------
                # Run environment
                # ------------------------------------------------

                env = OffshoreMaintenanceEnv(config)

                history = run_episode(
                    env=env,
                    policy=policy,
                    seed=seed,
                )

                # =================================================
                # Identify first replacement
                # =================================================

                replacement_mask = history[
                    "maintenance_type"
                ].apply(
                    _is_replacement_event
                )

                replacement_indices = history.index[
                    replacement_mask
                ]

                if len(replacement_indices) > 0:

                    first_replacement_index = (
                        replacement_indices[0]
                    )

                    replacement_position = (
                        history.index.get_loc(
                            first_replacement_index
                        )
                    )

                    # Only observations before replacement
                    life_history = history.iloc[
                        :replacement_position
                    ].copy()

                    replacement_occurred = True

                    # Logged month of first replacement
                    lifetime_before_replacement = (
                        history.loc[
                            first_replacement_index,
                            "elapsed_month",
                        ]
                        - history[
                            "elapsed_month"
                        ].iloc[0]
                    )

                else:

                    # No replacement within simulation horizon
                    life_history = history.copy()

                    replacement_occurred = False

                    # Right-censored at simulation horizon
                    lifetime_before_replacement = (
                        history[
                            "elapsed_month"
                        ].iloc[-1]
                        - history[
                            "elapsed_month"
                        ].iloc[0]
                    )

                # ------------------------------------------------
                # Extract maintenance events
                # ------------------------------------------------

                maintenance_events = []

                for row in life_history[
                    "maintenance_type"
                ]:

                    if row is None:
                        continue

                    if isinstance(row, list):
                        maintenance_events.extend(
                            m
                            for m in row
                            if m is not None
                        )
                    else:
                        maintenance_events.append(row)

                # ------------------------------------------------
                # Count maintenance before replacement
                # ------------------------------------------------

                repair_count = sum(
                    str(m).lower() == "repair"
                    for m in maintenance_events
                )

                # ------------------------------------------------
                # Health metrics
                # ------------------------------------------------

                if life_history.empty:
                    mean_HI = float("nan")
                    minimum_HI = float("nan")
                    final_HI = float("nan")
                else:
                    mean_HI = (
                        life_history["HI_mean"].mean()
                    )

                    minimum_HI = (
                        life_history["HI_min"].min()
                    )

                    final_HI = (
                        life_history[
                            "HI_mean"
                        ].iloc[-1]
                    )

                # ------------------------------------------------
                # Store results
                # ------------------------------------------------

                results.append(
                    {
                        # Experiment parameters
                        "threshold": params[
                            "threshold"
                        ],
                        "multiplier": params[
                            "multiplier"
                        ],
                        "duration": params[
                            "duration"
                        ],
                        "decision_interval_months": (
                            params[
                                "decision_interval_months"
                            ]
                        ),
                        "seed": seed,

                        # First-life health
                        "mean_HI": mean_HI,
                        "minimum_HI": minimum_HI,
                        "final_HI": final_HI,

                        # First-life maintenance
                        "repair_count": repair_count,
                        "maintenance_cost": (
                            life_history[
                                "maintenance_cost"
                            ].fillna(0).sum()
                        ),

                        # Lifetime
                        "lifetime_before_replacement": (
                            lifetime_before_replacement
                        ),
                        "replacement_occurred": (
                            replacement_occurred
                        ),

                        # Useful diagnostic
                        "n_observations": len(
                            life_history
                        ),
                    }
                )

                pbar.update(1)

    return pd.DataFrame(results)

import math
import numpy as np
import matplotlib.pyplot as plt


def plot_sensitivity_heatmaps(
    results,
    parameter_1,
    parameter_2,
    metrics,
    parameter_1_label=None,
    parameter_2_label=None,
    metric_labels=None,
    ncols=3,
    figsize_per_plot=(4.5, 3.8),
    cmap="viridis",
    fmt=".3f",
):
    """Plot multiple 2D sensitivity heatmaps in an n×n-style grid."""

    if metric_labels is None:
        metric_labels = {}

    metrics = list(metrics)

    if not metrics:
        raise ValueError("metrics cannot be empty.")

    nplots = len(metrics)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            figsize_per_plot[0] * ncols,
            figsize_per_plot[1] * nrows,
        ),
        squeeze=False,
    )

    axes = axes.ravel()

    for ax, metric in zip(axes, metrics):

        required_columns = {
            parameter_1,
            parameter_2,
            metric,
            "seed",
        }

        missing_columns = (
            required_columns
            - set(results.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing columns for '{metric}': "
                f"{sorted(missing_columns)}"
            )

        summary = (
            results
            .groupby(
                [parameter_1, parameter_2],
                as_index=False,
            )[metric]
            .mean()
        )

        pivot = (
            summary
            .pivot(
                index=parameter_1,
                columns=parameter_2,
                values=metric,
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )

        values = pivot.to_numpy(dtype=float)

        im = ax.imshow(
            values,
            aspect="auto",
            origin="lower",
            cmap=cmap,
        )

        ax.set_xticks(
            np.arange(len(pivot.columns))
        )
        ax.set_xticklabels(
            pivot.columns
        )

        ax.set_yticks(
            np.arange(len(pivot.index))
        )
        ax.set_yticklabels(
            pivot.index
        )

        ax.set_xlabel(
            parameter_2_label or parameter_2,
            labelpad=4,
        )
        ax.set_ylabel(
            parameter_1_label or parameter_1,
            labelpad=4,
        )

        ax.set_title(
            metric_labels.get(metric, metric),
            fontsize=11,
            pad=6,
        )

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]

                text = (
                    "NaN"
                    if np.isnan(value)
                    else format(value, fmt)
                )

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                )

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.03,
        )

    for ax in axes[nplots:]:
        ax.remove()

    fig.tight_layout()
    plt.show()

parameter_1= 'multiplier'
parameter_2= 'duration'

sa_config = build_sensitivity_config(
    base_config=env_config
)

results = sensitivity_analysis_2d(
    base_config=sa_config,
    policy=always_maintenance_policy,

    baseline = {
        "threshold": 0.85,
        "multiplier": 0.45,
        "duration": 15,
        "decision_interval_months": 6,
    },
    thresholds=[0.70, 0.75, 0.80, 0.85, 0.90],
    damage_multipliers=[0.40, 0.45, 0.50, 0.55, 0.60],
    durations=[5, 10, 15, 20, 25],
    decision_interval_months=[4, 6, 8, 10, 12],

    seeds = [
        42,
        123,
        456,
        789,
        2024,
        31415,
        27182,
        16180,
        57721,
        99991,
    ],

    parameter_1 = parameter_1,
    parameter_2 = parameter_2
)

# results.to_csv('/content/drive/MyDrive/1openfast_data/SA_multiplier_v_duration.csv')

results = pd.read_csv('/content/drive/MyDrive/1openfast_data/SA_result/SA_threshold_v_duration.csv')
parameter_1= 'threshold'
parameter_2= 'duration'
parameter_1_label = 'Repair Threshold'
parameter_2_label = 'Protection Duration'

results.columns

plot_sensitivity_heatmaps(
    results=results,
    parameter_1=parameter_1,
    parameter_2=parameter_2,
    metrics=[
        "mean_HI",
        "minimum_HI",
        "replacement_occurred",
    ],
    parameter_1_label=parameter_1_label,
    parameter_2_label=parameter_2_label,
    metric_labels={
        "mean_HI": "Mean HI",
        "minimum_HI": "Minimum HI",
        "repair_count": "Repair Count",
        "replacement_occurred": "Replacement Probability",
        "lifetime_before_replacement": "Lifetime Before Replacement",
        "maintenance_cost": "Maintenance Cost",
    },
    ncols=3,
)

rng = np.random.default_rng(seed=42)
n_samples_per_month = 1000
wind_data = []

for month in range(1, 13):
    for _ in range(n_samples_per_month):
        ambient_u, ambient_wd = wind_climate.sample(month, rng)
        wind_data.append({
            'month': month,
            'ambient_u': ambient_u,
            'ambient_wd': ambient_wd
        })

wind_climate_df = pd.DataFrame(wind_data)

import calendar
wind_climate_df['month_name'] = wind_climate_df['month'].apply(lambda x: calendar.month_abbr[x])

# Ensure the months are in chronological order
month_order = [calendar.month_abbr[i] for i in range(1, 13)]
wind_climate_df['month_name'] = pd.Categorical(wind_climate_df['month_name'], categories=month_order, ordered=True)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 7))
sns.boxplot(x='month_name', y='ambient_u', data=wind_climate_df, whis=(0, 100))
plt.xlabel('Month')
plt.ylabel('Ambient Wind Speed (m/s)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import calendar

# Ensure reference_df has 'month_name' column for better labeling
if 'month_name' not in reference_df.columns:
    reference_df['month_name'] = reference_df['month'].apply(lambda x: calendar.month_abbr[x])
    # Ensure the months are in chronological order
    month_order = [calendar.month_abbr[i] for i in range(1, 13)]
    reference_df['month_name'] = pd.Categorical(reference_df['month_name'], categories=month_order, ordered=True)

# Define bins for ambient_u
wind_speed_bins = np.arange(0, 30, 2) # Bins from 0 to 28 with step of 2
reference_df['ambient_u_binned'] = pd.cut(reference_df['ambient_u'], bins=wind_speed_bins, right=False)

# Create a pivot table for the heatmap
heatmap_data = reference_df.groupby(['month_name', 'ambient_u_binned']).size().unstack(fill_value=0)

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt='d', linewidths=.5)
plt.title('Number of Occurrences of Ambient Wind Speed per Month (from reference_df)')
plt.xlabel('Ambient Wind Speed (m/s) Bins')
plt.ylabel('Month')
plt.tight_layout()
plt.show()

!pip install windrose
import matplotlib.pyplot as plt
from windrose import WindroseAxes

fig = plt.figure(figsize=(10, 10))
ax = WindroseAxes.from_ax(fig=fig)
ax.bar(wind_climate_df['ambient_wd'], wind_climate_df['ambient_u'], normed=True, opening=0.8, edgecolor='white')

ax.set_legend(title='Wind Speed (m/s)')
plt.title('Wind Rose of Ambient Wind Direction and Speed')
plt.show()

_rng = np.random.default_rng(seed=42)
ambient_u, ambient_wd = wind_climate.sample(1, _rng)

simul = wake_solver.solve(
    ambient_u,
    ambient_wd
)

print('ambient u :', ambient_u)
print('ambient wd:', ambient_wd)

flow_map = wake_solver.flow_map(ambient_u, ambient_wd)

# Custom Visualization Turbine Numbering

import numpy as np
import matplotlib.pyplot as plt


def plot_flow_map(
    flow_map,
    ambient_wd,
    variable="WS_eff",
    levels=50,
    cmap="Blues_r",
    show_numbers=True,
    rotor_length=126,
    ax=None,
):
    """
    Plot a publication-quality wake map from a PyWake FlowMap.

    Parameters
    ----------
    flow_map : FlowMap
        PyWake FlowMap object.

    ambient_wd : float
        Ambient wind direction (deg). All turbines are oriented
        according to this direction.

    variable : str
        Variable to plot ("WS_eff", "TI_eff", etc.).

    levels : int
        Number of contour levels.

    cmap : str
        Matplotlib colormap.

    show_numbers : bool
        Show turbine numbering.

    rotor_length : float
        Length of turbine symbol (m).

    ax : matplotlib.axes.Axes, optional
    """

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # ------------------------------------------------------------
    # Weighted average (same as FlowMap.plot_wake_map())
    # ------------------------------------------------------------
    data = getattr(flow_map, variable)

    sum_dims = [d for d in ["wd", "time", "ws"] if d in flow_map.P.dims]

    data = (
        data * flow_map.P /
        flow_map.P.sum(sum_dims)
    ).sum(sum_dims)

    Z = data.squeeze().values

    # ------------------------------------------------------------
    # Wake contour
    # ------------------------------------------------------------
    cf = ax.contourf(
        flow_map.X,
        flow_map.Y,
        Z,
        levels=levels,
        cmap=cmap,
        extend="neither",
    )

    cbar = fig.colorbar(
        cf,
        ax=ax,
        shrink=0.87
        )

    if variable == "WS_eff":
        cbar.set_label("Wind speed (m/s)")
    elif variable == "TI_eff":
        cbar.set_label("Effective turbulence intensity")
    else:
        cbar.set_label(variable)

    # ------------------------------------------------------------
    # Turbine coordinates
    # ------------------------------------------------------------
    sim = flow_map.simulationResult

    x = sim.x.mean(set(sim.x.dims) - {"wt"}).values
    y = sim.y.mean(set(sim.y.dims) - {"wt"}).values

    # ------------------------------------------------------------
    # Turbine orientation (same for all turbines)
    # ------------------------------------------------------------
    theta = np.deg2rad(-ambient_wd)

    half = rotor_length / 2

    dx = half * np.cos(theta)
    dy = half * np.sin(theta)

    # ------------------------------------------------------------
    # Draw turbines
    # ------------------------------------------------------------
    for xi, yi in zip(x, y):

        ax.plot(
            [xi - dx, xi + dx],
            [yi - dy, yi + dy],
            color="black",
            linewidth=2,
            solid_capstyle="round",
            zorder=10,
        )

    # ------------------------------------------------------------
    # Turbine numbering
    # ------------------------------------------------------------
    if show_numbers:

        offset = rotor_length * 0.10

        nx = -np.sin(theta)
        ny =  np.cos(theta)

        for i, (xi, yi) in enumerate(zip(x, y), start=1):

            ax.text(
                xi + 250*nx,
                yi - 100*ny,
                str(i),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="regular",
                color="black",
                zorder=11,
            )

    # ------------------------------------------------------------
    # Axis
    # ------------------------------------------------------------
    ax.set_aspect("equal")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    return fig, ax

fig, ax = plot_flow_map(
    flow_map,
    ambient_wd=ambient_wd,
    variable="WS_eff",
    levels=60,
    cmap="Blues_r",
)

plt.rcParams["font.size"] = 13
fig.set_size_inches(6, 5)
plt.tight_layout()
plt.show()

