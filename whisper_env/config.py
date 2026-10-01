import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any

def _default_monthly_weibull():
    return {
        1:  {"k": 2.2, "c": 13.0}, 2:  {"k": 2.2, "c": 12.5},
        3:  {"k": 2.2, "c": 11.5}, 4:  {"k": 2.2, "c": 10.5},
        5:  {"k": 2.2, "c": 9.5},  6:  {"k": 2.2, "c": 8.5},
        7:  {"k": 2.2, "c": 8.0},  8:  {"k": 2.2, "c": 8.5},
        9:  {"k": 2.2, "c": 9.5},  10: {"k": 2.2, "c": 11.0},
        11: {"k": 2.2, "c": 12.0}, 12: {"k": 2.2, "c": 13.0},
    }


CLIMATE_PRESETS = {
    "north_sea": _default_monthly_weibull(),
    "us_atlantic": {m: {"k": 2.1, "c": 9.2} for m in range(1, 13)},
    "taiwan_strait": {m: {"k": 1.8, "c": 11.0} for m in range(1, 13)}
}

def _default_wind_direction():
    return np.arange(0.0, 360.0, 10.0)

@dataclass
class WindClimateConfig:
    """
    Configuration for the ambient wind climate.
    """
    monthly_weibull: dict = field(default_factory=dict)
    wind_direction: np.ndarray = field(default_factory=_default_wind_direction)
    wind_direction_probability: Optional[np.ndarray] = None
    climate_preset: str = "north_sea"

    def __post_init__(self):
        if not self.monthly_weibull:
            self.monthly_weibull = CLIMATE_PRESETS.get(self.climate_preset, CLIMATE_PRESETS["north_sea"])

def _get_default_layout():
    from .models import RandomScatteredLayout
    return RandomScatteredLayout(
        D=126.0, n_turbines=25, min_spacing_D=7.0, farm_scale=1.0, seed=42
    )

@dataclass
class LayoutConfig:
    """
    Configuration for the wind farm layout coordinates.
    """
    x: np.ndarray = field(default_factory=lambda: np.array(_get_default_layout().x))
    y: np.ndarray = field(default_factory=lambda: np.array(_get_default_layout().y))

import os

DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

@dataclass
class TurbineConfig:
    """
    Configuration for the wind turbine model parameters.
    """
    name: str = "NREL 5-MW"
    rotor_diameter: float = 126.0
    hub_height: float = 90.0
    csv_path: str = os.path.join(DEFAULT_DATA_DIR, "NREL_Reference_5MW_126.csv") # source from https://github.com/NatLabRockies/turbine-models
    power_unit: str = "kW"



@dataclass
class DamageSolverConfig:
    """
    Configuration for the DamageSolver response surfaces and fatigue calculations.
    """
    csv_path: str = os.path.join(DEFAULT_DATA_DIR, "response_surface_200.csv")
    u_column: str = "u"
    ti_column: str = "ti"
    del_flap_column: str = "del_flap"
    del_edge_column: str = "del_edge"
    m_coef: float = 10.0
    del_flap_ref: Optional[float] = None
    del_edge_ref: Optional[float] = None
    design_life_years: float = 20.0

@dataclass
class MaintenanceType:
    """
    Defines a specific type of maintenance intervention and its properties.
    """
    name: str
    threshold: float

    # Maintenance consequences
    downtime_hours: float
    carbon_emission: float
    damage_multiplier: float

    # Duration of damage multiplier effectiveness.
    duration_months: int

    # Replacement resets degradation to zero.
    is_replacement: bool = False

def _default_maintenance_types():
    repair = MaintenanceType(
        name="repair", threshold=0.85, downtime_hours=8,
        carbon_emission=200, damage_multiplier=0.45, duration_months=15, is_replacement=False
    )
    replacement = MaintenanceType(
        name="replacement", threshold=0.10, downtime_hours=168,
        carbon_emission=5_000, damage_multiplier=1.0, duration_months=1, is_replacement=True
    )
    return (repair, replacement)

@dataclass
class MaintenanceConfig:
    """
    Configuration grouping available maintenance types.
    """
    maintenance_types: tuple[MaintenanceType, ...] = field(default_factory=_default_maintenance_types)

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
    objectives: dict[str, RewardObjective] = field(default_factory=dict)

    # Exposed Multi-Objective parameters
    w_damage: float = 0.85
    w_spatial: float = 0.15
    damage_scale: float = 25*6
    spatial_scale: float = 1.0

    def __post_init__(self):
        if not self.objectives:
            objectives = {
                "damage_burden": RewardObjective(
                    weight=self.w_damage, direction="min", normalize=True, scale=self.damage_scale,
                ),
                "spatial_grouping": RewardObjective(
                    weight=self.w_spatial, direction="min", normalize=True, scale=self.spatial_scale,
                ),
            }
            object.__setattr__(self, 'objectives', objectives)

def _default_wind_climate():
    from .models import WindClimate
    return WindClimate(WindClimateConfig())



def _default_damage_solver():
    from .physics import DamageSolver
    return DamageSolver(DamageSolverConfig())

def _default_maintenance_policy():
    from .maintenance import MaintenancePolicy
    return MaintenancePolicy(MaintenanceConfig())

def _default_transition_model():
    from .models import TransitionModel
    return TransitionModel()

def _default_spatial_grouping_objective():
    from .models import SpatialGroupingObjective
    return SpatialGroupingObjective(LayoutConfig())

def _default_reward_model():
    from .models import RewardModel
    return RewardModel(RewardConfig())

def _default_energy_solver():
    from .physics import EnergySolver
    return EnergySolver()

def _default_logger():
    from .models import LoggingModel
    return LoggingModel()

@dataclass(frozen=True)
class EnvironmentConfig:
    """
    Master configuration for the OffshoreMaintenanceEnv simulation environment.
    """
    wind_climate: Any = field(default_factory=_default_wind_climate)

    damage_solver: Any = field(default_factory=_default_damage_solver)
    maintenance_policy: Any = field(default_factory=_default_maintenance_policy)
    transition_model: Any = field(default_factory=_default_transition_model)
    spatial_grouping_objective: Any = field(default_factory=_default_spatial_grouping_objective)
    reward_model: Any = field(default_factory=_default_reward_model)
    energy_solver: Any = field(default_factory=_default_energy_solver)
    logger: Any = field(default_factory=_default_logger)
    max_protection_duration: int = 15
    randomize_initial_hi: bool = True
    initial_hi: float | np.ndarray = 1.0
    wind_time_slices_per_month: int = 30
    hours_per_month: float = 730.5
    decision_interval_months: int = 6
    max_simulation_years: int = 30
    u_min: float = 0.0
    u_max: float = 30.0
    ti_min: float = 0.03
    ti_max: float = 0.30
    seed: int = 42
    climate_preset: str = "north_sea"
    include_weather_window: bool = False

    def get_initial_hi(self, n_turbines: int) -> np.ndarray:
        """
        Retrieves the initial Health Index (HI) array for the turbines.

        Args:
            n_turbines (int): Number of turbines in the farm.

        Returns:
            np.ndarray: An array of initial Health Indices.
        """
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

def get_default_config(
    n_turbines: int = 25,
    w_damage: float = 0.85,
    w_spatial: float = 0.15,
    data_dir: str = DEFAULT_DATA_DIR,
    seed: int = 42,
    climate_preset: str = "north_sea",
    include_weather_window: bool = False,
    **kwargs
) -> EnvironmentConfig:
    """
    Generates a master EnvironmentConfig with top-level customizable parameters.
    Dynamically scales the layout, updates paths, and exposes MORL parameters.
    """
    from .models import RandomScatteredLayout, WindClimate, TransitionModel, SpatialGroupingObjective, RewardModel, LoggingModel
    from .physics import DamageSolver, EnergySolver
    from .maintenance import MaintenancePolicy
    import os

    # 1. Dynamic Layout Scaling
    layout = RandomScatteredLayout(
        D=126.0, n_turbines=n_turbines, min_spacing_D=7.0, farm_scale=1.0, seed=seed
    )
    layout_config = LayoutConfig(x=np.array(layout.x), y=np.array(layout.y))

    # 2. Path Eradication
    turbine_csv = os.path.join(data_dir, "NREL_Reference_5MW_126.csv")
    damage_csv = os.path.join(data_dir, "response_surface_200.csv")

    turbine_config = TurbineConfig(csv_path=turbine_csv)

    damage_solver_config = DamageSolverConfig(csv_path=damage_csv)

    # 3. Dynamic Reward Objectives
    x = layout_config.x
    y = layout_config.y
    spatial_reference_scale = float(np.max(np.sqrt((x[:, None] - x[None, :])**2 + (y[:, None] - y[None, :])**2)))

    reward_config = RewardConfig(
        w_damage=w_damage,
        w_spatial=w_spatial,
        damage_scale=n_turbines * 6,
        spatial_scale=spatial_reference_scale
    )

    # 4. Solvers and Models
    wind_climate = WindClimate(WindClimateConfig(climate_preset=climate_preset))

    damage_solver = DamageSolver(damage_solver_config)
    maintenance_policy = MaintenancePolicy(MaintenanceConfig())
    transition_model = TransitionModel()
    spatial_grouping_objective = SpatialGroupingObjective(layout_config)
    reward_model = RewardModel(reward_config)
    energy_solver = EnergySolver()
    logger = LoggingModel()

    return EnvironmentConfig(
        wind_climate=wind_climate,

        damage_solver=damage_solver,
        maintenance_policy=maintenance_policy,
        transition_model=transition_model,
        spatial_grouping_objective=spatial_grouping_objective,
        reward_model=reward_model,
        energy_solver=energy_solver,
        logger=logger,
        seed=seed,
        climate_preset=climate_preset,
        include_weather_window=include_weather_window,
        **kwargs
    )
