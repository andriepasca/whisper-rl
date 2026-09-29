import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class WindClimateConfig:
    monthly_weibull: dict
    wind_direction: np.ndarray
    wind_direction_probability: Optional[np.ndarray] = None

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
class EnvironmentConfig:
    wind_climate: Any
    wake_solver: Any
    damage_solver: Any
    maintenance_policy: Any
    transition_model: Any
    electricity_price_model: Any
    spatial_grouping_objective: Any
    reward_model: Any
    energy_solver: Any
    logger: Any
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
