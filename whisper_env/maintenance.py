from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from .config import MaintenanceType, MaintenanceConfig

@dataclass
class MaintenanceResult:
    """
    Result of MaintenancePolicy for a single turbine.
    """
    maintenance_type: Optional[MaintenanceType]
    downtime_hours: float
    damage_multiplier: float
    protection_duration: int


@dataclass
class VectorizedMaintenanceResult:
    """
    Vectorized result of MaintenancePolicy across multiple turbines.
    """
    maintenance_types: np.ndarray    # dtype=object, array of MaintenanceType or None
    downtime_hours: np.ndarray       # dtype=np.float64
    damage_multiplier: np.ndarray    # dtype=np.float64
    protection_duration: np.ndarray  # dtype=np.int32
    is_replacement: np.ndarray       # dtype=bool


class MaintenancePolicy:
    """
    Evaluates and applies the configured maintenance strategy based on turbine health.
    """

    def __init__(self, config: MaintenanceConfig):
        """
        Initializes the MaintenancePolicy.

        Args:
            config (MaintenanceConfig): The maintenance configuration.
        """
        self.config = config

    def solve(
        self,
        hi: Union[float, np.ndarray],
        action: Union[int, np.ndarray],
        protection_remaining: Union[int, np.ndarray],
    ) -> Union[MaintenanceResult, VectorizedMaintenanceResult]:
        """
        Determines the outcome of maintenance decision(s).

        Args:
            hi (Union[float, np.ndarray]): The current health index of turbine(s).
            action (Union[int, np.ndarray]): The chosen action(s) (0 for no intervention, 1 for intervention).
            protection_remaining (Union[int, np.ndarray]): Remaining months of protection.

        Returns:
            Union[MaintenanceResult, VectorizedMaintenanceResult]: The result of applied or rejected maintenance.
        """
        if np.ndim(action) == 0 and not isinstance(action, np.ndarray):
            action_int = int(action)
            hi_val = float(hi)
            prot_val = int(protection_remaining)

            if action_int not in (0, 1):
                raise ValueError(
                    f"Unknown action: {action_int}. "
                    "Expected 0 (No intervention) or 1 (Intervention)."
                )

            if not (0.0 <= hi_val <= 1.0):
                raise ValueError(
                    f"Health Index must be within [0, 1]. "
                    f"Received {hi_val}."
                )

            if action_int == 0 or prot_val > 0:
                return self._no_maintenance()

            maintenance = self._select_type(hi_val)

            if maintenance is None:
                return self._no_maintenance()

            return MaintenanceResult(
                maintenance_type=maintenance,
                downtime_hours=maintenance.downtime_hours,
                damage_multiplier=maintenance.damage_multiplier,
                protection_duration=maintenance.duration_months,
            )

        # Vectorized path for arrays across all turbines
        hi_arr = np.asarray(hi, dtype=np.float64)
        action_arr = np.asarray(action, dtype=np.int32)
        prot_arr = np.asarray(protection_remaining, dtype=np.int32)

        if np.any((action_arr != 0) & (action_arr != 1)):
            raise ValueError("Action values must be 0 or 1.")

        n_turbines = len(hi_arr)
        maint_types = np.full(n_turbines, None, dtype=object)
        downtime_hours = np.zeros(n_turbines, dtype=np.float64)
        damage_multiplier = np.ones(n_turbines, dtype=np.float64)
        protection_duration = np.zeros(n_turbines, dtype=np.int32)
        is_replacement = np.zeros(n_turbines, dtype=bool)

        eligible = (action_arr == 1) & (prot_arr == 0)

        if np.any(eligible):
            sorted_types = sorted(self.config.maintenance_types, key=lambda m: m.threshold)
            unassigned = eligible.copy()

            for m in sorted_types:
                match = unassigned & (hi_arr <= m.threshold)
                if np.any(match):
                    maint_types[match] = m
                    downtime_hours[match] = m.downtime_hours
                    damage_multiplier[match] = m.damage_multiplier
                    protection_duration[match] = m.duration_months
                    is_replacement[match] = m.is_replacement
                    unassigned[match] = False

        return VectorizedMaintenanceResult(
            maintenance_types=maint_types,
            downtime_hours=downtime_hours,
            damage_multiplier=damage_multiplier,
            protection_duration=protection_duration,
            is_replacement=is_replacement,
        )

    def _select_type(
        self,
        hi: float,
    ) -> Optional[MaintenanceType]:

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
            downtime_hours=0.0,
            damage_multiplier=1.0,
            protection_duration=0,
        )
