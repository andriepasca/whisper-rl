from dataclasses import dataclass
from typing import Optional
from .config import MaintenanceType, MaintenanceConfig

@dataclass
class MaintenanceResult:
    """
    Result of MaintenancePolicy.
    """
    maintenance_type: Optional[MaintenanceType]

    downtime_hours: float
    carbon_emission: float

    # Applied to future fatigue damage.
    damage_multiplier: float

    # Remaining duration of damage protection.
    protection_duration: int


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
        hi: float,
        action: int,
        protection_remaining: int,
    ) -> MaintenanceResult:
        """
        Determines the outcome of a maintenance decision.

        Args:
            hi (float): The current health index of the turbine.
            action (int): The chosen action (0 for no intervention, 1 for intervention).
            protection_remaining (int): Remaining months of protection from previous maintenance.

        Returns:
            MaintenanceResult: The result of the applied or rejected maintenance.
        """

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

            downtime_hours=0.0,
            carbon_emission=0.0,

            damage_multiplier=1.0,
            protection_duration=0,
        )
