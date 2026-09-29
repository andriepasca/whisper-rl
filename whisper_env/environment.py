import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .config import EnvironmentConfig

def cyclic_encode(x, period):
    angle = 2 * np.pi * x / period
    return np.array(
        [np.sin(angle), np.cos(angle)],
        dtype=np.float32
    )

class OffshoreMaintenanceEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, config: EnvironmentConfig):
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

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        randomize_maintenance_params = (
            options is not None
            and options.get("randomize_maintenance_params", False)
        )

        if randomize_maintenance_params:
            for maintenance in self.maintenance_policy.config.maintenance_types:
                if maintenance.name.lower() == "repair":
                    maintenance.threshold = self.rng.uniform(0.75, 0.90)
                    maintenance.damage_multiplier = self.rng.uniform(0.5, 0.9)
                    maintenance.duration_months = int(self.rng.integers(3, 13))
                    break

        self.repair_params = {
            "threshold": maintenance.threshold,
            "multiplier": maintenance.damage_multiplier,
            "duration": maintenance.duration_months,
        }

        self.current_year = 1
        self.current_month = 1
        self.elapsed_month = 0

        self.delta_damage = np.zeros(self.n_turbines, dtype=np.float32)
        self.damage_multiplier = np.ones(self.n_turbines, dtype=np.float32)
        self.protection_remaining = np.zeros(self.n_turbines, dtype=np.int32)
        self.repair_count = np.zeros(self.n_turbines, dtype=np.int32)
        self.replacement_count = np.zeros(self.n_turbines, dtype=np.int32)
        self.last_action = np.zeros(self.n_turbines, dtype=np.int32)

        self.HI = self.config.get_initial_hi(self.n_turbines)

        self.ambient_u, self.ambient_wd = self.wind_climate.sample(
            month=self.current_month,
            rng=self.rng
        )

        self.electricity_price = self.electricity_price_model.sample(
            month=self.current_month,
            rng=self.rng
        )

        self.logger.reset()

        wake = self.wake_solver.solve(
            ambient_u=self.ambient_u,
            ambient_wd=self.ambient_wd,
        )

        self.u_eff = wake["u_eff"]
        self.ti_eff = wake["ti_eff"]
        self.power = wake["power"]

        observation = self._get_obs()
        info = self._get_info()

        for maintenance in self.maintenance_policy.config.maintenance_types:
            if maintenance.name.lower() == "repair":
                info["repair_damage_multiplier"] = maintenance.damage_multiplier
                info["repair_duration_months"] = maintenance.duration_months
                break

        return observation, info

    def step(self, action):
        action = np.asarray(action, dtype=np.int32)

        if action.shape != (self.n_turbines,):
            raise ValueError(
                f"Expected action shape {(self.n_turbines,)}, got {action.shape}."
            )

        if np.any((action != 0) & (action != 1)):
            raise ValueError("Action values must be 0 or 1.")

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

        months_simulated = 0

        for simulation_month in range(self.decision_interval):
            months_simulated += 1

            # Using expected damage/power via stratified sampling (Option B) to avoid Jensen's inequality bias
            expected_monthly_damage = np.zeros(self.n_turbines)
            expected_monthly_power = np.zeros(self.n_turbines)

            n_slices = self.config.wind_time_slices_per_month

            for ambient_u, ambient_wd, p_joint in self.wind_climate.iter_stratified(
                month=self.current_month,
                n_u=n_slices
            ):
                # We save one arbitrarily to log the state
                self.ambient_u = ambient_u
                self.ambient_wd = ambient_wd

                wake = self.wake_solver.solve(
                    ambient_u=ambient_u,
                    ambient_wd=ambient_wd,
                )

                u_eff = wake["u_eff"]
                ti_eff = wake["ti_eff"]
                power = wake["power"]

                # Damage rate calculation
                # damage_10min is the damage occurred during a 10 minutes period
                damage_10min = self.damage_solver.solve(
                    u_eff=u_eff,
                    ti_eff=ti_eff,
                    duration_minutes=10.0,
                )

                # scale to the full month using expected value integration
                slice_damage = (damage_10min * duration_minutes / 10.0)

                expected_monthly_damage += p_joint * slice_damage
                expected_monthly_power += p_joint * power

            self.u_eff = u_eff
            self.ti_eff = ti_eff
            self.power = expected_monthly_power

            mean_farm_power += np.sum(self.power)

            self.delta_damage = expected_monthly_damage
            interval_damage += np.sum(self.delta_damage)

            monthly_cost = 0.0
            monthly_energy = 0.0

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

            self.elapsed_month += 1
            self.current_month += 1

            if self.current_month > 12:
                self.current_month = 1
                self.current_year += 1

            if self.elapsed_month >= self.config.max_simulation_years * 12:
                truncated = True
                break

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
