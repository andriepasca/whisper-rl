import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .config import EnvironmentConfig
from .utils import WeatherOracle

def cyclic_encode(x, period):
    """
    Encodes a periodic variable (like month) into its sine and cosine components.

    Args:
        x (float): The value to encode.
        period (float): The period of the variable.

    Returns:
        np.ndarray: Array containing [sin(angle), cos(angle)].
    """
    angle = 2 * np.pi * x / period
    return np.array(
        [np.sin(angle), np.cos(angle)],
        dtype=np.float32
    )

class OffshoreMaintenanceEnv(gym.Env):
    """
    Gymnasium environment for optimizing offshore wind farm maintenance strategies.

    This environment simulates turbine health degradation, energy production,
    and maintenance interventions over time. Actions correspond to intervention
    decisions (0: no intervention, 1: intervention) for each turbine.
    """
    metadata = {"render_modes": []}

    def __init__(self, config: EnvironmentConfig):
        """
        Initializes the OffshoreMaintenanceEnv.

        Args:
            config (EnvironmentConfig): Comprehensive configuration object for the environment.
        """
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

        if self.config.damage_solver.config.del_flap_ref is None or self.config.damage_solver.config.del_edge_ref is None:
            self._calibrate_del_refs()

        # Weather Oracle for window probability
        self.weather_oracle = None

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

        farm_spaces = {
            "month": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        }
        if self.config.include_weather_window:
            farm_spaces["weather_window_prob"] = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "farm": spaces.Dict(farm_spaces),

            "turbines": spaces.Dict({
                "HI": spaces.Box(low=0, high=1, shape=(self.n_turbines,), dtype=np.float32),
                "protection_remaining": spaces.Box(low=0, high=1, shape=(self.n_turbines,), dtype=np.float32)
            })
        })

        self.action_space = spaces.MultiDiscrete([2] * self.n_turbines)

    def _calibrate_del_refs(self):
        """
        Calibrate DEL reference values via Monte Carlo sampling to hit expected 20-year design life.
        """
        import dataclasses
        m = self.config.damage_solver.config.m_coef

        del_flap_samples = []
        del_edge_samples = []

        # Run a Monte Carlo loop: 1,000 samples per month for 12 months (12,000 total)
        for month in range(1, 13):
            for _ in range(1000):
                ambient_u, ambient_wd = self.wind_climate.sample(month, self.rng)
                wake = self.wake_solver.solve(ambient_u, ambient_wd)

                # Extract effective conditions for Turbine 0 (freestream index 0)
                u_eff_t0 = wake["u_eff"][0]
                ti_eff_t0 = wake["ti_eff"][0]

                # Predict DEL
                dels = self.damage_solver.predict_del(u_eff_t0, ti_eff_t0)
                del_flap_samples.append(dels["del_flap"])
                del_edge_samples.append(dels["del_edge"])

        # Calculate the m-th power expected value (cast to np.float64)
        del_flap_arr = np.array(del_flap_samples, dtype=np.float64)
        del_edge_arr = np.array(del_edge_samples, dtype=np.float64)

        mean_del_flap_m = np.mean(del_flap_arr ** m)
        mean_del_edge_m = np.mean(del_edge_arr ** m)

        calibrated_flap_ref = mean_del_flap_m ** (1.0 / m)
        calibrated_edge_ref = mean_del_edge_m ** (1.0 / m)

        new_damage_config = dataclasses.replace(
            self.config.damage_solver.config,
            del_flap_ref=float(calibrated_flap_ref),
            del_edge_ref=float(calibrated_edge_ref)
        )

        # Instantiate a new DamageSolver with the calibrated config
        from .physics import DamageSolver
        new_damage_solver = DamageSolver(new_damage_config)

        # Update the environment config using dataclasses.replace because it's frozen
        self.config = dataclasses.replace(
            self.config,
            damage_solver=new_damage_solver
        )

        # Update the reference to the new damage solver in the environment
        self.damage_solver = new_damage_solver

    @property
    def max_steps(self):
        """
        Returns the maximum number of decision steps in an episode.
        """
        return int(
            np.ceil(
                self.config.max_simulation_years * 12
                / self.decision_interval
            )
        )

    def _get_obs(self):
        """
        Constructs the observation dictionary for the current state.
        """
        month = cyclic_encode(self.current_month - 1, period=12)
        protection_remaining_norm = (self.protection_remaining / self.config.max_protection_duration)
        farm_obs = {
            "month": month,
        }
        if self.config.include_weather_window:
            current_wind_speed = self.ambient_u if self.ambient_u is not None else 0.0
            # If _get_obs is called before reset, weather_oracle might be None. Initialize it here just in case.
            if self.weather_oracle is None:
                # Default to month 1 if current_month is not yet set
                m = self.current_month if self.current_month is not None else 1
                weibull_params = self.wind_climate.config.monthly_weibull[m]
                self.weather_oracle = WeatherOracle(weibull_shape=weibull_params["k"], weibull_scale=weibull_params["c"])

            prob = self.weather_oracle.get_maintenance_window_probability(current_wind_speed=current_wind_speed, lookahead_days=3)
            farm_obs["weather_window_prob"] = np.array([prob], dtype=np.float32)

        return {
            "farm": farm_obs,
            "turbines": {
                "HI": self.HI.astype(np.float32),
                "protection_remaining": protection_remaining_norm.astype(np.float32)
            }
        }

    def _get_info(self):
        """
        Constructs the info dictionary with supplementary environment state.
        """
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
        """
        Resets the environment to the initial state.

        Args:
            seed (Optional[int]): Random seed for the environment.
            options (Optional[dict]): Dictionary of options (e.g., randomize_maintenance_params).

        Returns:
            tuple[dict, dict]: The initial observation and info dictionaries.
        """
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        randomize_maintenance_params = (
            options is not None
            and options.get("randomize_maintenance_params", False)
        )

        if randomize_maintenance_params:
            for m in self.maintenance_policy.config.maintenance_types:
                if m.name.lower() == "repair":
                    m.threshold = self.rng.uniform(0.75, 0.90)
                    m.damage_multiplier = self.rng.uniform(0.5, 0.9)
                    m.duration_months = int(self.rng.integers(3, 13))
                    break

        maintenance = next((m for m in self.maintenance_policy.config.maintenance_types if m.name.lower() == "repair"), None)

        if maintenance is not None:
            self.repair_params = {
                "threshold": maintenance.threshold,
                "multiplier": maintenance.damage_multiplier,
                "duration": maintenance.duration_months,
            }
        else:
            self.repair_params = {
                "threshold": 0.0,
                "multiplier": 1.0,
                "duration": 0,
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

        if self.config.include_weather_window:
            weibull_params = self.wind_climate.config.monthly_weibull[self.current_month]
            self.weather_oracle = WeatherOracle(weibull_shape=weibull_params["k"], weibull_scale=weibull_params["c"])

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
        """
        Executes a simulation step based on the provided action.

        Args:
            action (np.ndarray): Array of actions (0 or 1) for each turbine.

        Returns:
            tuple: Contains observation, reward, terminated flag, truncated flag, and info.
        """
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

            # Using expected damage/power via Monte Carlo Stochastic Sampling to introduce aleatoric uncertainty
            n_slices = self.config.wind_time_slices_per_month

            ambient_u_list = []
            ambient_wd_list = []

            for _ in range(n_slices):
                ambient_u, ambient_wd = self.wind_climate.sample(
                    month=self.current_month,
                    rng=self.rng
                )
                ambient_u_list.append(ambient_u)
                ambient_wd_list.append(ambient_wd)
            
            ambient_u_arr = np.array(ambient_u_list)
            ambient_wd_arr = np.array(ambient_wd_list)

            # We save one arbitrarily to log the state
            self.ambient_u = float(ambient_u_arr[-1])
            self.ambient_wd = float(ambient_wd_arr[-1])

            wake = self.wake_solver.solve(
                ambient_u=ambient_u_arr,
                ambient_wd=ambient_wd_arr,
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

            # scale to the full month using Monte Carlo expected value integration
            slice_damage = (damage_10min * duration_minutes / 10.0)

            expected_monthly_damage = np.mean(slice_damage, axis=1)
            expected_monthly_power = np.mean(power, axis=1)

            self.u_eff = u_eff[:, -1]
            self.ti_eff = ti_eff[:, -1]
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
                    * hours_per_month
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
