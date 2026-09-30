import numpy as np
import pandas as pd
import xarray as xr
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator

from py_wake.site import XRSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_farm_models import PropagateDownwind

from .config import WakeSolverConfig, DamageSolverConfig

class WakeSolver:
    """
    Solves wind farm wake effects using PyWake.

    This class handles the initialization of the site and turbine models,
    and calculates the effective wind speeds, turbulence intensities, and power output
    for a given ambient wind condition.
    """

    def __init__(self, config: WakeSolverConfig):
        """
        Initializes the WakeSolver with the given configuration.

        Args:
            config (WakeSolverConfig): Configuration for layout, turbine, and wake models.
        """
        self.config = config
        self.layout = config.layout
        self.layout_x = config.layout.x
        self.layout_y = config.layout.y
        self.n_turbines = len(self.layout_x)

        self.turbine_config = config.turbine

        import os
        if os.path.exists(self.turbine_config.csv_path):
            self.turbine_df = pd.read_csv(self.turbine_config.csv_path)
        else:
            print(f"Warning: Turbine CSV {self.turbine_config.csv_path} not found. Using fallback mock data.")
            self.turbine_df = pd.DataFrame({
                "Wind speed [m/s]": np.linspace(3, 25, 23),
                "Power [kW]": np.linspace(0, 5000, 23),
                "Thrust coefficient [-]": np.linspace(0.1, 0.8, 23),
                "Cp": np.linspace(0.2, 0.4, 23),
                "Ct": np.linspace(0.1, 0.8, 23)
            })

        self.wake_deficit_model = config.wake_deficit_model
        self.superposition_model = config.superposition_model
        self.turbulence_model = config.turbulence_model

        self.wind_turbines = None
        self.wind_farm_model = None
        self.site = None

        self._build_turbine()
        self._build_site()
        self._build_model()

    @staticmethod
    def ambient_ti(ws, z=90.0):
        """
        Calculates the ambient turbulence intensity based on the Extended ISO model.

        Args:
            ws (np.ndarray): Array of ambient wind speeds in m/s.
            z (float): Evaluation height (typically hub height) in meters.

        Returns:
            np.ndarray: Calculated turbulence intensity values.
        """
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
        z_hub = self.turbine_config.hub_height

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
        """
        Calculates effective wind conditions and power for a given ambient condition.

        Args:
            ambient_u (float or np.ndarray): Ambient wind speed in m/s.
            ambient_wd (float or np.ndarray): Ambient wind direction in degrees.

        Returns:
            dict: Dictionary containing arrays of effective wind speeds (`u_eff`),
                  effective turbulence intensities (`ti_eff`), and power in MW (`power`).
        """
        ambient_u = np.atleast_1d(ambient_u)
        ambient_wd = np.atleast_1d(ambient_wd)
        is_vectorized = len(ambient_u) > 1

        if is_vectorized:
            unique_wds = np.unique(ambient_wd)
            
            u_eff_out = np.zeros((self.n_turbines, len(ambient_u)), dtype=np.float32)
            ti_eff_out = np.zeros((self.n_turbines, len(ambient_u)), dtype=np.float32)
            power_out = np.zeros((self.n_turbines, len(ambient_u)), dtype=np.float32)
            
            for wd_val in unique_wds:
                idx = np.where(ambient_wd == wd_val)[0]
                ws_subset = ambient_u[idx]
                
                sim = self.wind_farm_model(
                    x=self.layout_x,
                    y=self.layout_y,
                    ws=ws_subset,
                    wd=np.array([wd_val])
                )
                
                u_eff_out[:, idx] = sim.WS_eff.values[:, 0, :].astype(np.float32)
                ti_eff_out[:, idx] = sim.TI_eff.values[:, 0, :].astype(np.float32)
                power_out[:, idx] = sim.Power.values[:, 0, :].astype(np.float32)

            u_eff = u_eff_out
            ti_eff = ti_eff_out
            power_kw = power_out
        else:
            simulation = self.wind_farm_model(
                x=self.layout_x,
                y=self.layout_y,
                ws=ambient_u,
                wd=ambient_wd
            )
            u_eff = simulation.WS_eff.isel(wt=range(self.n_turbines), wd=0, ws=0).values.astype("float32")
            ti_eff = simulation.TI_eff.isel(wt=range(self.n_turbines), wd=0, ws=0).values.astype("float32")
            power_kw = simulation.Power.isel(wt=range(self.n_turbines), wd=0, ws=0).values.astype("float32")

        power = power_kw / 1e6  # MW

        return {
            "u_eff": u_eff,
            "ti_eff": ti_eff,
            "power": power,
        }

    def solve_complete(self, ambient_u, ambient_wd):
        """
        Calculates a comprehensive set of effective conditions and power data.

        Args:
            ambient_u (float): Ambient wind speed in m/s.
            ambient_wd (float): Ambient wind direction in degrees.

        Returns:
            dict: Detailed simulation results including ratios and added turbulence.
        """
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


class DamageSolver:
    """
    Computes fatigue damage accumulation using structural response surfaces.
    """

    def __init__(self, config: DamageSolverConfig):
        """
        Initializes the DamageSolver.

        Args:
            config (DamageSolverConfig): Configuration containing the CSV path for the response surface and fatigue parameters.
        """
        self.config = config

        import os
        if os.path.exists(self.config.csv_path):
            self.response_surface_df = pd.read_csv(self.config.csv_path)
        else:
            print(f"Warning: Damage CSV {self.config.csv_path} not found. Using fallback mock data.")
            u_dummy = np.linspace(3, 25, 5)
            ti_dummy = np.linspace(0.05, 0.25, 5)
            U, TI = np.meshgrid(u_dummy, ti_dummy)
            np.random.seed(42) # Deterministic fallback
            self.response_surface_df = pd.DataFrame({
                "u": U.flatten(),
                "ti": TI.flatten(),
                "del_flap": np.random.uniform(1000, 5000, 25),
                "del_edge": np.random.uniform(2000, 8000, 25)
            })

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
        """
        Predicts Damage Equivalent Loads (DEL) for flap and edge based on the response surface.

        Args:
            u_eff (np.ndarray): Array of effective wind speeds.
            ti_eff (np.ndarray): Array of effective turbulence intensities.

        Returns:
            dict: Dictionary containing arrays of predicted DEL for flap (`del_flap`) and edge (`del_edge`).
        """
        orig_shape = np.shape(u_eff)
        u_flat = np.ravel(u_eff)
        ti_flat = np.ravel(ti_eff)
        points = (u_flat, ti_flat)

        del_flap = self.interpolators["del_flap"](points).astype(np.float32).reshape(orig_shape)
        del_edge = self.interpolators["del_edge"](points).astype(np.float32).reshape(orig_shape)

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
        """
        Calculates the accumulated fatigue damage over a given duration.

        Args:
            u_eff (np.ndarray): Array of effective wind speeds.
            ti_eff (np.ndarray): Array of effective turbulence intensities.
            duration_minutes (float): Time duration for the damage accumulation.

        Returns:
            np.ndarray: Array of accumulated fatigue damage values.
        """
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
class EnergyResult:
    """
    Energy production result.
    """
    energy: float

class EnergySolver:
    """
    Convert turbine power into generated energy.
    """
    def solve(self, power: float, operating_hours: float) -> EnergyResult:
        """
        Calculates energy produced given power and operating hours.

        Args:
            power (float): Farm or turbine power output.
            operating_hours (float): Number of operating hours.

        Returns:
            EnergyResult: The resulting energy production.
        """
        energy = (
            power
            * operating_hours
        )

        return EnergyResult(
            energy=energy,
        )
