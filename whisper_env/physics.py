import numpy as np
import pandas as pd
import xarray as xr
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator

from .config import DamageSolverConfig

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

        damage_flap_10min = ((del_pred["del_flap"].astype(np.float64)/self.config.del_flap_ref)**m)/n_ref
        damage_edge_10min = ((del_pred["del_edge"].astype(np.float64)/self.config.del_edge_ref)**m)/n_ref
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