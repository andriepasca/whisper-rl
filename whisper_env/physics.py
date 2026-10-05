import numpy as np
import pandas as pd
import xarray as xr
from dataclasses import dataclass

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
        elif not getattr(self.config, "allow_mock", False):
            raise FileNotFoundError(
                f"Damage response-surface CSV not found: '{self.config.csv_path}'. "
                "Check the path (hint: a doubled '.csv.csv' extension is a common mistake). "
                "Mock data is only available with DamageSolverConfig(allow_mock=True)."
            )
        else:
            print(f"WARNING: Damage CSV {self.config.csv_path} not found. allow_mock=True: using RANDOM MOCK data (not scientifically valid).")
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

        self.u_unique = np.sort(self.response_surface_df["u"].unique())
        self.ti_unique = np.sort(self.response_surface_df["ti"].unique())
        
        self.flap_grid = (
            self.response_surface_df
            .pivot(
                index=self.config.u_column,
                columns=self.config.ti_column,
                values=self.config.del_flap_column
            )
            .values
        )
        self.edge_grid = (
            self.response_surface_df
            .pivot(
                index=self.config.u_column,
                columns=self.config.ti_column,
                values=self.config.del_edge_column
            )
            .values
        )

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
        u_eff = np.asarray(u_eff)
        ti_eff = np.asarray(ti_eff)
        
        orig_shape = u_eff.shape
        
        u_eff_clip = np.clip(u_eff, self.u_unique[0], self.u_unique[-1])
        ti_eff_clip = np.clip(ti_eff, self.ti_unique[0], self.ti_unique[-1])
        
        idx_u = np.clip(np.searchsorted(self.u_unique, u_eff_clip), 0, len(self.u_unique) - 1)
        idx_ti = np.clip(np.searchsorted(self.ti_unique, ti_eff_clip), 0, len(self.ti_unique) - 1)
        
        del_flap = self.flap_grid[idx_u, idx_ti].astype(np.float32)
        del_edge = self.edge_grid[idx_u, idx_ti].astype(np.float32)

        return {
            "del_flap": del_flap.reshape(orig_shape),
            "del_edge": del_edge.reshape(orig_shape),
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