import pytest
import numpy as np
import sys
import os

# Add scripts directory to path to allow importing evaluate_policy components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from evaluate_policy import ElectricityPriceModel, CarbonEmissionModel, DummyAgent

def test_electricity_price_model():
    price_model = ElectricityPriceModel()
    rng = np.random.default_rng(42)
    
    for month in range(1, 13):
        price = price_model.sample(month, rng)
        assert not np.isnan(price)
        assert price >= 0.0

def test_carbon_emission_model():
    carbon_model = CarbonEmissionModel(carbon_intensity_factor=0.05)
    
    # Test zero mobilization
    assert carbon_model.calculate_emissions(0.0) == 0.0
    
    # Test non-zero mobilization distance (e.g., 1000m)
    emissions = carbon_model.calculate_emissions(1000.0)
    assert emissions == 50.0
