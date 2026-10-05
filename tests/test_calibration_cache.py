import pytest
import numpy as np
import whisper_env
from whisper_env import (
    get_calibrated_del_refs,
    make_fixed_damage_solver_config,
    get_default_config,
    OffshoreMaintenanceEnv,
    DamageSolverConfig,
)
from whisper_env.config import _CACHED_DEL_REFS


def test_get_calibrated_del_refs_caching():
    # Force reset for testing clean initial state
    whisper_env.config._CACHED_DEL_REFS = None

    assert whisper_env.config._CACHED_DEL_REFS is None

    # First call - performs calibration
    refs1 = get_calibrated_del_refs()
    assert refs1 is not None
    assert len(refs1) == 2
    assert isinstance(refs1[0], float)
    assert isinstance(refs1[1], float)

    # Check cache is set
    assert whisper_env.config._CACHED_DEL_REFS == refs1

    # Second call - returns cached value
    refs2 = get_calibrated_del_refs()
    assert refs2 is refs1

    # Call with force_recalibrate=True
    refs3 = get_calibrated_del_refs(force_recalibrate=True)
    assert refs3 is not None
    assert abs(refs3[0] - refs1[0]) < 1e-3
    assert abs(refs3[1] - refs1[1]) < 1e-3


def test_make_fixed_damage_solver_config_uses_cache():
    ds_config = make_fixed_damage_solver_config()
    refs = get_calibrated_del_refs()

    assert ds_config.del_flap_ref == refs[0]
    assert ds_config.del_edge_ref == refs[1]


def test_env_uses_calibration_cache():
    config = get_default_config(seed=42)
    env = OffshoreMaintenanceEnv(config)
    refs = get_calibrated_del_refs()

    assert env.damage_solver.config.del_flap_ref == refs[0]
    assert env.damage_solver.config.del_edge_ref == refs[1]
