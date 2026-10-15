import copy
from pathlib import Path

import pytest
import yaml

from qnnbench.core.config import (
    BASE_BENCHMARK,
    Config,
    freeze,
    load_config,
    load_frozen,
)
from qnnbench.core.errors import ConfigFrozenError


def test_t_m01_01_yaml_roundtrip_preserves_values_and_hash(tmp_path: Path):
    """T-M01-01: YAML round-trip preserves values and hash."""
    base_file = Path("configs/base.yaml")
    cfg = load_config(base_file)

    out_file = tmp_path / "saved_base.yaml"
    freeze(cfg, out_file)

    loaded = load_frozen(out_file)
    assert loaded.raw == cfg.raw
    assert loaded.hash == cfg.hash


def test_t_m01_03_tampered_frozen_file_raises(tmp_path: Path):
    """T-M01-03: Tampering a frozen file raises ConfigFrozenError."""
    base_file = Path("configs/base.yaml")
    cfg = load_config(base_file)

    frozen_file = tmp_path / "frozen.yaml"
    freeze(cfg, frozen_file)

    # Tamper with file
    with open(frozen_file) as f:
        data = yaml.safe_load(f)
    data["task"]["n_samples"] = 999
    with open(frozen_file, "w") as f:
        yaml.dump(data, f)

    with pytest.raises(ConfigFrozenError):
        load_frozen(frozen_file)


def test_t_m01_05_config_validation():
    """T-M01-05: Validation rejects invalid configurations."""
    # Invalid split fractions sum
    bad_cfg = copy.deepcopy(BASE_BENCHMARK)
    bad_cfg["splits"]["main"]["train"] = 0.5
    with pytest.raises(ValueError, match="Split fractions must sum to 1.0"):
        Config(raw=bad_cfg).validate()

    # Non-stratified
    bad_cfg2 = copy.deepcopy(BASE_BENCHMARK)
    bad_cfg2["splits"]["main"]["stratified"] = False
    with pytest.raises(ValueError, match="stratified=True"):
        Config(raw=bad_cfg2).validate()

    # n_total_params mismatch
    bad_cfg3 = copy.deepcopy(BASE_BENCHMARK)
    bad_cfg3["model"]["n_total_params"] = 99
    with pytest.raises(ValueError, match="n_total_params"):
        Config(raw=bad_cfg3).validate()
