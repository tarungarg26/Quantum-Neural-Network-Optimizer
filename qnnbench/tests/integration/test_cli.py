
from qnnbench.io.logger import get_env_manifest
from qnnbench.report.claims_lint import check_claims


def test_t_m17_01_doctor_manifest():
    """T-M17-01: Doctor manifest outputs live environment metadata."""
    manifest = get_env_manifest()
    assert "python" in manifest
    assert "pennylane" in manifest
    assert "numpy" in manifest


def test_t_m16_09_claims_lint():
    """T-M16-09: No wording of speedup, quantum advantage, or energy savings in source tree."""
    violations = check_claims("src/qnnbench")
    assert not violations, "Claims lint failed:\n" + "\n".join(violations)
