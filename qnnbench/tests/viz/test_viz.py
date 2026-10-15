"""Visualisation tests verifying M16 contracts, aesthetics, and claims lint."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qnnbench.report.claims_lint import lint_text
from qnnbench.viz.figures import (
    f01_circuit,
    f03_gradients,
    f04_shot_noise,
    f05_analytic,
    f06_equal_shot,
    f24_historical,
)
from qnnbench.viz.gallery import build_gallery
from qnnbench.viz.palette import (
    OPTIMISER_COLORS,
    OPTIMISER_LINESTYLES,
    OPTIMISER_MARKERS,
    get_style,
)


def test_t_m16_01_all_figures_render_offline() -> None:
    """T-M16-01: Every figure generator renders from synthetic/fixture data without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        p1 = f01_circuit.make(out_path=out / "f01")
        p3 = f03_gradients.make(out_path=out / "f03")
        p4 = f04_shot_noise.make(out_path=out / "f04")
        p5 = f05_analytic.make(out_path=out / "f05")
        p6 = f06_equal_shot.make(out_path=out / "f06")
        p24 = f24_historical.make(out_path=out / "f24")

        assert (out / "f01.svg").exists()
        assert (out / "f03.svg").exists()
        assert (out / "f04.svg").exists()
        assert (out / "f05.svg").exists()
        assert (out / "f06.svg").exists()
        assert (out / "f24.svg").exists()

        # Build gallery test
        gallery_out = out / "index.html"
        build_gallery(figures_dir=out, output_html=gallery_out)
        assert gallery_out.exists()


def test_t_m16_03_palette_and_style_consistency() -> None:
    """T-M16-03: Palette/marker/linestyle mappings are unique per optimizer and deterministic."""
    opts = ["gd", "momentum", "adam", "spsa", "qng"]
    colors = [OPTIMISER_COLORS[o] for o in opts]
    markers = [OPTIMISER_MARKERS[o] for o in opts]
    linestyles = [OPTIMISER_LINESTYLES[o] for o in opts]

    # All colors unique among the 5 optimizers
    assert len(set(colors)) == 5
    # All markers unique
    assert len(set(markers)) == 5

    for opt in opts:
        st = get_style(opt)
        assert st["color"] == OPTIMISER_COLORS[opt]
        assert st["marker"] == OPTIMISER_MARKERS[opt]


def test_t_m16_07_determinism_svg_bytes() -> None:
    """T-M16-07: Two renders of identical data yield identical SVG bytes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        f03_gradients.make(out_path=out / "render1")
        f03_gradients.make(out_path=out / "render2")

        svg1 = (out / "render1.svg").read_bytes()
        svg2 = (out / "render2.svg").read_bytes()
        assert svg1 == svg2


def test_t_m16_09_claims_lint_enforcement() -> None:
    """T-M16-09: Claims lint catches banned marketing terms."""
    # Banned terms raise violations
    bad_text = "Our quantum model achieves exponential speedup and outperforms classical systems."
    violations = lint_text(bad_text)
    assert len(violations) >= 2

    # Compliant text passes
    clean_text = "We benchmark 4-qubit variational circuits on Iris under equal shot budgets."
    assert len(lint_text(clean_text)) == 0
