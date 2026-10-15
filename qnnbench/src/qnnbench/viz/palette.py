"""Colour-blind-safe Okabe-Ito palette and fixed styling mappings.

Section 5.1 of FLIGHT_MANUAL:
Fixed mapping defined once and imported everywhere.
Also fixed marker + linestyle per optimiser so figures survive greyscale printing.
"""

from __future__ import annotations

# Okabe-Ito colour mappings
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#999999",
    "dark_grey": "#444444",
}

# Fixed optimizer aesthetics
OPTIMISER_COLORS = {
    "gd": OKABE_ITO["blue"],
    "momentum": OKABE_ITO["sky_blue"],
    "adam": OKABE_ITO["orange"],
    "spsa": OKABE_ITO["vermilion"],
    "qng": OKABE_ITO["bluish_green"],
    "logreg": OKABE_ITO["grey"],
    "mlp": OKABE_ITO["dark_grey"],
    "adaptive": OKABE_ITO["reddish_purple"],
    "uniform": OKABE_ITO["black"],
}

OPTIMISER_MARKERS = {
    "gd": "o",
    "momentum": "s",
    "adam": "^",
    "spsa": "D",
    "qng": "v",
    "logreg": "x",
    "mlp": "+",
    "adaptive": "*",
    "uniform": "p",
}

OPTIMISER_LINESTYLES = {
    "gd": "-",
    "momentum": "--",
    "adam": "-.",
    "spsa": ":",
    "qng": "-",
    "logreg": "--",
    "mlp": ":",
    "adaptive": "-",
    "uniform": "--",
}


def get_style(optimiser: str) -> dict[str, str]:
    """Retrieve color, marker, and linestyle for an optimizer."""
    opt = optimiser.lower()
    return {
        "color": OPTIMISER_COLORS.get(opt, OKABE_ITO["black"]),
        "marker": OPTIMISER_MARKERS.get(opt, "o"),
        "linestyle": OPTIMISER_LINESTYLES.get(opt, "-"),
    }
