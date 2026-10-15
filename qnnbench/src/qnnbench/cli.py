"""Command-line interface for QNNBench (M17).

Commands:
  qnnbench doctor            Check runtime environment and output manifest
  qnnbench run <E#> [--tiny] Run benchmark experiment (E1..E8)
  qnnbench analyse           Aggregate runs and compute comparisons
  qnnbench figures           Render publication figures
  qnnbench report            Compile HTML benchmark report with claims lint
  qnnbench lint              Run claims lint across repository
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qnnbench.analysis.tables import (
    make_t01_spec,
    make_t02_splits,
    make_t03_gradient_verification,
    make_t11_failures,
    make_t12_env,
    table_to_formats,
)
from qnnbench.analysis.tidy import load_runs_dir
from qnnbench.experiments.runner import run_experiment
from qnnbench.io.logger import get_env_manifest
from qnnbench.report.build import build_report
from qnnbench.report.claims_lint import check_claims
from qnnbench.viz.figures import (
    f01_circuit,
    f03_gradients,
    f04_shot_noise,
    f05_analytic,
    f06_equal_shot,
    f24_historical,
)
from qnnbench.viz.gallery import build_gallery


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="qnnbench",
        description="Quantum Neural Network Optimisation Benchmark (QNNBench) CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # doctor
    subparsers.add_parser("doctor", help="Check runtime environment and generate manifest")

    # lint
    subparsers.add_parser("lint", help="Verify claims policy compliance across repository")

    # run
    run_parser = subparsers.add_parser("run", help="Execute an experiment (E1..E8)")
    run_parser.add_argument("experiment", help="Experiment identifier (e.g. E1, E2, ... E8)")
    run_parser.add_argument("--tiny", action="store_true", help="Run in fast smoke mode")

    # analyse
    analyse_parser = subparsers.add_parser("analyse", help="Aggregate runs and build tables")
    analyse_parser.add_argument("--runs-dir", default="runs", help="Directory containing run artifacts")
    analyse_parser.add_argument("--out-dir", default="outputs/tables", help="Output directory for tables")

    # figures
    figures_parser = subparsers.add_parser("figures", help="Render publication figures")
    figures_parser.add_argument("--out-dir", default="outputs/figures", help="Output directory for figures")

    # report
    report_parser = subparsers.add_parser("report", help="Compile complete benchmark report")
    report_parser.add_argument("--runs-dir", default="runs", help="Directory containing run artifacts")
    report_parser.add_argument("--out-dir", default="outputs/report", help="Output directory for report")

    args = parser.parse_args()

    if args.command == "doctor":
        manifest = get_env_manifest()
        print(json.dumps(manifest, indent=2))
        return 0

    elif args.command == "lint":
        violations = check_claims(".")
        if violations:
            print(f"Found {len(violations)} claims violations:")
            for v in violations:
                print("  " + v)
            return 1
        print("Claims lint: clean. No banned claims detected.")
        return 0

    elif args.command == "run":
        exp_id = args.experiment.upper()
        print(f"Executing experiment {exp_id} (tiny={args.tiny})...")
        res = run_experiment(exp_id, tiny=args.tiny)
        print(f"Experiment {exp_id} completed successfully.")
        return 0

    elif args.command == "analyse":
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        bundle = load_runs_dir(args.runs_dir)

        t01 = make_t01_spec()
        table_to_formats(t01, "T01: Benchmark Specification", out / "t01_spec")

        t02 = make_t02_splits()
        table_to_formats(t02, "T02: Dataset Splits", out / "t02_splits")

        t11 = make_t11_failures(bundle)
        table_to_formats(t11, "T11: Failure and Fallback Log", out / "t11_failures")

        t12 = make_t12_env()
        table_to_formats(t12, "T12: Environment Manifest", out / "t12_env")

        print(f"Generated tables saved to {out}")
        return 0

    elif args.command == "figures":
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)

        f01_circuit.make(out_path=out / "f01_circuit")
        f03_gradients.make(out_path=out / "f03_gradients")
        f04_shot_noise.make(out_path=out / "f04_shot_noise")
        f05_analytic.make(out_path=out / "f05_analytic")
        f06_equal_shot.make(out_path=out / "f06_equal_shot")
        f24_historical.make(out_path=out / "f24_historical")

        build_gallery(out, out / "index.html")
        print(f"Rendered publication figures and gallery at {out}")
        return 0

    elif args.command == "report":
        result = build_report(args.runs_dir, args.out_dir)
        print(f"Report compiled successfully at {result['output_dir']}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
