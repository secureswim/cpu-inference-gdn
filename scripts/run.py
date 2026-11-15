"""Run correctness, benchmarks, plots, and a report in an isolated output folder."""
import argparse
from pathlib import Path
import subprocess

from system_config import ROOT, use_project_python


def main():
    config = use_project_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/reproduction")
    parser.add_argument("--allow-battery", action="store_true", help="Label measurements as provisional")
    parser.add_argument("--long-repeats", action="store_true", help="Also collect the two-second timing study")
    args = parser.parse_args()
    if args.long_repeats and args.allow_battery:
        parser.error("--long-repeats requires AC power; omit --allow-battery")
    folder = args.output.resolve()
    if folder == ROOT or folder == ROOT / "results" or folder == ROOT / "output":
        parser.error("Choose a separate run folder, such as build/reproduction")
    results, out = folder / "results", folder / "output"
    # Mixing earlier and current sweeps can silently contaminate analysis.
    if results.exists() and any(results.iterdir()):
        parser.error("This run folder already contains results; choose a new --output folder")
    results.mkdir(parents=True, exist_ok=True)

    def run(script, *options):
        subprocess.run([config["python"], str(ROOT / "scripts" / script), *map(str, options)], cwd=ROOT, check=True)

    flags = ["--allow-battery"] if args.allow_battery else []
    run("run_benchmarks.py", "--clean-rerun", "--output", results, *flags)
    if args.long_repeats:
        run("run_benchmarks.py", "--long-repeats", "--output", results)
    run("analyze.py", "--results", results, "--output", out / "plots")
    run("make_report.py", "--results", results, "--output", out)
    print(f"Workflow complete. Report: {out / 'pdf/Project_1_Report.pdf'}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"Workflow stopped: {error}") from None
