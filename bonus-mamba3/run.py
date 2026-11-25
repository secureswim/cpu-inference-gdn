"""Rebuild, validate, benchmark and plot the bonus from a single command."""
import argparse
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--cpu", type=int)
parser.add_argument("--ggml", action="store_true")
args = parser.parse_args()
build = root/"build/cmake"
(root/"results").mkdir(exist_ok=True)

def run(*cmd):
    subprocess.run(cmd, cwd=root, check=True)

run("cmake","-S",".","-B",str(build),"-G","Ninja","-DCMAKE_BUILD_TYPE=Release",
    "-DBONUS_GGML="+("ON" if args.ggml else "OFF"))
run("cmake","--build",str(build),"--parallel","4")
run(sys.executable,"tests/check_mamba.py")
run(sys.executable,"tests/check_identity.py")
if args.ggml:
    executable = "mamba_ggml.exe" if sys.platform == "win32" else "mamba_ggml"
    output = subprocess.check_output([build/executable],cwd=root,text=True)
    (root/"results/ggml.txt").write_text(output)
cpu_args = [] if args.cpu is None else ["--cpu",str(args.cpu)]
run(sys.executable,"scripts/benchmark.py",*cpu_args)
run(sys.executable,"scripts/plot.py")
