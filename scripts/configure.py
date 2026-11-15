"""Detect this host, install dependencies, build, and validate the CPU project."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import venv
import urllib.request

from system_config import CONFIG, ROOT, SCHEMA, identity, source_signature


def run(args, **kwargs):
    print("+", " ".join(str(arg) for arg in args), flush=True)
    return subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True, **kwargs)


def output(args):
    return subprocess.check_output([str(arg) for arg in args], text=True, stderr=subprocess.DEVNULL).strip()


def inspect_system():
    cpu, ram = platform.processor(), None
    system = platform.system()
    try:
        if system == "Darwin":
            cpu = output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"])
            ram = int(output(["/usr/sbin/sysctl", "-n", "hw.memsize"]))
        elif system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
            ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGESIZE")
        elif system == "Windows":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in
                    ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
            memory = MemoryStatus()
            memory.length = ctypes.sizeof(memory)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                ram = memory.total
            cpu = os.environ.get("PROCESSOR_IDENTIFIER", cpu)
    except (OSError, ValueError, subprocess.CalledProcessError):
        pass  # Native topology and memory queries run again after compilation.
    return {"identity": identity(), "platform": platform.platform(), "python": sys.version,
            "cpu": cpu or "not reported", "ram_bytes": ram,
            "tools": {name: shutil.which(name) for name in
                      ("git", "cmake", "ninja", "gcc", "g++", "clang", "clang++", "cl", "brew", "winget")}}


def install_linux_compiler():
    commands = (("apt-get", ["install", "-y", "build-essential", "git"]),
                ("dnf", ["install", "-y", "gcc", "gcc-c++", "libgomp", "git"]),
                ("yum", ["install", "-y", "gcc", "gcc-c++", "libgomp", "git"]),
                ("pacman", ["-S", "--needed", "--noconfirm", "gcc", "git"]),
                ("zypper", ["--non-interactive", "install", "gcc", "gcc-c++", "git"]))
    for manager, args in commands:
        if shutil.which(manager):
            prefix = [] if os.geteuid() == 0 else ["sudo"]
            if prefix and not shutil.which("sudo"):
                break
            if manager == "apt-get":
                run(prefix + [manager, "update"])
            run(prefix + [manager] + args)
            return
    raise RuntimeError("No supported system package manager with installation privileges. Install Git and a C/C++ compiler, then rerun setup.")


def activate_msvc():
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.exists():
        return False
    installations = output([vswhere, "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"])
    if not installations:
        return False
    script = Path(installations) / "Common7/Tools/VsDevCmd.bat"
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    # cmd requires a single quoted command string for a .bat followed by set.
    result = subprocess.check_output(f'cmd /d /s /c ""{script}" -no_logo -arch={arch} && set"', text=True)
    for line in result.splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            os.environ[key] = value
    return bool(shutil.which("cl"))


def select_compilers():
    system = platform.system()
    if os.environ.get("CC") or os.environ.get("CXX"):
        if not (os.environ.get("CC") and os.environ.get("CXX")):
            raise RuntimeError("Set both CC and CXX to use a custom toolchain.")
        return os.environ["CC"], os.environ["CXX"], []
    if system == "Windows":
        if shutil.which("cl") or activate_msvc():
            return "cl", "cl", []
        if shutil.which("gcc") and shutil.which("g++"):
            return shutil.which("gcc"), shutil.which("g++"), []
        if not shutil.which("cl"):
            if not shutil.which("winget"):
                raise RuntimeError("Install Visual Studio C++ Build Tools or MinGW-w64; winget is unavailable for automatic installation.")
            run(["winget", "install", "--id", "Microsoft.VisualStudio.2022.BuildTools", "--exact",
                 "--accept-package-agreements", "--accept-source-agreements", "--override",
                 "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" +
                 (" --add Microsoft.VisualStudio.Component.VC.Tools.ARM64" if platform.machine().lower() in ("arm64", "aarch64") else "")])
            if not activate_msvc():
                raise RuntimeError("C++ Build Tools installation requires a restart or the ARM64 compiler component. Rerun from a native Developer PowerShell after installation.")
        return "cl", "cl", []
    if system == "Darwin":
        # Homebrew GCC includes OpenMP and avoids a separate Apple-Clang libomp setup.
        for base in (Path("/opt/homebrew/bin"), Path("/usr/local/bin")):
            for gcc in sorted(base.glob("gcc-[0-9]*"), reverse=True):
                cxx = gcc.with_name(gcc.name.replace("gcc-", "g++-"))
                target = output([gcc, "-dumpmachine"]) if cxx.exists() else ""
                native = (any(tag in target for tag in ("arm64", "aarch64"))
                          if platform.machine().lower() in ("arm64", "aarch64") else "x86_64" in target)
                if cxx.exists() and native:
                    probe = subprocess.run([str(cxx), "-std=c++17", "-fopenmp", "-x", "c++", "-c", "-o",
                                            str(ROOT / "build/compiler-probe.o"), "-"],
                                           input="#include <mach/mach.h>\nint main() { return 0; }\n",
                                           text=True, capture_output=True)
                    if probe.returncode == 0:
                        return str(gcc), str(cxx), []
                    print(f"{cxx} is incompatible with the active macOS SDK; trying Apple Clang.", flush=True)
        if not shutil.which("clang") or subprocess.run(["xcrun", "--find", "clang"], capture_output=True).returncode:
            run(["xcode-select", "--install"])
            print("Complete Apple's Command Line Tools installer. Setup will continue automatically.", flush=True)
            deadline = time.monotonic() + 1800
            while subprocess.run(["xcrun", "--find", "clang"], capture_output=True).returncode:
                if time.monotonic() > deadline:
                    raise RuntimeError("Command Line Tools installation is still incomplete; rerun setup when it finishes.")
                time.sleep(5)
        native_brew = Path("/opt/homebrew/bin/brew" if platform.machine().lower() in ("arm64", "aarch64") else "/usr/local/bin/brew")
        brew = str(native_brew) if native_brew.exists() else None
        if not brew:
            installer = ROOT / "build/install-homebrew.sh"
            with urllib.request.urlopen("https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh", timeout=60) as response:
                installer.write_bytes(response.read())
            run(["/bin/bash", installer])
            brew = str(native_brew)
        os.environ["PATH"] = str(Path(brew).parent) + os.pathsep + os.environ["PATH"]
        prefix = Path(output([brew, "--prefix", "libomp"]))
        if not (prefix / "lib/libomp.dylib").exists():
            run([brew, "install", "libomp"], env={**os.environ, "HOMEBREW_NO_AUTO_UPDATE": "1", "HOMEBREW_NO_INSTALL_CLEANUP": "1"})
        return shutil.which("clang"), shutil.which("clang++"), [
            "-DOpenMP_C_FLAGS=-Xpreprocessor -fopenmp", "-DOpenMP_CXX_FLAGS=-Xpreprocessor -fopenmp",
            "-DOpenMP_C_LIB_NAMES=omp", "-DOpenMP_CXX_LIB_NAMES=omp",
            f"-DOpenMP_omp_LIBRARY={prefix}/lib/libomp.dylib",
            f"-DOpenMP_C_INCLUDE_DIR={prefix}/include", f"-DOpenMP_CXX_INCLUDE_DIR={prefix}/include"]
    if not (shutil.which("gcc") and shutil.which("g++") and shutil.which("git")):
        install_linux_compiler()
    return shutil.which("gcc"), shutil.which("g++"), []


def save(data):
    temporary = CONFIG.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(CONFIG)


def configure(args):
    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11 or newer is required.")
    if not args.check and any(value is not None and value < 1 for value in (args.jobs, args.threads)):
        raise RuntimeError("--jobs and --threads must be positive")
    info = inspect_system()
    print(json.dumps(info, indent=2), flush=True)
    if info["identity"]["os"] not in ("Windows", "Linux", "Darwin") or info["identity"]["bits"] != 64:
        raise RuntimeError("A 64-bit Windows, Linux, or macOS installation is required.")
    if platform.machine().lower() not in ("x86_64", "amd64", "arm64", "aarch64"):
        raise RuntimeError("Supported CPU architectures are x86-64 and ARM64.")
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    (CONFIG.parent / "system-check.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    if args.check:
        print("Inspection only; setup is not unlocked. Run again without --check to install and configure.")
        return
    try:
        previous = json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    # Invalidate earlier success before touching the environment or build.
    save({**previous, "status": "configuring"})
    python = ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    if Path(sys.prefix).resolve() != python.parent.parent.resolve():
        old_host = previous.get("identity", {})
        if old_host and any(old_host.get(key) != identity().get(key) for key in ("root", "os", "architecture")):
            environment = ROOT / ".venv"
            if environment.exists():
                environment.rename(ROOT / (".venv-previous-" + str(time.time_ns())))
    pip_available = python.exists() and subprocess.run([str(python), "-m", "pip", "--version"], capture_output=True).returncode == 0
    if not pip_available:
        try:
            venv.EnvBuilder(with_pip=True).create(ROOT / ".venv")
        except (subprocess.CalledProcessError, SystemExit):
            if platform.system() != "Linux" or not shutil.which("apt-get"):
                raise RuntimeError("Python's venv/ensurepip support is required; install it and rerun setup.") from None
            prefix = [] if os.geteuid() == 0 else ["sudo"]
            run(prefix + ["apt-get", "update"])
            run(prefix + ["apt-get", "install", "-y", f"python{sys.version_info.major}.{sys.version_info.minor}-venv"])
            venv.EnvBuilder(with_pip=True).create(ROOT / ".venv")
    if Path(sys.prefix).resolve() != python.parent.parent.resolve():
        run([python, Path(__file__).resolve(), *sys.argv[1:]])
        return
    run([python, "-m", "pip", "install", "-r", ROOT / "requirements.txt", "cmake>=3.20,<5", "ninja>=1.11,<2"])
    os.environ["PATH"] = str(python.parent) + os.pathsep + os.environ["PATH"]
    cc, cxx, extra = select_compilers()
    if not shutil.which("git"):
        if platform.system() == "Windows" and shutil.which("winget"):
            run(["winget", "install", "--id", "Git.Git", "--exact", "--accept-package-agreements", "--accept-source-agreements"])
            git_bin = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/cmd"
            os.environ["PATH"] += os.pathsep + str(git_bin)
        if not shutil.which("git"):
            raise RuntimeError("Git is required. Install Git and rerun setup.")
    import psutil
    jobs = args.jobs or min(psutil.cpu_count(logical=False) or 1, max(1, int(psutil.virtual_memory().available / 2**30)))
    if jobs < 1:
        raise RuntimeError("--jobs must be positive")
    build = ROOT / "build/cmake"
    # A toolchain change cannot safely reuse CMake's old compiler cache.
    cache = build / "CMakeCache.txt"
    if cache.exists() and (previous.get("identity") != identity() or
                           previous.get("compiler", {}).get("c") != str(cc) or
                           previous.get("compiler", {}).get("cxx") != str(cxx)):
        shutil.rmtree(build)
    data = {"schema": SCHEMA, "identity": identity(), "source_signature": source_signature(),
            "status": "building", "python": str(python), "bin_dir": str(build / "bin"),
            "compiler": {"c": str(cc), "cxx": str(cxx)}, "jobs": jobs,
            "ram_bytes": psutil.virtual_memory().total, "system": info,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    save(data)
    try:
        source = ROOT / "build/vendor/ggml"
        marker = source / ".project-source.json"
        expected_source = {"commit": "8bb3cf0d216a5ae48360179f2931daf284be4d01", "instrumentation": 1}
        try:
            prepared = json.loads(marker.read_text(encoding="utf-8")) == expected_source
        except (OSError, ValueError):
            prepared = False
        if source.exists() and not prepared:
            source.rename(source.with_name("ggml-previous-" + str(time.time_ns())))
        if not prepared:
            command = [python, ROOT / "scripts/prepare_ggml.py"]
            if args.local_source:
                command += ["--local-source", args.local_source.resolve()]
            run(command)
        run(["cmake", "-S", ROOT, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
             f"-DPython3_EXECUTABLE={python}", f"-DCMAKE_C_COMPILER={cc}", f"-DCMAKE_CXX_COMPILER={cxx}", *extra])
        run(["cmake", "--build", build, "--parallel", jobs])
        run(["ctest", "--test-dir", build, "--output-on-failure"])
        suffix = ".exe" if os.name == "nt" else ""
        hardware = output([build / "bin" / f"machine{suffix}"])
        fields = dict(line.split("=", 1) for line in hardware.splitlines() if "=" in line)
        data["hardware"] = hardware
        data["compiler"]["version"] = output([cc, "--version"]).splitlines()[0] if Path(cc).name.lower() not in ("cl", "cl.exe") else "MSVC"
        llc = sum(int(line.split(",")[1]) for line in hardware.splitlines() if line.startswith("cache=3,"))
        elements = max(20000000, 4 * llc // 8 + 1)
        if elements * 8 * 3 + 1024**3 > psutil.virtual_memory().total:
            raise RuntimeError("Insufficient total RAM for STREAM's 4x LLC sizing rule plus a 1 GiB reserve.")
        data["stream_elements"] = elements
        if elements != 20000000:
            run(["cmake", "-S", ROOT, "-B", build, f"-DPROJECT_STREAM_ARRAY_SIZE={elements}"])
            run(["cmake", "--build", build, "--target", "stream", "--parallel", jobs])
        cpus = [int(x) for x in fields.get("cpu_indices", "").split(",") if x]
        count = int(fields.get("worker_count", "0")) or psutil.cpu_count(logical=False) or 1
        data["cpus"] = cpus
        data["threads"] = min(args.threads or count, count, 512)
        if data["threads"] < 1:
            raise RuntimeError("--threads must be positive")
        data["affinity"] = fields.get("affinity", "unavailable")
        data["status"] = "ready"
        save(data)
        with (ROOT / "build/python-lock.txt").open("w", encoding="utf-8") as lock:
            run([python, "-m", "pip", "freeze"], stdout=lock)
    except BaseException:
        data["status"] = "failed"
        save(data)
        raise
    print(f"Setup complete: {platform.system()} {platform.machine()}, {data['threads']} workers, affinity: {data['affinity']}.")
    print("Next: python scripts/run.py (automatically uses the project Python environment).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Inspect only; do not install or unlock the workflow")
    parser.add_argument("--local-source", type=Path, help="Existing llama.cpp checkout at the pinned revision")
    parser.add_argument("--jobs", type=int, help="Build parallelism; default adapts to available RAM")
    parser.add_argument("--threads", type=int, help="Limit benchmark workers; default uses detected physical/performance cores")
    try:
        configure(parser.parse_args())
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Setup failed: {error}\nResolve the issue and rerun python scripts/configure.py.") from None
