"""Record the peak RSS of a single Linux benchmark child."""
import json
from pathlib import Path
import resource
import subprocess
import sys


def main():
    destination, *command = sys.argv[1:]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, check=False)
    if result.returncode:
        return result.returncode if result.returncode > 0 else 128 - result.returncode
    peak_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if peak_kib <= 0:
        raise RuntimeError("Linux did not report the benchmark process's peak RSS")
    Path(destination).write_text(json.dumps({"peak_rss_bytes": peak_kib * 1024,
                                             "method": "Linux getrusage(RUSAGE_CHILDREN)"},
                                            indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
