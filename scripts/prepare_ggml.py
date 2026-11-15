"""Fetch a pinned upstream tree and insert optional, scoped GDN phase timers."""
import argparse
import json
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMIT = "8bb3cf0d216a5ae48360179f2931daf284be4d01"


def prepare(local_source=None):
    target = ROOT / "build/vendor/ggml"
    if target.exists():
        raise SystemExit("Source already prepared. Reuse it or use a fresh build directory.")
    if local_source:
        source = pathlib.Path(local_source)
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if actual != COMMIT:
            raise SystemExit(f"Expected {COMMIT}, found {actual}")
        # Export the committed tree, so unrelated local changes never enter this project.
        archive = subprocess.check_output(["git", "-C", str(source), "archive", "HEAD", "ggml"])
        import io
        import tarfile
        target.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(target.parent, filter="data")
        (target / "LICENSE").write_bytes(subprocess.check_output(
            ["git", "-C", str(source), "show", "HEAD:LICENSE"]))
    else:
        clone = ROOT / "build/vendor/llama.cpp"
        clone.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(clone)], check=True)
        if subprocess.run(["git", "-C", str(clone), "remote", "get-url", "origin"], capture_output=True).returncode:
            subprocess.run(["git", "-C", str(clone), "remote", "add", "origin", "https://github.com/ggml-org/llama.cpp.git"], check=True)
        subprocess.run(["git", "-C", str(clone), "fetch", "--depth=1", "origin", COMMIT], check=True)
        subprocess.run(["git", "-C", str(clone), "checkout", "--detach", "FETCH_HEAD"], check=True)
        shutil.copytree(clone / "ggml", target)
        shutil.copyfile(clone / "LICENSE", target / "LICENSE")
    cpu = target / "src/ggml-cpu/ggml-cpu.c"
    text = cpu.read_text(encoding="utf-8")
    text = text.replace("#if _WIN32_WINNT >= 0x0602\n        THREAD_POWER", "#if _WIN32_WINNT >= 0x0602 && defined(THREAD_POWER_THROTTLING_CURRENT_VERSION) && defined(THREAD_POWER_THROTTLING_EXECUTION_SPEED)\n        THREAD_POWER")
    cpu.write_text(text, encoding="utf-8")
    ops = target / "src/ggml-cpu/ops.cpp"
    text = ops.read_text(encoding="utf-8")
    start = text.index("static void ggml_compute_forward_gated_delta_net_one_chunk(")
    end = text.index("static void ggml_compute_forward_gated_delta_net_f32(", start)
    part = text[start:end]
    inserts = [
        ("        memcpy(s_out, s_in, S_v * S_v * sizeof(float));", "        PROJECT_TICK(0);\n        memcpy(s_out, s_in, S_v * S_v * sizeof(float));\n        PROJECT_TOCK(0);"),
        ("            if (kda) {", "            PROJECT_TICK(1);\n            if (kda) {"),
        ("            // delta[j]", "            PROJECT_TOCK(1);\n            PROJECT_TICK(2);\n            // delta[j]"),
        ("            // outer product:", "            PROJECT_TOCK(2);\n            PROJECT_TICK(3);\n            // outer product:"),
        ("            // attn_out[j]", "            PROJECT_TOCK(3);\n            PROJECT_TICK(4);\n            // attn_out[j]"),
        ("            attn_data +=", "            PROJECT_TOCK(4);\n            attn_data +="),
    ]
    for old,new in inserts:
        if part.count(old)!=1:
            raise RuntimeError(f"Pinned source mismatch for {old!r}")
        part=part.replace(old,new)
    helpers = '''#include <chrono>
static bool project_profiling = false;
static double project_phase_seconds[512][5] = {};
extern "C" void project_profile_enable(int enabled) {
    project_profiling = enabled != 0;
    if (enabled) for (auto &row : project_phase_seconds) for (double &v : row) v=0;
}
extern "C" double project_profile_read(int phase) {
    double sum=0;
    for (auto &row : project_phase_seconds) sum+=row[phase];
    return sum;
}
#define PROJECT_TICK(P) auto project_start_##P = project_profiling ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}
#define PROJECT_TOCK(P) if(project_profiling) project_phase_seconds[params->ith][P] += std::chrono::duration<double>(std::chrono::steady_clock::now()-project_start_##P).count()
'''
    ops.write_text(text[:start]+helpers+part+text[end:],encoding="utf-8")
    (target / ".project-source.json").write_text(json.dumps({"commit": COMMIT, "instrumentation": 1, "license": True}), encoding="utf-8")
    print(f"Prepared ggml {COMMIT}; CPU-only build, optional phase timers")


if __name__=="__main__":
    from system_config import require_configuration
    require_configuration(for_build=True)
    parser=argparse.ArgumentParser()
    parser.add_argument("--local-source")
    args=parser.parse_args()
    prepare(args.local_source)
