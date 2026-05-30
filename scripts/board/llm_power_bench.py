"""Edge power/throughput bench: distilled student vs previously-deployed LLM.

Measures, on the DK-2500 iGPU, for each model:
  - sustained throughput (tok/s, greedy, 128 new tokens/call)
  - package power (PkgWatt) and **iGPU/graphics power (GFXWatt)** via turbostat
  - one-time load latency + on-disk size

Power is sampled by ``turbostat`` (root) for a fixed window while a worker
thread keeps the model busy; idle is sampled with no workload. GFXWatt is the
RAPL graphics domain — the iGPU power draw (no intel_gpu_top on this board).

Run on the board (stop uvicorn first to free the iGPU):
    pkill -f uvicorn
    cd /home/bupt/LabGuardian-Server
    SUDO_PASS=*** /home/bupt/miniconda3/envs/labguardian/bin/python \
        -m scripts.board.llm_power_bench --output /tmp/llm_power_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import openvino_genai as ov_genai

PROMPT = "请分步骤说明共射放大电路输出失真的排查方法，并解释静态工作点的作用。"

_STUDENT = "/home/bupt/models/labguardian-student-1p5-int4-ov"
_GEMMA = "/home/bupt/models/gemma-3-4b-it-int4-ov"

# (label, dir, device) — student on GPU is the production edge config;
# student on CPU justifies the GPU device-routing choice; gemma-3-4b is the
# previously-shipped baseline (3.5x larger) — wrapped in try/except since its
# multimodal export may not load in the text LLMPipeline path.
MODELS = [
    ("student-1.5B-int4 @GPU", _STUDENT, "GPU"),
    ("student-1.5B-int4 @CPU", _STUDENT, "CPU"),
    ("gemma-3-4b-int4 @GPU", _GEMMA, "GPU"),
]


def dir_size_mb(path: str) -> float:
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return round(total / 1e6, 1)


def sample_power(window_s: float, pw: str) -> tuple[list[float], list[float]]:
    """Run turbostat for ``window_s`` and return (pkg_watts, gfx_watts) samples."""
    n = max(2, int(window_s / 0.5))
    proc = subprocess.run(
        ["sudo", "-S", "turbostat", "--quiet",
         "--show", "PkgWatt,CorWatt,GFXWatt,RAMWatt",
         "--interval", "0.5", "--num_iterations", str(n)],
        input=pw + "\n", capture_output=True, text=True, timeout=window_s + 40,
    )
    pkg: list[float] = []
    gfx: list[float] = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            vals = [float(x) for x in parts[:4]]
        except ValueError:
            continue  # header / non-numeric line
        pkg.append(vals[0])
        gfx.append(vals[2])
    return pkg, gfx


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"avg": None, "peak": None, "n": 0}
    return {"avg": round(sum(xs) / len(xs), 2), "peak": round(max(xs), 2), "n": len(xs)}


def bench_model(name: str, model_dir: str, device: str, window_s: float, pw: str) -> dict:
    size_mb = dir_size_mb(model_dir)
    t = time.time()
    pipe = ov_genai.LLMPipeline(model_dir, device=device)
    load_s = time.time() - t

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = 128
    if hasattr(cfg, "do_sample"):
        cfg.do_sample = False
    if hasattr(cfg, "ignore_eos"):
        cfg.ignore_eos = True  # force exactly 128 new tokens → honest tok/s
    # warmup (JIT compile + first-token cost out of the measured window)
    pipe.generate(PROMPT, cfg)

    # 1) accurate tok/s from a single timed full-length generate
    t = time.time()
    pipe.generate(PROMPT, cfg)
    single_lat = time.time() - t
    tps = round(128 / single_lat, 1) if single_lat > 0 else 0.0

    # 2) power over a sustained window (worker keeps the device busy)
    stop = threading.Event()
    counter = [0]

    def hammer() -> None:
        while not stop.is_set():
            pipe.generate(PROMPT, cfg)
            counter[0] += 1

    th = threading.Thread(target=hammer, daemon=True)
    th.start()
    pkg, gfx = sample_power(window_s, pw)   # blocks ~window_s while device is busy
    stop.set()
    th.join(timeout=60)

    return {
        "model": name,
        "model_dir": model_dir,
        "device": device,
        "size_mb": size_mb,
        "load_s": round(load_s, 1),
        "tokens_per_s": tps,
        "per_call_128tok_s": round(single_lat, 2),
        "gen_calls_in_window": counter[0],
        "pkg_watt": _stats(pkg),
        "gfx_watt": _stats(gfx),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/llm_power_bench.json")
    parser.add_argument("--window", type=float, default=18.0)
    args = parser.parse_args()
    pw = os.environ.get("SUDO_PASS", "")

    report: dict = {"idle": None, "models": []}

    print("[idle] sampling baseline power (no model) ...")
    pkg, gfx = sample_power(6.0, pw)
    report["idle"] = {"pkg_watt": _stats(pkg), "gfx_watt": _stats(gfx)}
    print(f"  idle Pkg={report['idle']['pkg_watt']['avg']}W  GFX={report['idle']['gfx_watt']['avg']}W")

    for name, path, device in MODELS:
        if not Path(path).exists():
            print(f"[skip] {name}: dir missing {path}")
            report["models"].append({"model": name, "error": "dir_missing"})
            continue
        print(f"[bench] {name} ...")
        try:
            rec = bench_model(name, path, device, args.window, pw)
        except Exception as exc:  # one model failing must not abort the rest
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:160]}")
            report["models"].append({"model": name, "device": device, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            continue
        report["models"].append(rec)
        print(f"  size={rec['size_mb']}MB load={rec['load_s']}s "
              f"tok/s={rec['tokens_per_s']} (128tok/{rec['per_call_128tok_s']}s)  "
              f"Pkg avg/peak={rec['pkg_watt']['avg']}/{rec['pkg_watt']['peak']}W  "
              f"GFX avg/peak={rec['gfx_watt']['avg']}/{rec['gfx_watt']['peak']}W")

    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
