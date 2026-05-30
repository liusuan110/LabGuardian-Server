"""Deep edge profile for the deployed student model (student-1.5B-int4 @ iGPU).

Collects, in one board run:
  - TTFT / TPOT / throughput via openvino_genai perf_metrics
  - runtime memory footprint (VmRSS delta + VmHWM peak)
  - continuous package + iGPU(GFX) power trace across idle → inference → cooldown
  - energy per answer (J) = inference-phase avg PkgWatt x per-answer latency
  - a publication-ready power time-series PNG (matplotlib, Agg)

Outputs: /tmp/llm_edge_profile.json, /tmp/llm_power_ts.csv, /tmp/llm_power_ts.png

Run on the board (stop uvicorn first to free iGPU):
    SUDO_PASS=*** python -m scripts.board.llm_edge_profile
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import openvino_genai as ov_genai

MODEL = "/home/bupt/models/labguardian-student-1p5-int4-ov"
PROMPT = "请分步骤说明共射放大电路输出失真的排查方法，并解释静态工作点的作用。"
PW = os.environ.get("SUDO_PASS", "")
RAW_CSV = "/tmp/llm_ts_raw.csv"   # turbostat --out (root-owned)
CSV = "/tmp/llm_power_ts.csv"      # labeled, user-written


def _proc_kb(field: str) -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])  # kB
    return -1


def _mean(pair) -> float:
    try:
        return float(pair.mean)
    except Exception:
        try:
            return float(pair)
        except Exception:
            return -1.0


def main() -> None:
    base_rss = _proc_kb("VmRSS")

    t = time.time()
    pipe = ov_genai.LLMPipeline(MODEL, device="GPU")
    load_s = time.time() - t
    load_rss = _proc_kb("VmRSS")

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = 128
    if hasattr(cfg, "do_sample"):
        cfg.do_sample = False
    if hasattr(cfg, "ignore_eos"):
        cfg.ignore_eos = True
    pipe.generate(PROMPT, cfg)  # warmup (JIT)

    # --- TTFT / TPOT / throughput + per-answer latency (manual, version-robust) ---
    # generate() returns a plain str in this openvino_genai build, so perf_metrics
    # isn't on the return value; we time a 1-token generate (TTFT = prefill +
    # first token) and a full 128-token generate, then derive TPOT.
    cfg1 = ov_genai.GenerationConfig()
    cfg1.max_new_tokens = 1
    if hasattr(cfg1, "do_sample"):
        cfg1.do_sample = False
    if hasattr(cfg1, "ignore_eos"):
        cfg1.ignore_eos = True
    t = time.time()
    pipe.generate(PROMPT, cfg1)
    ttft = (time.time() - t) * 1000.0  # ms (time-to-first-token incl. prefill)

    t = time.time()
    pipe.generate(PROMPT, cfg)
    single_lat = time.time() - t      # full 128-token answer latency (s)
    tpot = (single_lat * 1000.0 - ttft) / 127.0  # ms per subsequent token
    thr = 128.0 / single_lat if single_lat > 0 else -1.0

    peak_rss = _proc_kb("VmHWM")

    # --- continuous power trace: launch turbostat (~40s) ---
    proc = subprocess.Popen(
        ["sudo", "-S", "turbostat", "--quiet",
         "--show", "Time_Of_Day_Seconds,PkgWatt,GFXWatt,CorWatt",
         "--interval", "0.5", "--num_iterations", "80", "--out", RAW_CSV],
        stdin=subprocess.PIPE, text=True,
    )
    proc.stdin.write(PW + "\n")
    proc.stdin.flush()
    time.sleep(1.5)  # let turbostat warm up

    b_idle = time.time()
    time.sleep(6)
    b_infer = time.time()
    stop = threading.Event()

    def hammer() -> None:
        while not stop.is_set():
            pipe.generate(PROMPT, cfg)

    th = threading.Thread(target=hammer, daemon=True)
    th.start()
    time.sleep(22)
    stop.set()
    th.join(timeout=60)
    b_cool = time.time()
    time.sleep(6)
    b_end = time.time()
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.terminate()
    subprocess.run(["sudo", "-S", "chmod", "644", RAW_CSV], input=PW + "\n", text=True)

    # --- parse trace ---
    ts: list[float] = []
    pkg: list[float] = []
    gfx: list[float] = []
    col = {}
    for line in Path(RAW_CSV).read_text().splitlines():
        parts = line.split()
        if "PkgWatt" in line:  # header
            col = {name: idx for idx, name in enumerate(parts)}
            continue
        if not col or len(parts) < len(col):
            continue
        try:
            ts.append(float(parts[col["Time_Of_Day_Seconds"]]))
            pkg.append(float(parts[col["PkgWatt"]]))
            gfx.append(float(parts[col["GFXWatt"]]))
        except (ValueError, KeyError):
            continue

    t0 = ts[0] if ts else b_idle
    elapsed = [x - t0 for x in ts]

    def phase_avg(lo: float, hi: float, series: list[float]) -> dict:
        # select samples whose absolute epoch falls inside the phase window
        vals = [v for tt, v in zip(ts, series) if lo <= tt <= hi]
        if not vals:
            return {"avg": None, "peak": None, "n": 0}
        return {"avg": round(sum(vals) / len(vals), 2), "peak": round(max(vals), 2), "n": len(vals)}

    idle_pkg = phase_avg(b_idle, b_infer, pkg)
    idle_gfx = phase_avg(b_idle, b_infer, gfx)
    infer_pkg = phase_avg(b_infer, b_cool, pkg)
    infer_gfx = phase_avg(b_infer, b_cool, gfx)

    energy_j = None
    if infer_pkg["avg"] and single_lat > 0:
        energy_j = round(infer_pkg["avg"] * single_lat, 1)

    summary = {
        "model": "student-1.5B-int4 @ iGPU",
        "load_s": round(load_s, 2),
        "ttft_ms": round(ttft, 1),
        "tpot_ms_per_token": round(tpot, 2),
        "throughput_tok_s": round(thr, 1),
        "per_answer_128tok_s": round(single_lat, 2),
        "energy_per_128tok_answer_J": energy_j,
        "mem_model_footprint_mb": round((load_rss - base_rss) / 1024, 1),
        "mem_runtime_rss_mb": round(load_rss / 1024, 1),
        "mem_peak_vmhwm_mb": round(peak_rss / 1024, 1),
        "power_idle_pkg_w": idle_pkg,
        "power_idle_gfx_w": idle_gfx,
        "power_infer_pkg_w": infer_pkg,
        "power_infer_gfx_w": infer_gfx,
        "trace_samples": len(ts),
    }
    Path("/tmp/llm_edge_profile.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # write a clean labeled CSV
    def phase_of(tt: float) -> str:
        if tt < b_infer:
            return "idle"
        if tt < b_cool:
            return "inference"
        return "cooldown"
    lines = ["t_s,PkgWatt,GFXWatt,phase"]
    for tt, p, g in zip(ts, pkg, gfx):
        lines.append(f"{tt - t0:.2f},{p:.2f},{g:.2f},{phase_of(tt)}")
    Path(CSV).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.plot(elapsed, pkg, label="Package power (W)", color="#c0392b", lw=1.6)
        ax.plot(elapsed, gfx, label="iGPU / GFX power (W)", color="#2980b9", lw=1.6)
        for (lo, hi, c, lab) in [
            (b_idle - t0, b_infer - t0, "#ecf0f1", "idle"),
            (b_infer - t0, b_cool - t0, "#fdebd0", "LLM inference"),
            (b_cool - t0, b_end - t0, "#ecf0f1", "cooldown"),
        ]:
            ax.axvspan(lo, hi, color=c, alpha=0.6, zorder=0)
            ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.95 if False else 0.5, lab,
                    ha="center", va="bottom", fontsize=8, color="#555")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("power (W)")
        ax.set_title("Student-1.5B-INT4 @ DK-2500 iGPU — power timeline")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("/tmp/llm_power_ts.png", dpi=140)
        print("plot saved /tmp/llm_power_ts.png")
    except Exception as exc:
        print("plot skipped:", exc)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
