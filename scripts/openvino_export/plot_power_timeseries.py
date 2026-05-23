"""Plot power vs time from RAPL turbostat samples.

Inputs:  scripts/openvino_export/results/power_timeseries.csv + power_phases.json
Outputs: scripts/openvino_export/results/power_timeseries.{png,pdf}
"""
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent / "results"
CSV = ROOT / "power_timeseries.csv"
META = ROOT / "power_phases.json"

# 学术 IEEE 风格
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
})


def load():
    rows = []
    with open(CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "t": float(row["t_s"]),
                "pkg": float(row["PkgWatt"]),
                "cor": float(row["CorWatt"]),
                "gfx": float(row["GFXWatt"]),
                "phase": row["phase"],
            })
    with open(META) as f:
        meta = json.load(f)
    return rows, meta


def main():
    rows, meta = load()
    # 把 turbostat 报的 t 重新校准到真实 wallclock (脚本说 SAMPLE_INTERVAL=0.5 但实际更快)
    # 用最后一个 phase 的 end_s 推算
    last_end = meta["phases"][-1]["end_s"]  # 真实结束时间
    n = len(rows)
    actual_interval = last_end / n
    for i, r in enumerate(rows):
        r["t"] = i * actual_interval

    pkg = [r["pkg"] for r in rows]
    cor = [r["cor"] for r in rows]
    gfx = [r["gfx"] for r in rows]
    t = [r["t"] for r in rows]
    # NPU 是 PkgWatt - CorWatt - GFXWatt (扣 uncore 噪声，给个近似)
    npu_approx = [max(0, p - c - g - 4.42) for p, c, g in zip(pkg, cor, gfx)]  # 4.42 = idle baseline

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=120)

    # Phase background bands (彩色弱衬)
    phase_colors = {
        "idle_pre": "#F5F5F5",
        "idle_post": "#F5F5F5",
        "cooldown_1": "#F5F5F5",
        "cooldown_2": "#F5F5F5",
        "CPU_workload": "#FEE2E2",   # 浅红
        "GPU_workload": "#DBEAFE",   # 浅蓝
        "NPU_workload": "#DCFCE7",   # 浅绿
    }
    phase_labels = {
        "CPU_workload": "CPU INT8",
        "GPU_workload": "iGPU INT8",
        "NPU_workload": "NPU INT8",
    }
    for ph in meta["phases"]:
        ax.axvspan(ph["start_s"], ph["end_s"], color=phase_colors.get(ph["name"], "#FAFAFA"),
                   alpha=0.55, zorder=0)
        if ph["name"] in phase_labels:
            mid = (ph["start_s"] + ph["end_s"]) / 2
            ax.text(mid, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 25,
                    phase_labels[ph["name"]], ha="center", va="top", fontsize=9,
                    fontweight="bold", color="#374151")

    # 主曲线
    ax.plot(t, pkg, label="Package (total)", color="#1F2937", linewidth=2.0, zorder=3)
    ax.plot(t, cor, label="CPU cores", color="#DC2626", linewidth=1.4, zorder=2)
    ax.plot(t, gfx, label="iGPU", color="#2563EB", linewidth=1.4, zorder=2)

    # idle baseline 虚线
    ax.axhline(y=4.42, color="#6B7280", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
    ax.text(t[-1] * 0.98, 4.42 + 0.4, "idle 4.4W", fontsize=8, color="#6B7280",
            ha="right", va="bottom")

    # 标注 peak
    cpu_peak = max(pkg[int(len(pkg) * 0.1):int(len(pkg) * 0.3)])  # CPU phase
    gpu_peak = max(pkg[int(len(pkg) * 0.4):int(len(pkg) * 0.6)])  # GPU phase
    npu_peak = max(pkg[int(len(pkg) * 0.7):int(len(pkg) * 0.9)])  # NPU phase

    def annotate(x, y, text, color):
        ax.annotate(text, xy=(x, y), xytext=(x, y + 2.5),
                    fontsize=9, fontweight="bold", color=color,
                    ha="center", arrowprops=dict(arrowstyle="-", color=color, alpha=0.6, lw=0.8))

    # 找 peak 时间
    for ph in meta["phases"]:
        if "workload" not in ph["name"]:
            continue
        mask = [ph["start_s"] <= ti <= ph["end_s"] for ti in t]
        seg_pkg = [p for p, m in zip(pkg, mask) if m]
        seg_t = [ti for ti, m in zip(t, mask) if m]
        if not seg_pkg: continue
        peak_idx = seg_pkg.index(max(seg_pkg))
        peak_t = seg_t[peak_idx]
        peak_w = seg_pkg[peak_idx]
        color = {"CPU_workload": "#DC2626", "GPU_workload": "#2563EB", "NPU_workload": "#16A34A"}[ph["name"]]
        annotate(peak_t, peak_w, f"{peak_w:.1f}W", color)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Power (Watts)", fontsize=11)
    ax.set_title("YOLOv8s-pose INT8 — RAPL Power Consumption per Compute Unit\n"
                 "Intel® Core™ Ultra 5 225U (DK-2500)",
                 fontsize=11, pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, max(pkg) * 1.15)
    ax.grid(True, alpha=0.3, linestyle=":", zorder=0)

    plt.tight_layout()
    out_png = ROOT / "power_timeseries.png"
    out_pdf = ROOT / "power_timeseries.pdf"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"saved: {out_png}")
    print(f"saved: {out_pdf}")

    # 数据汇总
    print("\nphase peak / mean / throughput:")
    for ph in meta["phases"]:
        mask = [ph["start_s"] <= ti <= ph["end_s"] for ti in t]
        seg = [p for p, m in zip(pkg, mask) if m]
        if not seg: continue
        dur = ph["end_s"] - ph["start_s"]
        ips = ph["n_infs"] / dur if dur > 0 else 0
        print(f"  {ph['name']:<14} peak={max(seg):5.2f}W  mean={sum(seg)/len(seg):5.2f}W"
              f"  ips={ips:5.1f}  n_inf={ph['n_infs']}")


if __name__ == "__main__":
    main()
