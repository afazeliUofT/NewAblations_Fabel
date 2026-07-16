#!/usr/bin/env python3
"""Upgraded TWC figure set -> Upgraded_Plots/ (same filenames as the paper)."""
from __future__ import annotations

import glob
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("Upgraded_Plots")
OUT.mkdir(exist_ok=True)

ROOTS = [Path(p) for p in os.environ.get(
    "UPGRADE_ROOTS",
    ".:" + os.environ.get("UPGRADE_WAVE3", ".") + ":" + os.environ.get("UPGRADE_S1P2", ".")
).split(":")]
W3IN = Path(os.environ.get("UPGRADE_WAVE3", ".")) / "_isolated_eval_chunks"   # tuned DnCNN
S1IN = Path(os.environ.get("UPGRADE_S1P2", ".")) / "_isolated_eval_chunks"    # Lite-96/64
IN = "_isolated_eval_chunks"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.4, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.2, "lines.markersize": 3.6,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "grid.alpha": 0.35, "legend.framealpha": 0.9, "legend.handlelength": 2.3,
})

STY = {
    "ls":      dict(color="tab:blue",   ls="--", marker="o", label="LS"),
    "2dl":     dict(color="tab:green",  ls="-.", marker="D", label="LS + 2D LMMSE (matched)"),
    "2dl_st":  dict(color="0.35",       ls=":",  marker="v", label="LS + 2D LMMSE (stale $\\mathbf{R}$)"),
    "2dl_wc":  dict(color="darkgoldenrod", ls="-.", marker="P", label="2D LMMSE (worst-case $\\mathbf{R}$)"),
    "upair":   dict(color="tab:red",    ls="-",  marker="^", label="UPAIR"),
    "perfect": dict(color="tab:purple", ls=(0, (3, 1, 1, 1)), marker="*", label="Perfect CSI"),
    "a3":      dict(color="black",      ls="--", marker="x", label="UPAIR w/o prompt"),
    "p2c":     dict(color="0.55",       ls=":",  marker="s", label="UPAIR, static FiLM"),
    "p3m":     dict(color="tab:orange", ls="--", marker="P", label="UPAIR, mean prompt"),
    "p3w":     dict(color="tab:brown",  ls=":",  marker="X", label="UPAIR, wrong prompt"),
    "a1":      dict(color="tab:cyan",   ls="--", marker="s", label="width $d{=}192$"),
    "a2":      dict(color="tab:olive",  ls="-.", marker="D", label="depth $L{=}2$"),
    "l192":    dict(color="tab:pink",   ls="--", marker="v", label="UPAIR-Lite ($d{=}192$)"),
    "l128":    dict(color="m",          ls=":",  marker="<", label="UPAIR-Lite ($d{=}128$)"),
    "l96":     dict(color="indigo",     ls=(0, (5, 1)), marker=">", label="UPAIR-Lite ($d{=}96$)"),
    "l64":     dict(color="darkgoldenrod", ls=":", marker="P", label="UPAIR-Lite ($d{=}64$)"),
}
RX = {"upair": "upair5g_lmmse", "ls": "baseline_ls_lmmse",
      "2dl": "baseline_ls_2dlmmse_lmmse", "perfect": "perfect_csi_lmmse"}

_cache: dict[str, pd.DataFrame] = {}
_where: dict[str, Path] = {}


def find_root(name: str) -> Path:
    if name not in _where:
        for r in ROOTS:
            p = r / name
            if glob.glob(str(p / "*" / "chunk_result.csv")):
                _where[name] = p
                break
        else:
            raise FileNotFoundError(f"no chunk data for '{name}' under any of {[str(r) for r in ROOTS]}")
    return _where[name]


def load_root(root) -> pd.DataFrame:
    p = Path(root) if str(root).startswith("/") else find_root(str(root))
    key = str(p)
    if key not in _cache:
        files = glob.glob(str(p / "*" / "chunk_result.csv"))
        if not files:
            raise FileNotFoundError(f"no chunk_result.csv under {p}")
        _cache[key] = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return _cache[key]


def curve(root, variant: str, receiver: str, u: int):
    df = load_root(root)
    d = df[(df.variant == variant) & (df.receiver == receiver) & (df.num_users == u)]
    g = d.groupby("ebno_db").agg(be=("block_errors", "sum"), nb=("num_blocks", "sum"))
    g = g[g.be > 0]
    if not len(g):
        raise FileNotFoundError(f"no rows for {variant}/{receiver}/u{u} in {root}")
    g["bler"] = g.be / g.nb
    return g.index.values.astype(float), g.bler.values


def try_curve(root, variant, receiver, u, what=""):
    try:
        return curve(root, variant, receiver, u)
    except FileNotFoundError as e:
        note(f"SKIP {what or variant}: {e}")
        return None


def crossing(x, y, target=1e-2):
    for (x1, y1), (x2, y2) in zip(zip(x, y), zip(x[1:], y[1:])):
        if y1 >= target >= y2:
            return x1 + (x2 - x1) * np.log10(target / y1) / np.log10(y2 / y1)
    return None


def point(xy, ebno):
    if xy is None:
        return None
    for xv, yv in zip(*xy):
        if abs(xv - ebno) < 1e-6:
            return yv
    return None


def bler_axes(ax, xlabel="$E_b/N_0$ (dB)"):
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("BLER")
    ax.grid(True, which="both")


def plot_curve(ax, xy, key, **over):
    if xy is None:
        return
    st = dict(STY[key]); st.update(over)
    ax.plot(*xy, **st)


NUMBERS: list[str] = []


def note(msg):
    NUMBERS.append(msg)
    print("[NUM]", msg)


def fig_r1():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65))
    ax = axes[0]
    for key in ["ls", "2dl", "upair", "perfect"]:
        plot_curve(ax, curve(IN, "main_d256_b4_r2", RX[key], 3), key,
                   **({"label": "LS + 2D LMMSE"} if key == "2dl" else {}))
    bler_axes(ax)
    ax.legend(loc="lower left")
    ax.set_title("(a)", loc="left", fontweight="bold", pad=3)

    ax = axes[1]
    users = [1, 2, 3, 4]
    for key in ["ls", "2dl", "upair"]:
        snr = [crossing(*curve(IN, "main_d256_b4_r2", RX[key], u)) for u in users]
        ax.plot(users, snr, **dict(STY[key]))
        note(f"R1b required-SNR@1e-2 {key}: " + ", ".join(
            f"u{u}:{s:.2f}" if s is not None else f"u{u}:--" for u, s in zip(users, snr)))
    ax.set_xlabel("number of scheduled users $U$")
    ax.set_ylabel("required $E_b/N_0$ at BLER $10^{-2}$ (dB)")
    ax.set_xticks(users)
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.set_title("(b)", loc="left", fontweight="bold", pad=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_r1_indist_scaling.pdf")
    plt.close(fig)


def fig_r2():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65))
    ax = axes[0]
    plot_curve(ax, curve(IN, "main_d256_b4_r2", RX["2dl"], 3), "2dl")
    arms = [("main_d256_b4_r2", "upair"), ("width_light_d192_b4_r2", "a1"),
            ("shallow_light_d256_b2_r2", "a2"), ("upair_lite_d192_b2", "l192"),
            ("upair_lite_d128_b2", "l128"), ("upair_lite_d96_b2", "l96"),
            ("upair_lite_d64_b2", "l64")]
    for v, key in arms:
        plot_curve(ax, curve(IN, v, RX["upair"], 3), key,
                   label=("UPAIR (full)" if key == "upair" else STY[key]["label"]))
    bler_axes(ax)
    ax.legend(loc="lower left", fontsize=5.2)
    ax.set_title("(a)", loc="left", fontweight="bold", pad=3)

    ax = axes[1]
    ref = crossing(*curve(IN, "main_d256_b4_r2", RX["upair"], 3))
    pts = [(IN, "main_d256_b4_r2", 4.189, "full"),
           (IN, "width_light_d192_b4_r2", 2.380, "$d{=}192$"),
           (IN, "shallow_light_d256_b2_r2", 2.207, "$L{=}2$"),
           (IN, "upair_lite_d192_b2", 0.965, "Lite-192"),
           (IN, "upair_lite_d128_b2", 0.447, "Lite-128"),
           (str(S1IN), "upair_lite_d96_b2", 0.261, "Lite-96"),
           (str(S1IN), "upair_lite_d64_b2", 0.125, "Lite-64")]
    for root, v, mp, lab in pts:
        xy = try_curve(root, v, RX["upair"], 3, what=lab)
        if xy is None:
            continue
        d = crossing(*xy) - ref
        note(f"R2b {lab}: params={mp}M delta={d:+.2f} dB")
        ax.plot(mp, d, "o", color="tab:red", ms=5)
        offs = {"full": ((5, 3), "left"), "$d{=}192$": ((4, -12), "left"),
                "$L{=}2$": ((-5, 4), "right"), "Lite-128": ((6, -10), "left"),
                "Lite-96": ((6, -10), "left"), "Lite-192": ((5, 4), "left"),
                "Lite-64": ((6, -10), "left")}
        off, ha = offs.get(lab, ((5, 4), "left"))
        ax.annotate(lab, (mp, d), textcoords="offset points", xytext=off, ha=ha, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (millions)")
    ax.set_ylabel("loss vs.\\ full UPAIR at BLER $10^{-2}$ (dB)")
    ax.grid(True, which="both")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_title("(b)", loc="left", fontweight="bold", pad=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_r2_capacity_frontier.pdf")
    plt.close(fig)


def fig_r3():
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    plot_curve(ax, curve(IN, "main_d256_b4_r2", RX["2dl"], 3), "2dl")
    arms = [
        (IN, "main_d256_b4_r2",                     dict(color="tab:red",   ls="-",  marker="^", label="UPAIR (full)")),
        (IN, "freq_attn_only_d256_b4_r2",           dict(color="tab:green", ls="--", marker="p", label="frequency attention only")),
        (IN, "time_attn_only_d256_b4_r2",           dict(color="tab:blue",  ls="-.", marker="s", label="time attention only")),
        (IN, "local_only_no_axial_attn_d256_b4_r2", dict(color="tab:orange",ls=":",  marker="o", label="no attention")),
        (IN, "no_attn_no_film_d256_b4_r2",          dict(color="0.35",      ls="--", marker="x", label="no attention, no conditioning")),
        (str(W3IN), "dncnn_trunk_d256_l7",          dict(color="tab:brown", ls=":",  marker="v", label="generic DnCNN trunk (tuned)")),
    ]
    ref = crossing(*curve(IN, "main_d256_b4_r2", RX["upair"], 3))
    for root, v, st in arms:
        xy = try_curve(root, v, RX["upair"], 3, what=st["label"])
        if xy is None:
            continue
        ax.plot(*xy, **st)
        c = crossing(*xy)
        note(f"R3 {st['label']}: {(c - ref):+.2f} dB")
    bler_axes(ax)
    ax.legend(loc="lower left", fontsize=5.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig_r3_mechanism_curves.pdf")
    plt.close(fig)


def fig_r4():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7))
    ax = axes[0]
    npz = None
    for r in ROOTS:
        p = r / "prompt_probe_out" / "prompts.npz"
        if p.exists():
            npz = p
            break
    dat = np.load(npz)
    P, L = dat["prompts"].astype(float), dat["labels"]
    Z = P - P.mean(0)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    pc = Z @ Vt[:2].T
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    for u in [1, 2, 3, 4]:
        m = L[:, 1] == u
        sc = ax.scatter(pc[m, 0], pc[m, 1], c=L[m, 0], cmap="viridis",
                        marker=markers[u], s=9, alpha=0.75, linewidths=0,
                        vmin=L[:, 0].min(), vmax=L[:, 0].max(), label=f"$U={u}$")
    cb = fig.colorbar(sc, ax=ax, pad=0.015)
    cb.set_label("$E_b/N_0$ (dB)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_xlabel("principal component 1")
    ax.set_ylabel("principal component 2")
    leg = ax.legend(loc="upper left", ncol=2, columnspacing=0.8, handletextpad=0.2)
    for h in leg.legend_handles:
        h.set_color("0.3")
    ax.set_title("(a)", loc="left", fontweight="bold", pad=3)

    ax = axes[1]
    plot_curve(ax, curve(IN, "main_d256_b4_r2", RX["2dl"], 3), "2dl")
    for v, key in [("main_d256_b4_r2", "upair"), ("prompt_mean_swap_d256_b4_r2", "p3m"),
                   ("prompt_wrong_swap_d256_b4_r2", "p3w")]:
        plot_curve(ax, curve(IN, v, RX["upair"], 3), key,
                   label=("UPAIR (own prompt)" if key == "upair" else STY[key]["label"]))
    bler_axes(ax)
    ax.legend(loc="lower left", fontsize=6.0)
    ax.set_title("(b)", loc="left", fontweight="bold", pad=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_r4_prompt_evidence.pdf")
    plt.close(fig)


def fig_r5():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45))
    ax = axes[0]
    plot_curve(ax, curve("_gen_ds3x_chunks", "main_d256_b4_r2", RX["upair"], 3), "upair",
               label="UPAIR (frozen)")
    plot_curve(ax, try_curve("_gen_ds3x_chunks", "main_d256_b4_r2", RX["2dl"], 3, "matched@3x"), "2dl",
               label="2D LMMSE (matched $\\mathbf{R}$)")
    plot_curve(ax, curve("_r1_ds3x_300ns_chunks", "main_d256_b4_r2", RX["2dl"], 3), "2dl_st",
               label="2D LMMSE (stale $\\mathbf{R}$)")
    plot_curve(ax, curve("_gen_ds3x_p2c_chunks", "constant_prompt_d256_b4_r2", RX["upair"], 3), "p2c")
    plot_curve(ax, curve("_gen_ds3x_chunks", "no_prompt_film_d256_b4_r2", RX["upair"], 3), "a3")
    plot_curve(ax, curve("_gen_ds3x_chunks", "main_d256_b4_r2", RX["ls"], 3), "ls")
    bler_axes(ax)
    ax.legend(loc="lower left", fontsize=5.6)
    ax.set_title("(a)", loc="left", fontweight="bold", pad=3)

    ax = axes[1]
    mults = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    genroot = {1.5: "_gen_ds1p5_chunks", 2.0: "_gen_ds2x_chunks", 3.0: "_gen_ds3x_chunks",
               4.0: "_gen_ds4x_chunks", 5.0: "_gen_ds5x_chunks"}
    stroot = {1.5: "_r1_ds1p5_300ns_chunks", 2.0: "_r1_ds2x_300ns_chunks", 3.0: "_r1_ds3x_300ns_chunks",
              4.0: "_r1_ds4x_300ns_chunks", 5.0: "_r1_ds5x_300ns_chunks"}
    series = {"upair": [], "a3": [], "2dl_st": [], "2dl": []}
    for m in mults:
        if m == 1.0:
            series["upair"].append(point(try_curve(IN, "main_d256_b4_r2", RX["upair"], 3), 0))
            series["a3"].append(point(try_curve(IN, "no_prompt_film_d256_b4_r2", RX["upair"], 3), 0))
            series["2dl_st"].append(point(try_curve(IN, "main_d256_b4_r2", RX["2dl"], 3), 0))
            series["2dl"].append(point(try_curve(IN, "main_d256_b4_r2", RX["2dl"], 3), 0))
        else:
            series["upair"].append(point(try_curve(genroot[m], "main_d256_b4_r2", RX["upair"], 3, f"upair@{m}x"), 0))
            series["a3"].append(point(try_curve(genroot[m], "no_prompt_film_d256_b4_r2", RX["upair"], 3, f"a3@{m}x"), 0))
            series["2dl_st"].append(point(try_curve(stroot[m], "main_d256_b4_r2", RX["2dl"], 3, f"stale@{m}x"), 0))
            series["2dl"].append(point(try_curve(genroot[m], "main_d256_b4_r2", RX["2dl"], 3, f"matched@{m}x"), 0))
    for key, lab in [("upair", "UPAIR (frozen)"), ("2dl_st", "2D LMMSE (stale $\\mathbf{R}$)"),
                     ("2dl", "2D LMMSE (matched $\\mathbf{R}$)"), ("a3", "UPAIR w/o prompt")]:
        xs = [m for m, v in zip(mults, series[key]) if v is not None]
        ys = [v for v in series[key] if v is not None]
        if not xs:
            continue
        st = dict(STY[key]); st["label"] = lab
        ax.plot(xs, ys, **st)
        note(f"R5b {lab} BLER@0dB: " + ", ".join(f"{m}x:{v:.1e}" for m, v in zip(xs, ys)))
    ax.set_yscale("log")
    ax.set_xlabel("delay-spread scale (evaluation / training)")
    ax.set_ylabel("BLER at $E_b/N_0=0$ dB")
    ax.set_xticks(mults)
    ax.grid(True, which="both")
    ax.legend(loc="lower right", fontsize=5.6)
    ax.set_title("(b)", loc="left", fontweight="bold", pad=3)

    ax = axes[2]
    plot_curve(ax, curve("_gen_ds1p5_chunks", "main_d256_b4_r2", RX["2dl"], 4), "2dl",
               label="2D LMMSE (matched $\\mathbf{R}$)")
    plot_curve(ax, curve("_gen_ds1p5_chunks", "main_d256_b4_r2", RX["upair"], 4), "upair",
               label="UPAIR (frozen)")
    plot_curve(ax, curve("_gen_ds1p5_chunks", "no_prompt_film_d256_b4_r2", RX["upair"], 4), "a3")
    bler_axes(ax)
    ax.legend(loc="lower left", fontsize=6.0)
    ax.set_title("(c)", loc="left", fontweight="bold", pad=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_r5_robustness.pdf")
    plt.close(fig)


def fig_r6():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))
    ax = axes[0]
    conds = [(0.5, "_dr_ds0p5_chunks"), (1.0, IN), (1.5, "_dr_ds1p5_chunks"),
             (2.0, "_dr_ds2x_chunks"), (3.0, "_dr_ds3x_chunks")]
    cmap = plt.get_cmap("viridis")
    for i, (m, root) in enumerate(conds):
        x, y = curve(root, "main_dr_d256_b4_r2", RX["upair"], 3)
        ax.plot(x, y, color=cmap(0.12 + 0.19 * i), ls="-", marker="o", ms=2.8,
                label=f"UPAIR-DR, {m}$\\times$")
    bler_axes(ax)
    ax.legend(loc="lower left", fontsize=5.8)
    ax.set_title("(a)", loc="left", fontweight="bold", pad=3)

    ax = axes[1]
    plot_curve(ax, curve("_gen_ds3x_chunks", "main_d256_b4_r2", RX["perfect"], 3), "perfect")
    x, y = curve("_dr_ds3x_chunks", "main_dr_d256_b4_r2", RX["upair"], 3)
    ax.plot(x, y, color=cmap(0.88), ls="-", marker="o", ms=2.8, label="UPAIR-DR")
    plot_curve(ax, curve("_gen_ds3x_chunks", "main_d256_b4_r2", RX["2dl"], 3), "2dl",
               label="2D LMMSE (matched $\\mathbf{R}$)")
    plot_curve(ax, curve("_gen_ds3x_chunks", "main_d256_b4_r2", RX["upair"], 3), "upair",
               label="UPAIR (fixed training)")
    bler_axes(ax)
    ax.legend(loc="lower left", fontsize=6.0)
    ax.set_title("(b)", loc="left", fontweight="bold", pad=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_r6_dr.pdf")
    plt.close(fig)


if __name__ == "__main__":
    ok = 0
    for fn in [fig_r1, fig_r2, fig_r3, fig_r4, fig_r5, fig_r6]:
        try:
            fn()
            print(f"[OK] {fn.__name__}")
            ok += 1
        except Exception:
            print(f"[FAIL] {fn.__name__}:")
            traceback.print_exc(limit=2)
    print(f"\n{ok}/6 figures written to {OUT}/")
    (OUT / "numbers.txt").write_text("\n".join(NUMBERS) + "\n")
    print(f"key numbers saved to {OUT}/numbers.txt")
    sys.exit(0 if ok == 6 else 1)
