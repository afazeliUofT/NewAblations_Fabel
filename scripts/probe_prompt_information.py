from __future__ import annotations

"""P0: what does the learned prompt encode? Zero-training analysis.

Loads M0's frozen checkpoint, sweeps (Eb/N0, user count, pinned speed),
captures the post-MLP prompt vector for every slot, then fits closed-form
ridge probes prompt -> {Eb/N0, U, speed} with *held-out config cells* and
saves R^2 + 2-D PCA scatters.

Run inside the venv on a GPU node (eager mode, ~20-40 min):
    python scripts/probe_prompt_information.py
"""

import argparse, itertools, sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for q in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(q) not in sys.path:
        sys.path.insert(0, str(q))

from upair5g.builders import (  # noqa: E402
    build_channel, build_ls_estimator, build_pusch_transmitter,
    extract_true_dmrs_mask_per_stream, get_resource_grid,
)
from upair5g.config import load_config, set_cfg  # noqa: E402
from upair5g.estimator import UPAIRChannelEstimator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "twc_comprehensive_mu32_base.yaml"))
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / "TWC_plots_comprehensive/runs_rx16/seed7/1dmrs/main_d256_b4_r2/checkpoints/best.weights.h5"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "prompt_probe_out"))
    ap.add_argument("--ebnos", default="-4,-2,0,2,4")
    ap.add_argument("--users", default="1,2,3,4")
    ap.add_argument("--speeds", default="8.33,16.67")
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    import tensorflow as tf
    from upair5g.evaluation import _make_eval_batch

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ebnos = [float(x) for x in args.ebnos.split(",")]
    users = [int(x) for x in args.users.split(",")]
    speeds = [float(x) for x in args.speeds.split(",")]

    P, L = [], []  # prompts, labels (ebno, U, speed)
    for U in users:
        base = load_config(args.config)
        set_cfg(base, "system.batch_size_eval", args.batch_size)
        set_cfg(base, "system.graph_mode", False)
        set_cfg(base, "multiuser.fixed_num_users", U)
        for spd in speeds:
            cfg = load_config(args.config)
            for k, vv in base.items():
                cfg[k] = vv
            for key in [k for k in cfg.get("channel", {}) if "speed" in k]:
                set_cfg(cfg, f"channel.{key}", spd)
            tx, _ = build_pusch_transmitter(cfg, num_users=U)
            channel = build_channel(cfg, tx)
            rg = get_resource_grid(tx)
            pm = extract_true_dmrs_mask_per_stream(tx, rg)
            ls = build_ls_estimator(tx, cfg, interpolation_type="lin")
            est = UPAIRChannelEstimator(ls_estimator=ls, resource_grid=rg, cfg=cfg, pilot_mask=pm)
            warm = _make_eval_batch(tx=tx, channel=channel, cfg=cfg, batch_size=2, ebno_db=0.0)
            est.estimate_with_ls(warm["y"], warm["no"], training=False)
            est.load_weights(args.checkpoint)

            captured: list[np.ndarray] = []
            orig_mlp = est.prompt_mlp

            def hooked(x, _orig=orig_mlp, _cap=captured):
                y = _orig(x)
                _cap.append(np.asarray(y))
                return y

            est.prompt_mlp = hooked
            for e in ebnos:
                for _ in range(args.batches):
                    captured.clear()
                    b = _make_eval_batch(tx=tx, channel=channel, cfg=cfg, batch_size=args.batch_size, ebno_db=e)
                    est.estimate_with_ls(b["y"], b["no"], training=False)
                    pr = captured[-1]
                    P.append(pr)
                    L.append(np.tile([e, U, spd], (pr.shape[0], 1)))
            est.prompt_mlp = orig_mlp
            del est, tx, channel, ls
            tf.keras.backend.clear_session()
            print(f"[P0] U={U} speed={spd}: collected", flush=True)

    X = np.concatenate(P, 0).astype(np.float64)
    Y = np.concatenate(L, 0)
    np.savez_compressed(out / "prompts.npz", prompts=X, labels=Y)

    # held-out-cell ridge probes
    cells = np.unique(Y, axis=0)
    rng = np.random.default_rng(0); rng.shuffle(cells)
    test_cells = {tuple(c) for c in cells[: max(1, len(cells) // 4)]}
    is_test = np.array([tuple(r) in test_cells for r in Y])
    Xc = X - X[~is_test].mean(0); Xc /= (X[~is_test].std(0) + 1e-9)
    A = np.hstack([Xc, np.ones((len(Xc), 1))])
    lam = 1e-2 * np.eye(A.shape[1]); lam[-1, -1] = 0.0
    rows = []
    for j, name in enumerate(["ebno_db", "num_users", "speed"]):
        t = Y[:, j].astype(np.float64)
        w = np.linalg.solve(A[~is_test].T @ A[~is_test] + lam, A[~is_test].T @ t[~is_test])
        pred = A[is_test] @ w
        ss = 1.0 - np.sum((t[is_test] - pred) ** 2) / np.sum((t[is_test] - t[is_test].mean()) ** 2)
        rows.append((name, ss))
        print(f"[P0] linear probe R^2 ({name}, held-out cells) = {ss:.3f}")
    with open(out / "probe_r2.csv", "w") as fh:
        fh.write("target,r2_heldout\n")
        for n, r2 in rows:
            fh.write(f"{n},{r2:.4f}\n")

    # 2-D PCA scatters
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Z = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    pc = Z @ Vt[:2].T
    for j, (name, cmap) in enumerate([("ebno_db", "viridis"), ("num_users", "tab10"), ("speed", "coolwarm")]):
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        sc = ax.scatter(pc[:, 0], pc[:, 1], c=Y[:, j], cmap=cmap, s=6, alpha=.7)
        fig.colorbar(sc, ax=ax, label=name)
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        fig.tight_layout(); fig.savefig(out / f"pca_by_{name}.png", dpi=300)
        plt.close(fig)
    print(f"[P0] wrote {out}/prompts.npz, probe_r2.csv, pca_by_*.png")


if __name__ == "__main__":
    main()
