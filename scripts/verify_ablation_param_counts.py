from __future__ import annotations

"""Instantiate every ablation arm once (CPU is fine) and assert that the
trainable-parameter count matches the paper table exactly. Run this inside the
project venv BEFORE submitting any training jobs:

    python scripts/verify_ablation_param_counts.py \
        --config configs/twc_comprehensive_mu32_base.yaml

Exit code 0 iff all arms match.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from upair5g.builders import (  # noqa: E402
    build_channel,
    build_ls_estimator,
    build_pusch_transmitter,
    extract_true_dmrs_mask_per_stream,
    get_resource_grid,
)
from upair5g.config import load_config, set_cfg  # noqa: E402
from upair5g.estimator import UPAIRChannelEstimator  # noqa: E402
from scripts.run_comprehensive_mu32_ablation import VARIANTS, _variant_cfg  # noqa: E402

# variant name -> (ablation ID, expected trainable params from the paper table)
EXPECTED: dict[str, tuple[str, int]] = {
    "main_d256_b4_r2": ("M0", 4_189_376),
    "main_full_d256_b4_r2": ("M0(alias)", 4_189_376),
    "width_light_d192_b4_r2": ("A1", 2_380_224),
    "shallow_light_d256_b2_r2": ("A2", 2_206_912),
    "no_prompt_film_d256_b4_r2": ("A3", 3_531_456),
    "local_only_no_axial_attn_d256_b4_r2": ("A4", 2_079_936),
    "freq_attn_only_d256_b4_r2": ("A5", 3_134_656),
    "time_attn_only_d256_b4_r2": ("A5b", 3_134_656),
    "convnext_axial_d256_b4_r2": ("S1", 4_208_832),
    "oracle_prompt_d256_b4_r2": ("P2o", 4_190_912),
    "constant_prompt_d256_b4_r2": ("P2c", 4_189_376),
    "no_attn_no_film_d256_b4_r2": ("C1", 1_422_016),
    "dncnn_trunk_d256_l7": ("C2", 4_227_008),
    "prompt_mean_swap_d256_b4_r2": ("P3m", 4_189_376),
    "prompt_wrong_swap_d256_b4_r2": ("P3w", 4_189_376),
    "no_raw_y_d256_b4_r2": ("A6", 4_189_376),
    "no_ls_anchor_d256_b4_r2": ("A7", 4_189_376),
    "no_learned_errvar_d256_b4_r2": ("A8", 4_172_928),
    "errvar_eval_swap_d256_b4_r2": ("A8+", 4_189_376),
    "global_pool_prompt_d256_b4_r2": ("A3+", 4_189_376),
}


def count_params(cfg) -> int:
    import tensorflow as tf

    tx, _ = build_pusch_transmitter(cfg, num_users=3)
    channel = build_channel(cfg, tx)
    resource_grid = get_resource_grid(tx)
    pilot_mask = extract_true_dmrs_mask_per_stream(tx, resource_grid)
    ls_estimator = build_ls_estimator(tx, cfg, interpolation_type="lin")
    estimator = UPAIRChannelEstimator(
        ls_estimator=ls_estimator, resource_grid=resource_grid, cfg=cfg, pilot_mask=pilot_mask
    )
    # One tiny forward pass to build all variables.
    batch_size = 2
    from upair5g.evaluation import _make_eval_batch  # local import: heavy module

    batch = _make_eval_batch(tx=tx, channel=channel, cfg=cfg, batch_size=batch_size, ebno_db=0.0)
    estimator.estimate_with_ls(batch["y"], batch["no"], training=False)
    n = int(np.sum([np.prod(v.shape) for v in estimator.trainable_variables]))
    del estimator, tx, channel, ls_estimator, batch
    tf.keras.backend.clear_session()
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "twc_comprehensive_mu32_base.yaml"))
    parser.add_argument("--arms", default=",".join(EXPECTED.keys()))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    # Keep instantiation light and CPU-friendly.
    set_cfg(base_cfg, "system.batch_size_eval", 2)
    set_cfg(base_cfg, "system.graph_mode", False)

    failures = 0
    for variant in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if variant not in VARIANTS:
            print(f"[PARAMS] SKIP unknown variant {variant}")
            continue
        expected_id, expected_n = EXPECTED[variant]
        cfg = _variant_cfg(base_cfg, variant, "1dmrs", args.seed)
        set_cfg(cfg, "multiuser.fixed_num_users", 3)
        n = count_params(cfg)
        ok = n == expected_n
        failures += 0 if ok else 1
        print(f"[PARAMS] {expected_id:9s} {variant:38s} counted={n:>9,d} expected={expected_n:>9,d} {'OK' if ok else 'MISMATCH'}")
    if failures:
        print(f"[PARAMS] {failures} mismatch(es).")
        sys.exit(1)
    print("[PARAMS] all arms match the paper table.")


if __name__ == "__main__":
    main()
