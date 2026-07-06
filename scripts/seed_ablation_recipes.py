from __future__ import annotations

"""Freeze main's Stage-B winner as the single shared training recipe for the
mechanism-ablation arms (M0/A1..A8/A3-dagger/A8-dagger).

This deliberately replaces per-arm Optuna Stage A/B: every arm reads the same
recipe JSON, so any performance difference is attributable to the single
architectural change and not to hyperparameter drift. The written JSONs carry
explicit provenance fields.

Typical use (from the repository root, venv active):

    python scripts/seed_ablation_recipes.py \
        --source-prefix clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB \
        --source-variant main_d256_b4_r2 \
        --target-prefix  clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB

Writing under the SAME prefix is safe for the new arm names (they have no
existing JSONs). Existing per-variant JSONs (main/shallow/narrow/...) are
never overwritten unless --overwrite is given.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Kept in sync with RECIPE_FALLBACK in scripts/run_comprehensive_mu32_ablation.py.
# (Not imported: that module imports TensorFlow, which can stall on login nodes.)
DEFAULT_ARMS = [
    "errvar_eval_swap_d256_b4_r2",
    "freq_attn_only_d256_b4_r2",
    "time_attn_only_d256_b4_r2",
    "convnext_axial_d256_b4_r2",
    "oracle_prompt_d256_b4_r2",
    "constant_prompt_d256_b4_r2",
    "no_attn_no_film_d256_b4_r2",
    "main_dr_d256_b4_r2",
    "no_prompt_film_dr_d256_b4_r2",
    "dncnn_trunk_d256_l7",
    "prompt_mean_swap_d256_b4_r2",
    "prompt_wrong_swap_d256_b4_r2",
    "global_pool_prompt_d256_b4_r2",
    "local_only_no_axial_attn_d256_b4_r2",
    "main_full_d256_b4_r2",
    "no_learned_errvar_d256_b4_r2",
    "no_ls_anchor_d256_b4_r2",
    "no_prompt_film_d256_b4_r2",
    "no_raw_y_d256_b4_r2",
    "shallow_light_d256_b2_r2",
    "width_light_d192_b4_r2",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--optuna-dir", default=str(PROJECT_ROOT / "optuna"))
    parser.add_argument("--source-prefix", default="clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB")
    parser.add_argument("--source-variant", default="main_d256_b4_r2")
    parser.add_argument("--target-prefix", default=None, help="Defaults to --source-prefix.")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS), help="Comma-separated arm names to seed.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    optuna_dir = Path(args.optuna_dir)
    target_prefix = args.target_prefix or args.source_prefix
    source_json = optuna_dir / f"{args.source_prefix}_{args.source_variant}_best_params.json"
    if not source_json.exists():
        raise FileNotFoundError(f"Source Stage-B best JSON not found: {source_json}")

    with open(source_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not payload.get("best_params"):
        raise ValueError(f"{source_json} has no best_params.")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    print(f"[SEED-RECIPES] source: {source_json}")
    print(f"[SEED-RECIPES] best_params: {json.dumps(payload['best_params'], sort_keys=True)}")

    written, skipped = [], []
    for arm in arms:
        out = optuna_dir / f"{target_prefix}_{arm}_best_params.json"
        if out.exists() and not args.overwrite:
            skipped.append(out)
            continue
        arm_payload = dict(payload)
        arm_payload["study_name"] = f"{target_prefix}_{arm}"
        arm_payload["recipe_frozen_from"] = {
            "source_study": str(payload.get("study_name", "")),
            "source_json": str(source_json),
            "source_variant": args.source_variant,
            "note": (
                "Shared-recipe ablation protocol: this arm did not run its own "
                "Optuna Stage A/B; it reuses the source variant's Stage-B winner "
                "verbatim so that the single architectural change is the only "
                "difference vs. M0."
            ),
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if args.dry_run:
            print(f"[SEED-RECIPES] would write {out}")
            continue
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(arm_payload, handle, indent=2, sort_keys=True)
        written.append(out)
        print(f"[SEED-RECIPES] wrote {out}")

    for out in skipped:
        print(f"[SEED-RECIPES] SKIPPED existing (use --overwrite to replace): {out}")
    print(f"[SEED-RECIPES] done: wrote {len(written)}, skipped {len(skipped)}, arms requested {len(arms)}")


if __name__ == "__main__":
    main()
