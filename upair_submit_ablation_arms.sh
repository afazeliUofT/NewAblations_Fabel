#!/usr/bin/env bash
# Submit the mechanism-ablation campaign (M0 already trained; arms A1..A8 train
# fresh under main's frozen Stage-B recipe; A8-dagger is eval-only on M0 weights).
#
# Prereqs (once, from the repo root, venv active):
#   python scripts/verify_ablation_param_counts.py          # must print all OK
#   python scripts/seed_ablation_recipes.py                 # freeze shared recipe JSONs
#
# Usage:
#   bash upair_submit_ablation_arms.sh                # all trainable arms + A8-dagger
#   UPAIR_ABLATION_ARMS="no_raw_y_d256_b4_r2" bash upair_submit_ablation_arms.sh
#   UPAIR_ABLATION_SPEED_PROBES=1 bash upair_submit_ablation_arms.sh   # add A5/M0 dual-speed evals
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

TRAINABLE_ARMS_DEFAULT="width_light_d192_b4_r2,shallow_light_d256_b2_r2,no_prompt_film_d256_b4_r2,local_only_no_axial_attn_d256_b4_r2,freq_attn_only_d256_b4_r2,no_raw_y_d256_b4_r2,no_ls_anchor_d256_b4_r2,no_learned_errvar_d256_b4_r2"
EVAL_ONLY_ARMS_DEFAULT="errvar_eval_swap_d256_b4_r2"
# A3-dagger is conditional: submit it explicitly once A3 shows a gap, e.g.
#   UPAIR_ABLATION_ARMS="global_pool_prompt_d256_b4_r2" UPAIR_ABLATION_EVAL_ONLY_ARMS="" bash upair_submit_ablation_arms.sh

ARMS="${UPAIR_ABLATION_ARMS:-${TRAINABLE_ARMS_DEFAULT}}"
EVAL_ONLY_ARMS="${UPAIR_ABLATION_EVAL_ONLY_ARMS:-${EVAL_ONLY_ARMS_DEFAULT}}"

# The shared-recipe prefix (arms' JSONs are seeded by scripts/seed_ablation_recipes.py).
export UPAIR_OPTUNA_STAGEB_PREFIX="${UPAIR_OPTUNA_STAGEB_PREFIX:-clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB}"
# The headline comparison point: 3 scheduled users, full Eb/N0 grid, all four receivers.
export UPAIR_PIPELINE_USERS="${UPAIR_PIPELINE_USERS:-3}"
export UPAIR_PIPELINE_RECEIVERS="${UPAIR_PIPELINE_RECEIVERS:-upair5g_lmmse}"
export UPAIR_PIPELINE_EBNOS="${UPAIR_PIPELINE_EBNOS:--4,-3,-2,-1,0,1}"
export UPAIR_TIME_PIPELINE="${UPAIR_TIME_PIPELINE:-20:00:00}"

# Sanity: shared-recipe JSONs must exist before submitting anything.
missing=0
for arm in ${ARMS//,/ } ${EVAL_ONLY_ARMS//,/ }; do
  [[ -n "${arm}" ]] || continue
  json="${ROOT}/optuna/${UPAIR_OPTUNA_STAGEB_PREFIX}_${arm}_best_params.json"
  if [[ ! -s "${json}" ]]; then
    echo "[ABLAT-SUBMIT] MISSING shared-recipe JSON: ${json}" >&2
    missing=1
  fi
done
if [[ "${missing}" == "1" ]]; then
  echo "[ABLAT-SUBMIT] run: python scripts/seed_ablation_recipes.py   (then retry)" >&2
  exit 2
fi

# A8-dagger needs M0's checkpoint in this working copy.
M0_CKPT="${ROOT}/TWC_plots_comprehensive/runs_rx16/seed${UPAIR_SEED:-7}/1dmrs/main_d256_b4_r2/checkpoints/best.weights.h5"
if [[ -n "${EVAL_ONLY_ARMS}" && ! -s "${M0_CKPT}" ]]; then
  echo "[ABLAT-SUBMIT] MISSING M0 checkpoint for errvar_eval_swap: ${M0_CKPT}" >&2
  exit 3
fi

echo "[ABLAT-SUBMIT] trainable arms: ${ARMS}"
if [[ -n "${ARMS}" ]]; then
  UPAIR_VARIANTS="${ARMS}" bash "${ROOT}/upair_submit_7variant_pipeline.sh"
fi

if [[ -n "${EVAL_ONLY_ARMS}" ]]; then
  echo "[ABLAT-SUBMIT] eval-only arms: ${EVAL_ONLY_ARMS}"
  UPAIR_EVAL_ONLY=1 UPAIR_VARIANTS="${EVAL_ONLY_ARMS}" bash "${ROOT}/upair_submit_7variant_pipeline.sh"
fi

# Optional: A5's physics check -- evaluate at both probe speeds (Doppler sweep).
# Uses the existing probe configs which pin channel.min/max_speed. The gap
# between freq_attn_only and M0 should widen from 8.33 m/s to 16.67 m/s.
if [[ "${UPAIR_ABLATION_SPEED_PROBES:-0}" == "1" ]]; then
  for spd in 8p33 16p67; do
    echo "[ABLAT-SUBMIT] dual-speed eval (speed=${spd}) for M0 + A5"
    UPAIR_EVAL_ONLY=1 \
    UPAIR_CONFIG="${ROOT}/configs/probe3_speed_${spd}.yaml" \
    UPAIR_EVAL_CHUNK_ROOT="${ROOT}/_ablation_speed_${spd}_chunks" \
    UPAIR_VARIANTS="main_d256_b4_r2,freq_attn_only_d256_b4_r2" \
      bash "${ROOT}/upair_submit_7variant_pipeline.sh"
  done
fi

echo "[ABLAT-SUBMIT] done."
