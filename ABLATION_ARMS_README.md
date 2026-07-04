# UPAIR mechanism-ablation arms (M0, A1–A8, A8†, A3†)

Single-change ablations against **M0 = `main_d256_b4_r2`** (the paper table's
`main_full_d256_b4_r2` is a registered alias of it — same architecture, same
checkpoint). Every arm is trained with **main's frozen Stage-B winner** as the
shared recipe (`optuna/clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB_main_d256_b4_r2_best_params.json`);
no arm runs its own Optuna Stage A/B.

| ID  | Variant name                          | Single change (config override)                                | Trainable params |
|-----|---------------------------------------|----------------------------------------------------------------|------------------|
| M0  | `main_d256_b4_r2` (= `main_full_...`) | — reference                                                     | 4,189,376 |
| A1  | `width_light_d192_b4_r2`              | `model.d_model: 192`                                            | 2,380,224 |
| A2  | `shallow_light_d256_b2_r2`            | `model.num_blocks: 2`                                           | 2,206,912 |
| A3  | `no_prompt_film_d256_b4_r2`           | `model.use_prompt_film: false`                                  | 3,531,456 |
| A4  | `local_only_no_axial_attn_d256_b4_r2` | `model.use_freq_attn: false` + `model.use_time_attn: false`     | 2,079,936 |
| A5  | `freq_attn_only_d256_b4_r2`           | `model.use_time_attn: false`                                    | 3,134,656 |
| A6  | `no_raw_y_d256_b4_r2`                 | `model.use_raw_y: false` (channels zeroed, arch untouched)      | 4,189,376 |
| A7  | `no_ls_anchor_d256_b4_r2`             | `model.use_ls_anchor: false` (direct Ĥ, Glorot head)            | 4,189,376 |
| A8  | `no_learned_errvar_d256_b4_r2`        | `model.use_err_head: false` + `training.loss_mode: nmse`        | 4,172,928 |
| A8† | `errvar_eval_swap_d256_b4_r2`         | eval-only: M0 weights + `evaluation.errvar_source: ls`          | 4,189,376 (M0's) |
| A3† | `global_pool_prompt_d256_b4_r2`       | `model.prompt_pool: global` (run only if A3 shows a gap)        | 4,189,376 |

Note on A2: the pre-existing `shallow_d256_b2_r2` was trained with **its own**
Stage-B winner, which differs from main's. `shallow_light_d256_b2_r2` is the
same architecture retrained under the **shared** recipe so that A2 obeys the
single-change protocol. (Comparing the two shallow runs is itself a free
robustness check on the shared-recipe assumption.)

## Workflow (in the clean working copy, venv active)

```bash
# 0) One-time sanity: parameter counts must match the table above.
python scripts/verify_ablation_param_counts.py

# 1) Freeze main's Stage-B winner as the shared recipe for every arm.
python scripts/seed_ablation_recipes.py

# 2) Submit training+eval pipelines for A1..A8 and the eval-only A8†.
bash upair_submit_ablation_arms.sh

# 3) After A1..A8 finish, add the A5 physics check (both probe speeds):
UPAIR_ABLATION_SPEED_PROBES=1 UPAIR_ABLATION_ARMS="" UPAIR_ABLATION_EVAL_ONLY_ARMS="" \
  bash upair_submit_ablation_arms.sh

# 4) Only if A3 shows a gap (>~0.2 dB at BLER 10%): run A3†.
UPAIR_ABLATION_ARMS="global_pool_prompt_d256_b4_r2" UPAIR_ABLATION_EVAL_ONLY_ARMS="" \
  bash upair_submit_ablation_arms.sh
```

Merged per-point results land in `_isolated_eval_chunks/merged_<variant>_u3_<receiver>_e<ebno>.csv`
(and `_ablation_speed_*/` for the dual-speed probes), exactly like the existing
7-variant pipeline.

## Reading the arms

- **A4 vs A1/A2**: A4's deficit beyond the equal-capacity controls isolates the
  global-receptive-field mechanism from raw parameter loss.
- **A8 minus A8†**: A8† isolates the detection-side value of learned
  uncertainty (ĥ literally fixed); the remainder of A8 is the training-side
  effect of the NLL on ĥ quality. Also compare training curves.
- **A7**: report training curves (`metrics/history.json`), not just final
  BLER — the anchor's value may be optimization speed.
- **A5**: evaluate at 8.33 and 16.67 m/s; the gap to M0 should widen with
  Doppler if temporal correlation is exploited.
- **A6**: the headline mechanistic test — gain over 2D-LMMSE that survives A6
  is learned interpolation of LS; gain that disappears is data-aided /
  semi-blind estimation from the raw grid.

## Protocol statement for the paper

All ablation arms reuse the reference model's Stage-B hyperparameter winner
verbatim (learning rate/schedule, weight decay, NMSE weight, gradient clip,
dropout, residual scale, batch size); no per-arm hyperparameter search was
performed. This holds the training recipe fixed so that each arm differs from
M0 by exactly one mechanism, at the cost of a conservative bias: ablated arms
could only improve under per-arm retuning, so reported deficits are lower
bounds on the component's value under this recipe. Recipe parameters that an
arm removes are inert in that arm (`residual_scale` in A7); A8 trains with the
pure NMSE objective since the Gaussian NLL is undefined without the learned
variance.
