#!/usr/bin/env bash
#SBATCH --job-name=upairP-oracle_prompt
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/rsadve1/links/scratch/Ablations_S1P2/logs/pipeline/pipeline_oracle_prompt_d256_b4_r2_%j.out
#SBATCH --gres=gpu:h100:1
set -euo pipefail
cd "/home/rsadve1/links/scratch/Ablations_S1P2"
source "/home/rsadve1/links/scratch/Ablations_S1P2/upair_portable_env.sh"
upair_activate

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

# Carry pipeline scope if the submitter set these once before calling this wrapper.
export UPAIR_CONFIG=""
export UPAIR_DMRS_CASE="1dmrs"
export UPAIR_SEED="7"
export UPAIR_OPTUNA_STAGEB_PREFIX="clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB"
export UPAIR_PIPELINE_RECEIVERS="upair5g_lmmse"
export UPAIR_PIPELINE_USERS="3"
export UPAIR_PIPELINE_EBNOS="-4,-3,-2,-1,0,1"
export UPAIR_PIPELINE_CHUNK_BATCHES="20"
export UPAIR_PIPELINE_MICRO="8"
export UPAIR_PIPELINE_TARGET_BLOCK_ERRORS=""
export UPAIR_PIPELINE_MAX_BATCHES="600"
export UPAIR_PIPELINE_MIN_BATCHES=""
export UPAIR_EVAL_ONLY="0"
export UPAIR_EVAL_CHUNK_ROOT="_isolated_eval_chunks"

bash "/home/rsadve1/links/scratch/Ablations_S1P2/upair_variant_pipeline_worker.sh" "oracle_prompt_d256_b4_r2"
