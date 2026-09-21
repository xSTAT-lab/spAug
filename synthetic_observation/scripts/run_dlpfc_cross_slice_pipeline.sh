#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PY="${PY:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

MODELS="${MODELS:-SRTsim Splatter SPARsim scGAN scDiffusion}"
SPLIT_TAGS="${SPLIT_TAGS:-${SPLIT_TAG:-fixed_8train_4test}}"
CLASSIFIERS="${CLASSIFIERS:-LR XGBoost MLP}"
FAMILIES="${FAMILIES:-real_baseline global_real_plus_synthetic global_synthetic_only local_spatial weighted_erm_package}"
USE_GPU="${USE_GPU:-cuda}"
N_JOBS="${N_JOBS:-4}"
RUN_BASELINE="${RUN_BASELINE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RESULTS_ROOT="${RESULTS_ROOT:-data/05_results}"
RATIO_FILTER="${RATIO_FILTER:-}"
ALPHA_FILTER="${ALPHA_FILTER:-}"

echo "[DLPFC cross-slice] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[DLPFC cross-slice] models=${MODELS}"
echo "[DLPFC cross-slice] split_tags=${SPLIT_TAGS}"
echo "[DLPFC cross-slice] classifiers=${CLASSIFIERS}"
echo "[DLPFC cross-slice] families=${FAMILIES}"
echo "[DLPFC cross-slice] use_gpu=${USE_GPU}; n_jobs=${N_JOBS}"
echo "[DLPFC cross-slice] results_root=${RESULTS_ROOT}; skip_existing=${SKIP_EXISTING}"

EXTRA_ARGS=(--results-root "${RESULTS_ROOT}")
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-existing)
fi
if [[ -n "${RATIO_FILTER// /}" ]]; then
  EXTRA_ARGS+=(--ratio-filter ${RATIO_FILTER})
fi
if [[ -n "${ALPHA_FILTER// /}" ]]; then
  EXTRA_ARGS+=(--alpha-filter ${ALPHA_FILTER})
fi

MODEL_FAMILIES=""
for FAMILY in ${FAMILIES}; do
  if [[ "${FAMILY}" != "real_baseline" ]]; then
    MODEL_FAMILIES="${MODEL_FAMILIES} ${FAMILY}"
  fi
done

for SPLIT_TAG in ${SPLIT_TAGS}; do
  echo "[DLPFC cross-slice] split_tag=${SPLIT_TAG}"
  if [[ "${RUN_BASELINE}" == "1" ]]; then
    echo "[DLPFC cross-slice] baseline split=${SPLIT_TAG}"
    "${PY}" src/05_downstream/cross_slice_generalization/DLPFC/predict.py \
      --models SRTsim \
      --split-tag "${SPLIT_TAG}" \
      --families real_baseline \
      --classifiers ${CLASSIFIERS} \
      --use-gpu "${USE_GPU}" \
      --n-jobs "${N_JOBS}" \
      "${EXTRA_ARGS[@]}"
  fi

  if [[ -n "${MODEL_FAMILIES// /}" ]]; then
    echo "[DLPFC cross-slice] model paradigms:${MODEL_FAMILIES}; split=${SPLIT_TAG}"
    "${PY}" src/05_downstream/cross_slice_generalization/DLPFC/predict.py \
      --models ${MODELS} \
      --split-tag "${SPLIT_TAG}" \
      --families ${MODEL_FAMILIES} \
      --classifiers ${CLASSIFIERS} \
      --skip-baseline \
      --use-gpu "${USE_GPU}" \
      --n-jobs "${N_JOBS}" \
      "${EXTRA_ARGS[@]}"
  fi
done

echo "[DLPFC cross-slice] done"
