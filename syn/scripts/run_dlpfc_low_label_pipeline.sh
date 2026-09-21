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
SEEDS="${SEEDS:-42 43 44}"
LABEL_FRACTIONS="${LABEL_FRACTIONS:-0.25 0.50 0.75}"
CLASSIFIERS="${CLASSIFIERS:-LR XGBoost MLP}"
FAMILIES="${FAMILIES:-real_baseline global_real_plus_synthetic global_synthetic_only local_spatial weighted_erm_package}"
USE_GPU="${USE_GPU:-cuda}"
N_JOBS="${N_JOBS:-4}"
RUN_BASELINE="${RUN_BASELINE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RESULTS_ROOT="${RESULTS_ROOT:-data/05_results}"
RATIO_FILTER="${RATIO_FILTER:-}"
ALPHA_FILTER="${ALPHA_FILTER:-}"

echo "[DLPFC low-label] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[DLPFC low-label] models=${MODELS}"
echo "[DLPFC low-label] seeds=${SEEDS}"
echo "[DLPFC low-label] label_fractions=${LABEL_FRACTIONS}"
echo "[DLPFC low-label] classifiers=${CLASSIFIERS}"
echo "[DLPFC low-label] families=${FAMILIES}"
echo "[DLPFC low-label] use_gpu=${USE_GPU}; n_jobs=${N_JOBS}"
echo "[DLPFC low-label] results_root=${RESULTS_ROOT}; skip_existing=${SKIP_EXISTING}"

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

for LABEL_FRACTION in ${LABEL_FRACTIONS}; do
  echo "[DLPFC low-label] label_fraction=${LABEL_FRACTION}"
  if [[ "${RUN_BASELINE}" == "1" ]]; then
    echo "[DLPFC low-label] baseline fraction=${LABEL_FRACTION}"
    "${PY}" src/05_downstream/supervised_low_label/DLPFC/predict.py \
      --models SRTsim \
      --seeds ${SEEDS} \
      --label-fraction "${LABEL_FRACTION}" \
      --families real_baseline \
      --classifiers ${CLASSIFIERS} \
      --use-gpu "${USE_GPU}" \
      --n-jobs "${N_JOBS}" \
      "${EXTRA_ARGS[@]}"
  fi

  if [[ -n "${MODEL_FAMILIES// /}" ]]; then
    echo "[DLPFC low-label] model paradigms:${MODEL_FAMILIES}; fraction=${LABEL_FRACTION}"
    "${PY}" src/05_downstream/supervised_low_label/DLPFC/predict.py \
      --models ${MODELS} \
      --seeds ${SEEDS} \
      --label-fraction "${LABEL_FRACTION}" \
      --families ${MODEL_FAMILIES} \
      --classifiers ${CLASSIFIERS} \
      --skip-baseline \
      --use-gpu "${USE_GPU}" \
      --n-jobs "${N_JOBS}" \
      "${EXTRA_ARGS[@]}"
  fi
done

echo "[DLPFC low-label] done"
