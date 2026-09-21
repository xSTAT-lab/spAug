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
FAMILIES="${FAMILIES:-global_real_plus_synthetic local_spatial}"
GLOBAL_RATIOS="${GLOBAL_RATIOS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}"
LOCAL_RATIOS="${LOCAL_RATIOS:-1 2 3 4}"
RATIOS="${RATIOS:-}"
REAL_FRACTIONS="${REAL_FRACTIONS:-1.0}"
ABLATIONS="${ABLATIONS:-path_B}"
SLICES="${SLICES:-151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676}"
DOWNSTREAM_BACKENDS="${DOWNSTREAM_BACKENDS:-pca_spatial_leiden}"
CLUSTER_SEEDS="${CLUSTER_SEEDS:-42}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_MODELS="${RUN_MODELS:-1}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

SPATIAL_INTERIM="data/02_interim/spatial_cluster/DLPFC"
LABELS="${SPATIAL_INTERIM}/evaluation/labels_for_evaluation.csv"
RESULTS_ROOT="data/05_results"

echo "[DLPFC unsupervised] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[DLPFC unsupervised] models=${MODELS}"
echo "[DLPFC unsupervised] families=${FAMILIES}"
echo "[DLPFC unsupervised] global_ratios=${GLOBAL_RATIOS}; local_ratios=${LOCAL_RATIOS}; real_fractions=${REAL_FRACTIONS}; ablations=${ABLATIONS}"
echo "[DLPFC unsupervised] downstream_backends=${DOWNSTREAM_BACKENDS}; cluster_seeds=${CLUSTER_SEEDS}; skip_existing=${SKIP_EXISTING}"
echo "[DLPFC unsupervised] slices=${SLICES}"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  for SID in ${SLICES}; do
    for ABLATION in ${ABLATIONS}; do
      for BACKEND in ${DOWNSTREAM_BACKENDS}; do
        for CLUSTER_SEED in ${CLUSTER_SEEDS}; do
          OUT_DIR="data/05_results/spatial_cluster/DLPFC/baseline/unsupervised/real_only/${SID}_${ABLATION}_${BACKEND}_seed${CLUSTER_SEED}"
          if [[ "${SKIP_EXISTING}" == "1" && -f "${OUT_DIR}/spatial_clusters.csv" ]]; then
            echo "[DLPFC unsupervised] skip existing baseline ${OUT_DIR}/spatial_clusters.csv"
            continue
          fi
          echo "[DLPFC unsupervised] baseline spatial_cluster slice=${SID} ablation=${ABLATION} backend=${BACKEND} seed=${CLUSTER_SEED}"
          "${PY}" src/05_downstream/spatial_cluster/DLPFC/spagcn_cluster.py \
            -i "${SPATIAL_INTERIM}/model_inputs/SRTsim/unsupervised/processed_with_split.h5ad" \
            -o "${OUT_DIR}" \
            --labels "${LABELS}" \
            -c configs/spatial_cluster/DLPFC/downstream.yaml \
            --slice-id "${SID}" \
            --ablation "${ABLATION}" \
            --backend "${BACKEND}" \
            --cluster-seed "${CLUSTER_SEED}"
        done
      done
    done
  done
fi

if [[ "${RUN_MODELS}" == "1" ]]; then
  for MODEL in ${MODELS}; do
    UNSUP_REAL="${SPATIAL_INTERIM}/model_inputs/${MODEL}/unsupervised/processed_with_split.h5ad"
    UNSUP_SYN="data/03_synthetic/spatial_cluster/DLPFC/${MODEL}/pool_40x/synthetic_pool_40x.h5ad"
    if [[ ! -f "${UNSUP_REAL}" ]]; then
      echo "[DLPFC unsupervised] missing real input: ${UNSUP_REAL}" >&2
      exit 1
    fi
    if [[ ! -f "${UNSUP_SYN}" ]]; then
      echo "[DLPFC unsupervised] missing synthetic pool: ${UNSUP_SYN}" >&2
      exit 1
    fi

    for FAMILY in ${FAMILIES}; do
      FAMILY_RATIOS="${RATIOS}"
      if [[ -z "${FAMILY_RATIOS}" ]]; then
        if [[ "${FAMILY}" == "local_spatial" ]]; then
          FAMILY_RATIOS="${LOCAL_RATIOS}"
        else
          FAMILY_RATIOS="${GLOBAL_RATIOS}"
        fi
      fi
      echo "[DLPFC unsupervised] model=${MODEL} family=${FAMILY}"
      "${PY}" src/05_downstream/spatial_cluster/DLPFC/run_streaming_paradigm.py \
        --real "${UNSUP_REAL}" \
        --synthetic-pool "${UNSUP_SYN}" \
        --family "${FAMILY}" \
        --dataset DLPFC \
        --mode unsupervised \
        --model "${MODEL}" \
        --task spatial_cluster \
        --reference "${UNSUP_REAL}" \
        --labels "${LABELS}" \
        --ratios ${FAMILY_RATIOS} \
        --real-fractions ${REAL_FRACTIONS} \
        --ablations ${ABLATIONS} \
        --slices ${SLICES} \
        --downstream-backends ${DOWNSTREAM_BACKENDS} \
        --cluster-seeds ${CLUSTER_SEEDS} \
        --results-root "${RESULTS_ROOT}" \
        -c configs/spatial_cluster/DLPFC/downstream.yaml \
        --paradigm-config configs/spatial_cluster/DLPFC/paradigm.yaml \
        $(if [[ "${SKIP_EXISTING}" == "1" ]]; then echo "--skip-existing"; fi)
    done
  done
fi

if [[ "${RUN_EVALUATION}" == "1" ]]; then
  echo "[DLPFC unsupervised] evaluate predictions"
  "${PY}" src/06_evaluation/spatial_cluster/DLPFC/evaluate_dlpfc_clustering_extended.py \
    --results-root data/05_results/spatial_cluster/DLPFC \
    -o data/05_results/summary/spatial_cluster/DLPFC \
    --labels "${LABELS}"
  echo "[DLPFC unsupervised] visualize downstream"
  "${PY}" src/06_evaluation/spatial_cluster/DLPFC/visualize_dlpfc_clustering_extended.py \
    --extended-dir data/05_results/summary/spatial_cluster/DLPFC \
    --output-dir data/05_results/figures/spatial_cluster/DLPFC
fi

echo "[DLPFC unsupervised] done"
