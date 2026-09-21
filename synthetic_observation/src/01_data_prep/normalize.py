"""
src/01_data_prep/normalize.py
=============================
Unified spatial transcriptomics preprocessing.

Supported datasets: DLPFC, Trastuzumab, SpatialGlueTutorial
input: raw data under data/01_raw/
output: normalized AnnData (.h5ad) under data/02_interim/{dataset}/

Standard AnnData output:
    adata.X              -> normalized expression matrix (sparse csr_matrix)
    adata.obs['label']   -> unified label column
    adata.obs['slice_id']-> slice/sample ID
    adata.obsm['spatial']-> (n_spots, 2) NumPy coordinate array
    adata.var['highly_variable'] -> HVG flag
    adata.uns['dataset'] -> dataset name
    adata.uns['task']    -> applicable task list
"""

import os
import sys
import glob
import pickle
import logging
import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
import yaml

# ============================================================
# Logging and helper functions
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("normalize")

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_yaml_config(filename: str) -> dict:
    """Load YAML configuration from configs/."""
    filepath = PROJECT_ROOT / "configs" / filename
    if not filepath.exists():
        logger.warning(f"configuration file does not exist: {filepath}, returning an empty dictionary")
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_sparse(matrix) -> sparse.csr_matrix:
    """Ensure that the matrix is a scipy sparse CSR matrix"""
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(matrix)


def print_adata_summary(adata: ad.AnnData, name: str):
    """Print a summary of the AnnData object"""
    logger.info(f"--- {name} AnnData summary ---")
    logger.info(f"  dimension: {adata.n_obs} spots x {adata.n_vars} genes")
    logger.info(f"  X dtype: {adata.X.dtype}, sparse: {sparse.issparse(adata.X)}")
    logger.info(f"  obs columns: {list(adata.obs.columns)}")
    if "spatial" in adata.obsm:
        coords = adata.obsm["spatial"]
        logger.info(f"  coordinates: shape={coords.shape}, range_x=[{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}], range_y=[{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}]")
    if "label" in adata.obs:
        label_counts = adata.obs["label"].value_counts()
        logger.info(f"  label distribution ({len(label_counts)}  ):")
        for lbl, cnt in label_counts.items():
            logger.info(f"    {lbl}: {cnt} ({cnt/len(adata.obs)*100:.1f}%)")
    if "highly_variable" in adata.var:
        n_hvg = adata.var["highly_variable"].sum()
        logger.info(f"  number of HVGs: {n_hvg}")


# ============================================================
# Base class: BasePreprocessor
# ============================================================

class BasePreprocessor(ABC):
    """dataset process Base class

    Subclasses must implement: load, qc_filter, normalize, select_hvg, extract_coords, extract_labels
    """

    def __init__(self, dataset_name: str, config: dict):
        self.dataset_name = dataset_name
        self.config = config
        self.raw_dir = PROJECT_ROOT / "data" / "01_raw" / dataset_name
        dataset_cfg = config.get("preprocessing", {}).get(dataset_name, {})
        configured_output = dataset_cfg.get("output_dir")
        if configured_output:
            configured_output = Path(configured_output)
            self.output_dir = configured_output if configured_output.is_absolute() else PROJECT_ROOT / configured_output
        else:
            self.output_dir = PROJECT_ROOT / "data" / "02_interim" / dataset_name

    @abstractmethod
    def load(self) -> Any:
        """Load raw data"""
        pass

    @abstractmethod
    def qc_filter(self, adata: ad.AnnData) -> ad.AnnData:
        """Quality-control filtering: removed  spot andlow-expression genes"""
        pass

    @abstractmethod
    def normalize(self, adata: ad.AnnData) -> ad.AnnData:
        """Normalization (some datasets may skip this step)"""
        pass

    @abstractmethod
    def select_hvg(self, adata: ad.AnnData) -> ad.AnnData:
        """Highly variable gene selection"""
        pass

    @abstractmethod
    def extract_coords(self, adata: ad.AnnData) -> np.ndarray:
        """Extract spatial coordinates -> (n_spots, 2)"""
        pass

    @abstractmethod
    def extract_labels(self, adata: ad.AnnData) -> pd.Series:
        """Extract labels"""
        pass

    def save(self, adata: ad.AnnData, filename: str) -> Path:
        """Save AnnData to 02_interim/"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / filename
        adata.write(filepath)
        logger.info(f"alreadysave: {filepath} ({filepath.stat().st_size / 1024 / 1024:.1f} MB)")
        return filepath

    @abstractmethod
    def run(self) -> List[Path]:
        """Run the complete preprocessing pipeline and return generated file paths"""
        pass


# ============================================================
# DLPFC  process 
# ============================================================

class DLPFCPreprocessor(BasePreprocessor):
    """Preprocess the 12-slice DLPFC human dorsolateral prefrontal cortex dataset

    datasource:
        - slices_h5ad/{slice_id}_adata.h5ad (12 independentfile)
        - DLPFC_12_slices.h5ad ( and )
    label: adata.obs['spatialLIBD'] (manual histology annotation, Ground Truth)
    coordinates: adata.obsm['spatial']
    normalization: prefer adata.layers['logcounts'] (already log-normalized)
    """

    SLICE_IDS = [
        "151507", "151508", "151509", "151510",
        "151669", "151670", "151671", "151672",
        "151673", "151674", "151675", "151676",
    ]

    def __init__(self, config: dict):
        super().__init__("DLPFC", config)
        preproc_cfg = config.get("preprocessing", {}).get("DLPFC", {})
        self.min_genes = preproc_cfg.get("min_genes_per_spot", 200)
        self.min_cells = preproc_cfg.get("min_cells_per_gene", 5)
        self.n_top_genes = preproc_cfg.get("n_top_genes", 3000)
        self.use_logcounts = preproc_cfg.get("use_logcounts", True)
        self.enable_hvg = preproc_cfg.get("hvg", True)
        self.load_combined = preproc_cfg.get("load_combined", False)
        self.filter_nan_labels = preproc_cfg.get("filter_nan_labels", True)

    def load(self) -> Dict[str, ad.AnnData]:
        """load DLPFC data

        load_combined=False:  from slices_h5ad/ read ( , memory)
        load_combined=True: directlyread DLPFC_12_slices.h5ad  and 
        """
        slices = {}

        if not self.load_combined:
            for sid in self.SLICE_IDS:
                filepath = self.raw_dir / "slices_h5ad" / f"{sid}_adata.h5ad"
                if not filepath.exists():
                    logger.warning(f"slice file not found; skipping: {filepath}")
                    continue
                adata = sc.read_h5ad(filepath)
                adata.obs["slice_id"] = sid
                slices[sid] = adata
                logger.info(f"Loaded slice {sid}: {adata.n_obs} spots x {adata.n_vars} genes")
        else:
            filepath = self.raw_dir / "DLPFC_12_slices.h5ad"
            logger.info(f"Loaded combined file: {filepath}")
            combined = sc.read_h5ad(filepath)
            # by slice_id  
            if "slice_id" in combined.obs:
                for sid in combined.obs["slice_id"].unique():
                    mask = combined.obs["slice_id"] == sid
                    slices[str(sid)] = combined[mask].copy()
            else:
                # attemptfrom library_id or batch  
                logger.warning("combined data does not contain 'slice_id' column; using the complete object")
                slices["combined"] = combined

        logger.info(f"DLPFC: loaded {len(slices)} slices")
        return slices

    def qc_filter(self, adata: ad.AnnData) -> ad.AnnData:
        """Quality-control filtering"""
        n_before = adata.n_obs
        # gene  QC
        sc.pp.filter_genes(adata, min_cells=self.min_cells)
        # Spot   QC
        sc.pp.filter_cells(adata, min_genes=self.min_genes)
        logger.info(f"  QC: {n_before} -> {adata.n_obs} spots (removed {n_before - adata.n_obs}), "
                     f"{adata.n_vars} genes (min_cells={self.min_cells})")
        return adata

    def normalize(self, adata: ad.AnnData) -> ad.AnnData:
        """normalization: preferuse precomputed logcounts"""
        if self.use_logcounts and "logcounts" in adata.layers:
            logger.info(f"  use precomputed adata.layers['logcounts']")
            adata.X = ensure_sparse(adata.layers["logcounts"])
        else:
            logger.info(f"  run normalize_total + log1p")
            adata.X = ensure_sparse(adata.X)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        return adata

    def select_hvg(self, adata: ad.AnnData) -> ad.AnnData:
        """HVG   (Seurat v3  )"""
        if not self.enable_hvg:
            return adata

        adata_for_hvg = adata.copy()
        # Seurat v3 requires raw counts in .raw in
        if not hasattr(adata_for_hvg, "raw") or adata_for_hvg.raw is None:
            adata_for_hvg.raw = adata.copy()

        try:
            sc.pp.highly_variable_genes(
                adata_for_hvg,
                n_top_genes=min(self.n_top_genes, adata.n_vars - 1),
                flavor="seurat_v3",
            )
        except Exception:
            # Use the Seurat method when seurat_v3 estimation raises an exception.
            logger.warning("seurat_v3 HVG failed,falling back to seurat method")
            sc.pp.highly_variable_genes(
                adata_for_hvg,
                n_top_genes=min(self.n_top_genes, adata.n_vars - 1),
                flavor="seurat",
            )

        adata.var["highly_variable"] = adata_for_hvg.var["highly_variable"]
        logger.info(f"  HVG: {adata.var['highly_variable'].sum()} / {adata.n_vars} genes")
        return adata

    def extract_coords(self, adata: ad.AnnData) -> np.ndarray:
        """Extract spatial coordinates"""
        if "spatial" not in adata.obsm:
            raise ValueError(f"  spatial coordinates are missing from obsm['spatial']")
        return np.array(adata.obsm["spatial"], dtype=np.float64)

    def extract_labels(self, adata: ad.AnnData, slice_id: Optional[str] = None) -> pd.Series:
        """Extract labels: prefer spatialLIBD,falling back to gpfm_update_adj_multi.pkl"""
        if "spatialLIBD" in adata.obs.columns:
            labels = adata.obs["spatialLIBD"].copy()
            labels = labels.astype(str)
            # filtered NaN label
            if self.filter_nan_labels:
                nan_mask = labels.isna() | (labels == "nan") | (labels == "NaN")
                if nan_mask.sum() > 0:
                    logger.info(f"  filtered {nan_mask.sum()} unlabeled spots")
                    labels = labels[~nan_mask]
            return labels
        else:
            # falling back to SpaGCN refined label (only reference)
            pkl_path = self.raw_dir / "gpfm_update_adj_multi.pkl"
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    refined_dict = pickle.load(f)
                if slice_id and slice_id in refined_dict:
                    logger.info(f"  use SpaGCN refined label (non-GT) for {slice_id}")
                    return pd.Series(refined_dict[slice_id], index=adata.obs_names)
            logger.warning(f"  no label information found")
            return pd.Series(["unknown"] * adata.n_obs, index=adata.obs_names)

    def run(self) -> List[Path]:
        """Run DLPFC preprocessing"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        slices = self.load()
        all_processed = []

        for sid, adata in slices.items():
            logger.info(f"\n{'='*50}\nProcessing slice: {sid}\n{'='*50}")
            adata = adata.copy()
            adata = self.qc_filter(adata)
            adata = self.normalize(adata)
            adata = self.select_hvg(adata)

            coords = self.extract_coords(adata)
            labels = self.extract_labels(adata, sid)

            # when labels contain NaN values, filter the AnnData object and coordinates
            if len(labels) < adata.n_obs:
                valid_idx = labels.index
                # create the AnnData mask before filtering coordinates
                coord_mask = adata.obs.index.isin(valid_idx)
                coords = coords[coord_mask]
                adata = adata[valid_idx].copy()

            result = self._build_result(adata, coords, labels, sid)
            print_adata_summary(result, f"DLPFC-{sid}")
            self.save(result, f"{sid}_processed.h5ad")
            all_processed.append(result)
            saved_files.append(self.output_dir / f"{sid}_processed.h5ad")

        # save and 
        if len(all_processed) > 1:
            #  use genes shared across all slices
            common_genes = set(all_processed[0].var_names)
            for a in all_processed[1:]:
                common_genes &= set(a.var_names)
            common_genes = sorted(common_genes)
            logger.info(f"\nintersection of genes across slices: {len(common_genes)} genes")

            aligned = [a[:, common_genes].copy() for a in all_processed]
            combined = ad.concat(aligned, merge="same")
            combined = self._select_global_hvg(combined)
            combined.uns["dataset"] = "DLPFC"
            combined.uns["task"] = ["task1_spot_clf", "task3_spatial_cluster"]
            combined.uns["n_slices"] = len(all_processed)
            combined.uns["slice_ids"] = [a.uns.get("slice_id", "?") for a in all_processed]

            print_adata_summary(combined, "DLPFC-combined")
            self.save(combined, "processed_combined.h5ad")
            saved_files.append(self.output_dir / "processed_combined.h5ad")

        logger.info(f"\n[ok] DLPFC preprocessing completed: {len(saved_files)} output files")
        return saved_files

    def _build_result(self, adata, coords, labels, slice_id):
        """Build the normalized AnnData result object"""
        obs_df = pd.DataFrame(index=adata.obs_names.copy())
        obs_df["slice_id"] = str(slice_id)
        if len(labels) == adata.n_obs:
            obs_df["label"] = labels.values
        else:
            obs_df["label"] = labels.reindex(adata.obs_names).values

        result = ad.AnnData(
            X=ensure_sparse(adata.X),
            obs=obs_df,
            var=adata.var.copy(),
        )
        result.obsm["spatial"] = coords
        result.obs_names = [f"{slice_id}_{name}" for name in result.obs_names.astype(str)]
        result.obs_names_make_unique()
        result.uns["dataset"] = "DLPFC"
        result.uns["task"] = ["task1_spot_clf", "task3_spatial_cluster"]
        result.uns["slice_id"] = slice_id
        return result

    def _select_global_hvg(self, adata: ad.AnnData) -> ad.AnnData:
        """Recompute one global HVG mask after all DLPFC slices are gene-aligned."""
        if not self.enable_hvg:
            adata.var["highly_variable"] = True
            return adata

        n_top = min(self.n_top_genes, max(adata.n_vars - 1, 1))
        adata_for_hvg = adata.copy()

        try:
            sc.pp.highly_variable_genes(
                adata_for_hvg,
                n_top_genes=n_top,
                flavor="seurat_v3",
            )
            hvg = adata_for_hvg.var["highly_variable"].astype(bool).values
            logger.info(f"  global HVG(seurat_v3): {int(hvg.sum())} / {adata.n_vars} genes")
        except Exception as e:
            logger.warning(f"global seurat_v3 HVG failed: {e},falling back to seurat method")
            try:
                adata_for_hvg = adata.copy()
                sc.pp.highly_variable_genes(
                    adata_for_hvg,
                    n_top_genes=n_top,
                    flavor="seurat",
                )
                hvg = adata_for_hvg.var["highly_variable"].astype(bool).values
                logger.info(f"  global HVG(seurat): {int(hvg.sum())} / {adata.n_vars} genes")
            except Exception as e2:
                logger.warning(f"global seurat HVG failed: {e2},falling back tovariance Top-N")
                if sparse.issparse(adata.X):
                    var_per_gene = np.asarray(adata.X.var(axis=0)).ravel()
                else:
                    var_per_gene = np.var(np.asarray(adata.X), axis=0)
                top_idx = np.argsort(var_per_gene)[::-1][:n_top]
                hvg = np.zeros(adata.n_vars, dtype=bool)
                hvg[top_idx] = True
                logger.info(f"  global HVG(variance): {int(hvg.sum())} / {adata.n_vars} genes")

        if not np.any(hvg):
            raise RuntimeError("DLPFC global HVG selection is empty; generators cannot be used")
        adata.var["highly_variable"] = hvg
        return adata


# ============================================================
# Trastuzumab  process 
# ============================================================

class TrastuzumabPreprocessor(BasePreprocessor):
    """Trastuzumab  drug-response data preprocessing

    datasource:
        - Trastuzumab_2_mRNA.csv: 18450 genes x 85 samples (already Z-score  !)
        - Trastuzumab_2_response.csv: 85 samples of label
    label: Response column (Responder / Non-responder,  classification)
    coordinates: image_feature_both.pickle insample 
    normalization: [warning] skip! data is already Z-score standardized
    """

    def __init__(self, config: dict):
        super().__init__("Trastuzumab", config)
        preproc_cfg = config.get("preprocessing", {}).get("Trastuzumab", {})
        self.min_cells_per_gene = preproc_cfg.get("min_cells_per_gene", 3)
        self.min_genes_per_sample = preproc_cfg.get("min_genes_per_sample", 200)
        self.enable_hvg = preproc_cfg.get("hvg", False)  # Z-score datadefault  HVG
        self.n_top_genes = preproc_cfg.get("n_top_genes", 5000)

    def load(self) -> ad.AnnData:
        """Load mRNA and response data"""
        # mRNA: genes(rows) x samples(columns) -> requires as AnnData  
        mrna_path = self.raw_dir / "Trastuzumab_2_mRNA.csv"
        df = pd.read_csv(mrna_path, index_col=0)
        logger.info(f"mRNA raw matrix: {df.shape[0]} genes x {df.shape[1]} samples")

        #  : samples(obs) x genes(vars)
        df_t = df.T
        adata = ad.AnnData(
            X=df_t.values.astype(np.float64),
            obs=pd.DataFrame(index=df_t.index.astype(str)),
            var=pd.DataFrame(index=df_t.columns.astype(str)),
        )
        adata.obs.index.name = "sample_id"
        adata.obs["sample_id"] = adata.obs.index  # create an explicit sample_id column for the split script
        logger.info(f"Transposed AnnData: {adata.n_obs} samples x {adata.n_vars} genes")

        # read Response label
        response_path = self.raw_dir / "Trastuzumab_2_response.csv"
        response_df = pd.read_csv(response_path)
        logger.info(f"Response table: {response_df.shape}, column: {list(response_df.columns)}")

        # alignmentsample
        resp_id_col = "Sample_ID" if "Sample_ID" in response_df.columns else response_df.columns[0]
        resp_label_col = "Response" if "Response" in response_df.columns else response_df.columns[-1]

        response_df[resp_id_col] = response_df[resp_id_col].astype(str)
        adata.obs.index = adata.obs.index.astype(str)

        common_ids = adata.obs.index.intersection(response_df[resp_id_col].values)
        logger.info(f"sample alignment: mRNA {adata.n_obs} intersection Response {len(response_df)} = {len(common_ids)}")

        # byalignmentresultfiltered
        adata = adata[common_ids].copy()
        resp_map = dict(zip(response_df[resp_id_col], response_df[resp_label_col]))
        adata.obs["response"] = adata.obs.index.map(lambda x: resp_map.get(x, "unknown"))

        # load coordinates from image_feature_both.pickle
        self._load_image_coords(adata)

        return adata

    def _load_image_coords(self, adata: ad.AnnData):
        """from image_feature_both.pickle Extract sample-point coordinates

        pickle  : dict[sample_id.svs] -> {'per_patch_features', 'pooled_feature', 'coords'}
        - coords: (n_patches, 2) patch  of coordinates
        - each WSI sample contains one section; aggregate patch coordinates as sample coordinates
        - key   .svs after  (such as 'O09-03495.svs'), with adata of obs_names alignment
        """
        pkl_path = self.raw_dir / "image_feature_both.pickle"
        if not pkl_path.exists():
            logger.warning(f"image feature file does not exist: {pkl_path}")
            adata.obsm["spatial"] = np.zeros((adata.n_obs, 2))
            return

        try:
            with open(pkl_path, "rb") as f:
                img_data = pickle.load(f)

            logger.info(f"  image feature type: {type(img_data).__name__}, keys count: {len(img_data)}")

            if not isinstance(img_data, dict):
                logger.warning(f"  expected a dict, got: {type(img_data)}")
                adata.obsm["spatial"] = np.zeros((adata.n_obs, 2))
                return

            # build of key  : support /  .svs after 
            key_map = {}  # normalized_id -> original_key
            for k in img_data.keys():
                key_map[k] = k                              #   key
                key_map[k.replace(".svs", "")] = k          #   .svs
                key_map[k.replace(".SVS", "")] = k          #   .SVS

            sample_coords = []
            n_matched = 0
            for sid in adata.obs_names:
                orig_key = key_map.get(str(sid))
                if orig_key is not None:
                    entry = img_data[orig_key]
                    coords = self._extract_coords_from_entry(entry)
                    if coords is not None and len(coords) > 0:
                        # WSI  :   patch coordinates by their mean as sample coordinates
                        centroid = coords.mean(axis=0)
                        sample_coords.append(centroid[:2])
                        n_matched += 1
                    else:
                        sample_coords.append(None)
                else:
                    sample_coords.append(None)

            if n_matched == adata.n_obs:
                adata.obsm["spatial"] = np.array(sample_coords)
                logger.info(f"  extract {n_matched}/{adata.n_obs} sample coordinate records (patch level)")
            elif n_matched > 0:
                # partially : use of
                filled = []
                for c in sample_coords:
                    filled.append(c if c is not None else np.zeros(2))
                adata.obsm["spatial"] = np.array(filled)
                logger.warning(f"  {n_matched}/{adata.n_obs} samples with coordinates ")
            else:
                logger.warning(f"  no samples matched coordinates; using zeros")
                logger.info(f"  pickle keys example: {list(img_data.keys())[:3]}")
                logger.info(f"  adata obs_names example: {list(adata.obs_names[:3])}")
                adata.obsm["spatial"] = np.zeros((adata.n_obs, 2))

        except Exception as e:
            logger.warning(f"  failed to load image coordinates: {e}")
            adata.obsm["spatial"] = np.zeros((adata.n_obs, 2))

    @staticmethod
    def _extract_coords_from_entry(entry):
        """Extract a coordinate array from one entry"""
        if isinstance(entry, dict):
            # prefer the configured coordinate key
            for key in ["positions", "coords", "coordinates", "locations", "xy",
                        "spot_positions", "patch_positions"]:
                if key in entry:
                    pos = np.array(entry[key])
                    if pos.ndim == 2 and pos.shape[1] >= 2:
                        return pos[:, :2]
            #  :   2D array ( / )
            skip_keys = {"features", "feature", "embedding", "embeddings", "tensor"}
            for key, val in entry.items():
                if key in skip_keys:
                    continue
                if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] >= 2:
                    return val[:, :2]
        elif isinstance(entry, np.ndarray) and entry.ndim == 2 and entry.shape[1] >= 2:
            return entry[:, :2]
        return None

    def qc_filter(self, adata: ad.AnnData) -> ad.AnnData:
        """Quality-control filtering (Z-score data with relaxed thresholds)"""
        n_genes_before = adata.n_vars
        # filter constant genes from Z-score data using their variance
        if sparse.issparse(adata.X):
            var_per_gene = np.array(adata.X.var(axis=0)).flatten()
        else:
            var_per_gene = np.var(adata.X, axis=0)
        keep_genes = var_per_gene > 1e-8
        adata = adata[:, keep_genes].copy()
        logger.info(f"  QC: removed {n_genes_before - adata.n_vars} constant genes")
        return adata

    def normalize(self, adata: ad.AnnData) -> ad.AnnData:
        """Preserve the supplied Z-score normalization of Trastuzumab mRNA data."""
        logger.info(f"  Skip normalization (data is already Z-score standardized:  mean~{adata.X.mean():.4f}, std~{adata.X.std():.4f})")
        return adata

    def select_hvg(self, adata: ad.AnnData) -> ad.AnnData:
        """Select highly variable genes by variance rank for Z-score data."""
        if not self.enable_hvg:
            adata.var["highly_variable"] = True
            return adata

        # use variance ranking directly for Z-score data
        if sparse.issparse(adata.X):
            var_per_gene = np.array(adata.X.var(axis=0)).flatten()
        else:
            var_per_gene = np.var(adata.X, axis=0)

        top_idx = np.argsort(var_per_gene)[::-1][:self.n_top_genes]
        adata.var["highly_variable"] = False
        adata.var.iloc[top_idx, adata.var.columns.get_loc("highly_variable")] = True
        logger.info(f"  HVG (varianceTop-N): {adata.var['highly_variable'].sum()} genes")
        return adata

    def extract_coords(self, adata: ad.AnnData) -> np.ndarray:
        """extract coordinates already loaded during preprocessing"""
        if "spatial" in adata.obsm:
            return np.array(adata.obsm["spatial"], dtype=np.float64)
        return np.zeros((adata.n_obs, 2), dtype=np.float64)

    def extract_labels(self, adata: ad.AnnData) -> pd.Series:
        """Extract drug-response labels"""
        return adata.obs["response"].copy()

    def run(self) -> List[Path]:
        """run Trastuzumab preprocessing"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        adata = self.load()
        adata = self.qc_filter(adata)
        adata = self.normalize(adata)
        adata = self.select_hvg(adata)

        coords = self.extract_coords(adata)
        labels = self.extract_labels(adata)

        obs_df = pd.DataFrame(index=adata.obs_names.copy())
        obs_df["sample_id"] = adata.obs["sample_id"].astype(str).values
        obs_df["label"] = labels.values
        obs_df["cohort_id"] = "Trastuzumab"

        result = ad.AnnData(
            X=adata.X,
            obs=obs_df,
            var=adata.var.copy(),
        )
        result.obsm["spatial"] = coords
        result.uns["dataset"] = "Trastuzumab"
        result.uns["task"] = ["task2_wsi_pred"]
        result.uns["normalization"] = "z_score"  #  normalization 

        print_adata_summary(result, "Trastuzumab")
        filepath = self.save(result, "processed.h5ad")

        logger.info(f"\n[ok] Trastuzumab preprocessing completed")
        return [filepath]


# ============================================================
# SpatialGlueTutorial  process 
# ============================================================

class SpatialGluePreprocessor(BasePreprocessor):
    """Preprocess SpatialGlueTutorial multi-omic spatial data

    datasource:
        - adata_RNA.h5ad (RNA modality)
        - adata_ADT.h5ad (proteins ADT modality, optional)
        - annotation.csv (spatial  Ground Truth)
    label: annotation.csv inspatial 
    coordinates: adata.obsm['spatial']
    normalization:  independent normalize_total + log1p
    """

    def __init__(self, config: dict):
        super().__init__("SpatialGlueTutorial", config)
        preproc_cfg = config.get("preprocessing", {}).get("SpatialGlueTutorial", {})
        self.min_genes = preproc_cfg.get("min_genes_per_spot", 50)
        self.min_cells = preproc_cfg.get("min_cells_per_gene", 3)
        self.n_top_genes = preproc_cfg.get("n_top_genes", 3000)
        self.enable_hvg = preproc_cfg.get("hvg", True)
        self.use_adt = preproc_cfg.get("use_adt", False)  # is also load the ADT modality

    def load(self) -> ad.AnnData:
        """Load the RNA (+ ADT) modalitydata"""
        rna_path = self.raw_dir / "adata_RNA.h5ad"
        adata_rna = sc.read_h5ad(rna_path)
        adata_rna.var_names_make_unique()  # ensuregene 
        logger.info(f"RNA modality: {adata_rna.n_obs} spots x {adata_rna.n_vars} genes")

        # load 
        annotation_path = self.raw_dir / "annotation.csv"
        if annotation_path.exists():
            ann_df = pd.read_csv(annotation_path)
            logger.info(f" file: {ann_df.shape}, column: {list(ann_df.columns)}")
            self._annotation_df = ann_df
        else:
            self._annotation_df = None
            logger.warning("annotation file does not exist: annotation.csv")

        # optional: load ADT modality
        if self.use_adt:
            adt_path = self.raw_dir / "adata_ADT.h5ad"
            if adt_path.exists():
                adata_adt = sc.read_h5ad(adt_path)
                adata_adt.var_names_make_unique()
                logger.info(f"ADT modality: {adata_adt.n_obs} spots x {adata_adt.n_vars} proteins")
                adata_rna.uns["adt_data"] = adata_adt  #  

        return adata_rna

    def qc_filter(self, adata: ad.AnnData) -> ad.AnnData:
        """Quality-control filtering"""
        n_before = adata.n_obs
        sc.pp.filter_genes(adata, min_cells=self.min_cells)
        sc.pp.filter_cells(adata, min_genes=self.min_genes)
        logger.info(f"  QC: {n_before} -> {adata.n_obs} spots")
        return adata

    def normalize(self, adata: ad.AnnData) -> ad.AnnData:
        """normalization: normalize_total + log1p"""
        adata.layers["counts"] = ensure_sparse(adata.X).copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        #   infinity value (log1p in numbervalue can )
        if sparse.issparse(adata.X):
            data = adata.X.data
        else:
            data = adata.X
        if not np.all(np.isfinite(data)):
            n_bad = np.sum(~np.isfinite(data))
            finite_vals = data[np.isfinite(data)]
            if len(finite_vals) > 0:
                fill_val = np.max(finite_vals)
            else:
                fill_val = 0.0
            data[~np.isfinite(data)] = fill_val
            logger.warning(f"  log1p aftercontains {n_bad} itemsnon-has value,usemaximumvalue {fill_val:.2f}  ")

        # ADT modality-independent normalization
        if "adt_data" in adata.uns:
            adata_adt = adata.uns["adt_data"]
            sc.pp.normalize_total(adata_adt, target_sum=1e4)
            sc.pp.log1p(adata_adt)
            logger.info(f"  ADT modality already normalized")

        return adata

    def select_hvg(self, adata: ad.AnnData) -> ad.AnnData:
        """HVG   (only RNA modality)"""
        if not self.enable_hvg:
            adata.var["highly_variable"] = True
            return adata

        adata_for_hvg = adata.copy()
        if "counts" in adata_for_hvg.layers:
            adata_for_hvg.X = adata_for_hvg.layers["counts"]
            adata_for_hvg.raw = adata_for_hvg.copy()

        try:
            sc.pp.highly_variable_genes(
                adata_for_hvg,
                n_top_genes=min(self.n_top_genes, adata.n_vars - 1),
                flavor="seurat_v3",
            )
        except Exception:
            logger.warning("seurat_v3 HVG failed,falling back to seurat method")
            sc.pp.highly_variable_genes(
                adata_for_hvg,
                n_top_genes=min(self.n_top_genes, adata.n_vars - 1),
                flavor="seurat",
            )

        adata.var["highly_variable"] = adata_for_hvg.var["highly_variable"]
        logger.info(f"  HVG: {adata.var['highly_variable'].sum()} genes")
        return adata

    def extract_coords(self, adata: ad.AnnData) -> np.ndarray:
        """Extract spatial coordinates"""
        if "spatial" in adata.obsm:
            return np.array(adata.obsm["spatial"], dtype=np.float64)
        # try the next available key 
        for key in ["X_spatial", "coords", "spatial_coords"]:
            if key in adata.obsm:
                return np.array(adata.obsm[key], dtype=np.float64)
        logger.warning("not tospatial coordinates")
        return np.zeros((adata.n_obs, 2), dtype=np.float64)

    def extract_labels(self, adata: ad.AnnData) -> pd.Series:
        """extractspatial """
        if self._annotation_df is None:
            return pd.Series(["unknown"] * adata.n_obs, index=adata.obs_names)

        ann = self._annotation_df
        # attempt alignment 
        #  1: annotation of index with adata of obs_names alignment
        ann_index = ann.index if ann.index.name else ann.iloc[:, 0]
        ann_index = ann_index.astype(str)

        #  column 
        label_col = None
        for candidate in ["annotation", "label", "cluster", "cell_type", "domain", "celltype"]:
            if candidate in ann.columns:
                label_col = candidate
                break
        if label_col is None and ann.shape[1] >= 2:
            label_col = ann.columns[-1]
            logger.info(f"  automatically selected annotation column: '{label_col}'")

        if label_col is None:
            logger.warning("  no label column found in the annotation file")
            return pd.Series(["unknown"] * adata.n_obs, index=adata.obs_names)

        # alignment
        ann_map = dict(zip(ann_index.astype(str), ann[label_col].astype(str)))
        labels = adata.obs_names.map(lambda x: ann_map.get(str(x), np.nan))

        n_matched = (~pd.isna(labels)).sum()
        logger.info(f"  label alignment: {n_matched}/{adata.n_obs} spots  ")

        # ifby index alignment ,attemptalign by position
        if n_matched < adata.n_obs * 0.5 and len(ann) == adata.n_obs:
            logger.info(f"  align by position (annotation number == adata number)")
            labels = ann[label_col].astype(str)

        return pd.Series(labels.values, index=adata.obs_names, name="label")

    def run(self) -> List[Path]:
        """run SpatialGlue  process"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        adata = self.load()
        adata = self.qc_filter(adata)
        adata = self.normalize(adata)
        adata = self.select_hvg(adata)

        coords = self.extract_coords(adata)
        labels = self.extract_labels(adata)

        # filter unlabeled spots because clustering evaluation uses ground-truth labels
        valid_mask = ~pd.isna(labels) & (labels != "nan") & (labels != "unknown")
        if valid_mask.sum() < adata.n_obs:
            n_removed = adata.n_obs - valid_mask.sum()
            logger.info(f"filtered {n_removed} unlabeled spots")
            adata = adata[valid_mask].copy()
            coords = coords[valid_mask]
            labels = labels[valid_mask]

        result = ad.AnnData(
            X=ensure_sparse(adata.X),
            obs=adata.obs.copy(),
            var=adata.var.copy(),
        )
        result.obsm["spatial"] = coords
        result.obs["label"] = labels.values
        result.obs["slice_id"] = "SpatialGlue"
        result.uns["dataset"] = "SpatialGlueTutorial"
        result.uns["task"] = ["task3_spatial_cluster"]

        # save the ADT modality when available
        if "adt_data" in adata.uns:
            result.uns["has_adt"] = True
            self.save(adata.uns["adt_data"], "processed_ADT.h5ad")

        print_adata_summary(result, "SpatialGlueTutorial")
        filepath = self.save(result, "processed.h5ad")

        logger.info(f"\n[ok] SpatialGlueTutorial preprocessing completed")
        return [filepath]


# ============================================================
#  main entry point
# ============================================================

PREPROCESSOR_REGISTRY = {
    "DLPFC": DLPFCPreprocessor,
    "Trastuzumab": TrastuzumabPreprocessor,
    "SpatialGlueTutorial": SpatialGluePreprocessor,
}

# dataset defaults
DEFAULT_DATASETS = ["DLPFC", "Trastuzumab"]


def get_preprocessor(dataset_name: str, config: dict) -> BasePreprocessor:
    """getprocess the selected dataset """
    if dataset_name not in PREPROCESSOR_REGISTRY:
        raise ValueError(
            f"unknown dataset: {dataset_name}. "
            f"available datasets: {list(PREPROCESSOR_REGISTRY.keys())}"
        )
    return PREPROCESSOR_REGISTRY[dataset_name](config)


def main(datasets: Optional[List[str]] = None, config_path: Optional[str] = None):
    """
    main entry point: for run preprocessing for datasets 

    Args:
        datasets: datasets to process; None processes all datasets
        config_path:  configuration file path; None uses the default configuration
    """
    # Load configuration
    if config_path:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        # defaultconfiguration ( default dataset parameters)
        config = {"preprocessing": {}}

    available = list(PREPROCESSOR_REGISTRY.keys())
    if datasets is None:
        datasets = DEFAULT_DATASETS
    else:
        for d in datasets:
            if d not in PREPROCESSOR_REGISTRY:
                logger.error(f"unknown dataset: {d}, skip. optional: {available}")

    logger.info(f"\n{'#'*60}")
    logger.info(f"# start process: {datasets}")
    logger.info(f"{'#'*60}\n")

    all_saved_files = {}
    for dataset_name in datasets:
        if dataset_name not in PREPROCESSOR_REGISTRY:
            continue
        logger.info(f"\n{'#'*60}")
        logger.info(f"# dataset: {dataset_name}")
        logger.info(f"{'#'*60}")

        try:
            preprocessor = get_preprocessor(dataset_name, config)
            saved_files = preprocessor.run()
            all_saved_files[dataset_name] = [str(f) for f in saved_files]
        except Exception as e:
            logger.error(f"[error] {dataset_name}  preprocessing failed: {e}", exc_info=True)

    # output 
    logger.info(f"\n{'#'*60}")
    logger.info(f"# preprocessing completed ")
    logger.info(f"{'#'*60}")
    for ds, files in all_saved_files.items():
        logger.info(f"  {ds}: {len(files)} output files")
        for f in files:
            logger.info(f"    -> {f}")

    return all_saved_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified spatial transcriptomics preprocessing")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"datasets to process (optional: {list(PREPROCESSOR_REGISTRY.keys())}), all by default",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="custom YAML configuration path",
    )
    args = parser.parse_args()

    main(datasets=args.datasets, config_path=args.config)
