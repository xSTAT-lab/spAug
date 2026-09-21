"""DLPFC expression preprocessing for feature augmentation."""
from __future__ import annotations
import logging
import pickle
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from .paths import FEATURE_ROOT as PROJECT_ROOT
logger = logging.getLogger(__name__)


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


class BasePreprocessor(ABC):
    """Base class for dataset preprocessing

    Subclasses must implement: load, qc_filter, normalize, select_hvg, extract_coords, extract_labels
    """

    def __init__(self, dataset_name: str, config: dict):
        self.dataset_name = dataset_name
        self.config = config
        self.raw_dir = PROJECT_ROOT / "data" / dataset_name
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
        """Filter observations and genes by expression thresholds"""
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
        logger.info(f"Saved: {filepath} ({filepath.stat().st_size / 1024 / 1024:.1f} MB)")
        return filepath

    @abstractmethod
    def run(self) -> List[Path]:
        """Run the complete preprocessing pipeline and return generated file paths"""
        pass


class DLPFCPreprocessor(BasePreprocessor):
    """Preprocess the 12-slice DLPFC human dorsolateral prefrontal cortex dataset

    datasource:
        - st/{slice_id}_adata.h5ad (individual section files)
        - st/DLPFC_12_slices.h5ad (combined sections)
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

        load_combined=False: read individual sections from st/
        load_combined=True: read the combined DLPFC_12_slices.h5ad 
        """
        slices = {}

        if not self.load_combined:
            for sid in self.SLICE_IDS:
                filepath = self.raw_dir / "st" / f"{sid}_adata.h5ad"
                if not filepath.exists():
                    logger.warning(f"slice file not found; skipping: {filepath}")
                    continue
                adata = sc.read_h5ad(filepath)
                adata.obs["slice_id"] = sid
                slices[sid] = adata
                logger.info(f"Loaded slice {sid}: {adata.n_obs} spots x {adata.n_vars} genes")
        else:
            filepath = self.raw_dir / "st" / "DLPFC_12_slices.h5ad"
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

        # Write combined expression data
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
            raise RuntimeError("DLPFC global HVG selection is empty; feature preparation requires a nonempty gene selection")
        adata.var["highly_variable"] = hvg
        return adata
