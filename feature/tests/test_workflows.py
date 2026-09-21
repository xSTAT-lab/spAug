"""Exercise feature-local numerical workflows with synthetic AnnData fixtures."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import anndata as ad
import numpy as np
import pandas as pd
from spaug_feature import low_label, cross_slice, spatial, sample
from spaug_feature.classifiers import make_classifier, train_classifier
from spaug_feature.alignment import align_observations, match_sample_barcodes
from spaug_feature.splits import make_low_label_split
from spaug_feature.preprocessing import DLPFCPreprocessor


def fixture():
    rng = np.random.default_rng(42)
    data = ad.AnnData(
        rng.poisson(4, size=(32, 12)).astype(np.float32),
        obs=pd.DataFrame({"slice_id": ["a"] * 16 + ["b"] * 16,
                          "sample_id": ["a"] * 16 + ["b"] * 16,
                          "label": ["A", "B"] * 16}, index=[f"spot{i}" for i in range(32)]),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(12)]),
    )
    data.obsm["spatial"] = rng.normal(size=(32, 2))
    return data


class FeatureWorkflowTest(unittest.TestCase):
    def test_split_projection_and_classifier(self):
        split, report = make_low_label_split(fixture(), 0.5, 42)
        train = split[split.obs["split"] == "train"].copy()
        test = split[split.obs["split"] == "test"].copy()
        self.assertFalse(set(train.obs_names) & set(test.obs_names))
        self.assertEqual(int(report.n_train.sum()), train.n_obs)
        for module in (low_label, cross_slice):
            x_train, x_test = module.transform_features(train, test, pca_fit=train, n_components=4, use_gpu="cpu")
            self.assertEqual(x_train.shape, (train.n_obs, 4))
            self.assertEqual(x_test.shape, (test.n_obs, 4))
            self.assertTrue(np.isfinite(x_test).all())
            clf = make_classifier("LR", module.classifier_params("LR", "cpu", 1), random_seed=42)
            clf.fit(x_train, train.obs.label)
            self.assertEqual(clf.predict(x_test).shape, (test.n_obs,))

    def test_graph_and_result_output(self):
        data = fixture()
        features = spatial.build_features(data, n_components=4)
        graph = spatial.build_backend_graph(features, data.obsm["spatial"], "pca_spatial_leiden", 3)
        self.assertEqual(graph.shape, (32, 32))
        self.assertEqual((graph != graph.T).nnz, 0)
        labels = spatial.cluster_with_graph_leiden(graph, 1.0, 42, features)
        with tempfile.TemporaryDirectory() as temp:
            table = spatial.write_spatial_cluster_result(data, labels, temp, {}, "a", "path_B", "pca_spatial_leiden", 42)
            self.assertEqual(len(table), data.n_obs)
            self.assertTrue((Path(temp) / "spatial_clusters.csv").is_file())

    def test_sample_summary(self):
        data = fixture()
        pca, genes = sample.fit_pca(data, ["a"], 3)
        subset = data[data.obs.sample_id == "b"].copy()
        weights = np.full(subset.n_obs, 1 / subset.n_obs)
        vector = sample.sample_feature(subset, weights, pca, genes, [0.1, 0.5, 0.9])
        self.assertEqual(vector.shape, (21,))
        self.assertTrue(np.isfinite(vector).all())

    def test_embedding_order_and_filtering(self):
        values = np.array([[10, 11], [20, 21], [30, 31]])
        names = ["section_a", "section_b", "section_c"]
        np.testing.assert_array_equal(
            align_observations(values, names, ["section_c", "section_a", "section_b"]),
            values[[2, 0, 1]],
        )
        np.testing.assert_array_equal(align_observations(values, names, ["section_b"]), values[[1]])
        with self.assertRaises(ValueError):
            align_observations(values, names, ["section_missing"])
        with self.assertRaises(ValueError):
            align_observations(values, ["a", "a", "b"], ["a"])

    def test_sample_barcode_alignment(self):
        for raw in (["a", "b"], ["sample:a", "sample:b"]):
            indices, names = match_sample_barcodes(raw, "sample", ["b", "a", "extra"])
            self.assertEqual(indices.tolist(), [0, 1])
            self.assertEqual(names, [raw[1], raw[0]])
            expression = pd.Series([1, 2], index=raw)
            self.assertEqual(expression.loc[names].tolist(), [2, 1])

    def test_sample_cache_is_scoped_to_dataset(self):
        first = fixture()
        second = fixture()
        second.X = second.X + 100
        a = sample.subset_sample(first, "a")
        first_pca, _ = sample.get_fold_pca(first, 0, ["a"], 3)
        b = sample.subset_sample(second, "a")
        second_pca, _ = sample.get_fold_pca(second, 0, ["a"], 3)
        np.testing.assert_allclose(b.X - a.X, 100)
        np.testing.assert_allclose(second_pca.mean_ - first_pca.mean_, 100)
        self.assertIsNot(first_pca, second_pca)

    def test_visium_patch_pixel_order_and_padding(self):
        import importlib.util
        import pickle
        from PIL import Image
        spec = importlib.util.spec_from_file_location("feature_extract", ROOT / "scripts/extract_pfm_embeddings.py")
        extractor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extractor)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for sub in ("st", "wsis", "pilot/section"):
                (folder / sub).mkdir(parents=True)
            data = ad.AnnData(np.ones((3, 2), dtype=np.float32), obs=pd.DataFrame(
                {"sample_id": ["section"] * 3}, index=["center", "edge", "outside"]))
            data.write_h5ad(folder / "st/DLPFC_12_slices.h5ad")
            pixels = np.arange(4 * 7 * 3, dtype=np.uint8).reshape(4, 7, 3)
            Image.fromarray(pixels).save(folder / "wsis/section_full_image.tif")
            (folder / "pilot/section/tissue_positions_list.txt").write_text(
                "center,1,0,0,1,5\nedge,1,0,0,0,0\noutside,0,0,0,99,99\n")
            extractor.crop_dlpfc_patches(folder, radius=1)
            with (folder / "pkl/patches/visium_section_1.pkl").open("rb") as handle:
                patches = pickle.load(handle)
            np.testing.assert_array_equal(patches[0][:, :, 0, :], pixels[0:3, 4:7])
            np.testing.assert_array_equal(patches[1][1, 1, 0], pixels[0, 0])
            self.assertEqual(int(patches[1][0].sum()), 0)
            self.assertEqual(int(patches[2].sum()), 0)

    def test_dlpfc_preparation(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "st").mkdir()
            data = fixture()
            for sid in ("a", "b"):
                subset = data[data.obs.slice_id == sid].copy()
                subset.obs["spatialLIBD"] = subset.obs.label
                subset.write_h5ad(folder / "st" / f"{sid}_adata.h5ad")
            prep = DLPFCPreprocessor({"preprocessing": {"DLPFC": {
                "min_genes_per_spot": 1, "min_cells_per_gene": 1,
                "hvg": False, "use_logcounts": False,
            }}})
            prep.raw_dir = folder
            prep.output_dir = folder / "processed"
            prep.SLICE_IDS = ["a", "b"]
            prep.run()
            combined = ad.read_h5ad(prep.output_dir / "processed_combined.h5ad")
            self.assertEqual(combined.shape, data.shape)
            self.assertEqual(set(combined.obs.slice_id), {"a", "b"})
            self.assertTrue(all(name.startswith(("a_", "b_")) for name in combined.obs_names))
            self.assertTrue(np.isfinite(combined.X.data).all())


if __name__ == "__main__":
    unittest.main()
