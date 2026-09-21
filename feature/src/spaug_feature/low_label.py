"""PCA and classifier configuration for feature-based prediction."""
from __future__ import annotations
import sys
import numpy as np
import anndata as ad
from scipy import sparse
from sklearn.decomposition import PCA
try:
    import torch
except ImportError:
    torch = None

def dense_array(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def classifier_params(classifier: str, use_gpu: str, n_jobs: int) -> dict:
    name = classifier.upper()
    if name == "LR":
        return {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "lr": 0.01,
            "epochs": 60,
            "batch_size": 4096,
            "weight_decay": 0.0001,
            "early_stopping_patience": 6,
            "use_gpu": use_gpu,
        }
    if name == "XGBOOST":
        return {
            "n_estimators": 160,
            "max_depth": 5,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "n_jobs": n_jobs,
            "use_gpu": use_gpu,
        }
    if name == "MLP":
        return {
            "hidden_layers": [256, 128],
            "dropout": 0.25,
            "lr": 0.001,
            "epochs": 50,
            "batch_size": 4096,
            "weight_decay": 0.0001,
            "early_stopping_patience": 5,
            "use_gpu": use_gpu,
        }
    raise ValueError(f"Unsupported classifier: {classifier}")


def choose_pca_components(n_train: int, n_features: int, requested: int) -> int:
    return max(1, min(int(requested), n_features, n_train - 1))


def gpu_enabled_flag(use_gpu: str) -> bool:
    value = str(use_gpu).lower()
    if value in {"0", "false", "no", "cpu"}:
        return False
    if torch is None or not torch.cuda.is_available():
        return False
    return value in {"1", "true", "yes", "cuda", "gpu", "auto"}


def project_gpu_pca(
    x_fit: np.ndarray,
    x_train: np.ndarray,
    x_test: np.ndarray,
    n_components: int,
    use_gpu: str,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not gpu_enabled_flag(use_gpu):
        return None
    try:
        device = torch.device("cuda")
        with torch.no_grad():
            x_fit_t = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
            mean = x_fit_t.mean(dim=0, keepdim=True)
            centered = x_fit_t - mean
            denom = max(1, int(x_fit_t.shape[0]) - 1)
            cov = centered.transpose(0, 1).matmul(centered) / float(denom)
            del centered, x_fit_t
            eigvals, eigvecs = torch.linalg.eigh(cov)
            order = torch.argsort(eigvals, descending=True)[:n_components]
            components = eigvecs[:, order].contiguous()
            del eigvals, eigvecs, cov

            def transform(x: np.ndarray) -> np.ndarray:
                parts = []
                for start in range(0, x.shape[0], chunk_size):
                    xb = torch.as_tensor(x[start : start + chunk_size], dtype=torch.float32, device=device)
                    z = (xb - mean).matmul(components)
                    parts.append(z.detach().cpu().numpy().astype(np.float32, copy=False))
                return np.concatenate(parts, axis=0)

            out_train = transform(x_train)
            out_test = transform(x_test)
        torch.cuda.empty_cache()
        return out_train, out_test
    except Exception as exc:
        print(f"[GPU PCA fallback] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


def transform_features(
    train: ad.AnnData,
    test: ad.AnnData,
    pca_fit: ad.AnnData,
    n_components: int,
    use_gpu: str,
) -> tuple[np.ndarray, np.ndarray]:
    x_fit = dense_array(pca_fit.X).astype(np.float32)
    x_train = dense_array(train.X).astype(np.float32)
    x_test = dense_array(test.X).astype(np.float32)
    n_comp = choose_pca_components(x_fit.shape[0], x_fit.shape[1], n_components)
    if n_comp < 2:
        return x_train, x_test
    gpu_result = project_gpu_pca(x_fit, x_train, x_test, n_comp, use_gpu=use_gpu)
    if gpu_result is not None:
        return gpu_result
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(x_fit)
    return pca.transform(x_train).astype(np.float32), pca.transform(x_test).astype(np.float32)
