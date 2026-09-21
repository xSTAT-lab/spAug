"""Classifier helpers for DLPFC spot-level supervised evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    import cupy as cp
except Exception:
    cp = None

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    from cuml.linear_model import LogisticRegression as CuMLLogisticRegression
except Exception:
    CuMLLogisticRegression = None


def gpu_enabled(params: dict[str, Any]) -> bool:
    """Return whether a classifier should use CUDA when a CUDA backend exists."""
    use_gpu = str(params.get("use_gpu", "auto")).lower()
    if use_gpu in {"0", "false", "no", "cpu"}:
        return False
    if use_gpu in {"1", "true", "yes", "cuda", "gpu"}:
        return True
    return torch is not None and bool(torch.cuda.is_available())


def make_sklearn_lr(params: dict[str, Any], random_seed: int = 42):
    lr_kwargs = {
        "C": float(params.get("C", 1.0)),
        "max_iter": int(params.get("max_iter", 1000)),
        "solver": str(params.get("solver", "lbfgs")),
        "random_state": random_seed,
    }
    multi_class = params.get("multi_class")
    if multi_class not in (None, "", "auto"):
        lr_kwargs["multi_class"] = str(multi_class)
    clf = LogisticRegression(**lr_kwargs)
    return Pipeline([("scale", StandardScaler(with_mean=False)), ("clf", clf)])


class TorchMLPClassifier:
    """Small sklearn-like MLP classifier with torch-native scaling and batches."""

    def __init__(
        self,
        hidden_layers=(256, 128),
        activation: str = "relu",
        dropout: float = 0.3,
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int = 256,
        weight_decay: float = 0.0001,
        early_stopping_patience: int = 10,
        random_seed: int = 42,
        device: str = "cuda",
    ):
        if torch is None or nn is None:
            raise RuntimeError("TorchMLPClassifier requires torch")
        self.hidden_layers = tuple(int(x) for x in hidden_layers)
        self.activation = activation
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.early_stopping_patience = int(early_stopping_patience)
        self.random_seed = int(random_seed)
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model_ = None
        self.classes_ = None
        self.mean_ = None
        self.scale_ = None

    def _build_model(self, n_features: int, n_classes: int):
        layers = []
        last = n_features
        activation = nn.ReLU if self.activation.lower() == "relu" else nn.Tanh
        for hidden in self.hidden_layers:
            layers.append(nn.Linear(last, hidden))
            layers.append(activation())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            last = hidden
        layers.append(nn.Linear(last, n_classes))
        return nn.Sequential(*layers).to(self.device)

    def _fit_transform_tensor(self, x: np.ndarray):
        x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        self.mean_ = x_t.mean(dim=0, keepdim=True)
        self.scale_ = x_t.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        return (x_t - self.mean_) / self.scale_

    def _transform_tensor(self, x: np.ndarray):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("TorchMLPClassifier is not fitted")
        x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        return (x_t - self.mean_) / self.scale_

    def fit(self, x, y, sample_weight=None):
        torch.manual_seed(self.random_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        x_t = self._fit_transform_tensor(x)
        y_t = torch.as_tensor(y_idx.astype(np.int64), device=self.device)
        weights_np = np.ones(y_idx.shape[0], dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        w_t = torch.as_tensor(weights_np, device=self.device)

        n_obs = int(y_t.shape[0])
        perm = torch.randperm(n_obs, device=self.device)
        if n_obs > 4:
            n_val = max(1, int(round(n_obs * 0.15)))
            n_val = min(n_val, n_obs - 1)
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]
        else:
            train_idx = perm
            val_idx = perm

        self.model_ = self._build_model(x_t.shape[1], len(self.classes_))
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        x_val = x_t[val_idx]
        y_val = y_t[val_idx]

        best_state = None
        best_loss = float("inf")
        patience = 0
        batch_size = max(1, min(int(self.batch_size), int(train_idx.shape[0])))
        for _ in range(self.epochs):
            self.model_.train()
            shuffled = train_idx[torch.randperm(train_idx.shape[0], device=self.device)]
            for start in range(0, int(shuffled.shape[0]), batch_size):
                batch_idx = shuffled[start : start + batch_size]
                xb = x_t[batch_idx]
                yb = y_t[batch_idx]
                wb = w_t[batch_idx]
                optimizer.zero_grad(set_to_none=True)
                loss_vec = nn.functional.cross_entropy(self.model_(xb), yb, reduction="none")
                loss = (loss_vec * wb).mean()
                loss.backward()
                optimizer.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = float(nn.functional.cross_entropy(self.model_(x_val), y_val).detach().cpu())
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, x):
        if self.model_ is None:
            raise ValueError("TorchMLPClassifier is not fitted")
        self.model_.eval()
        preds = []
        with torch.no_grad():
            x_np = np.asarray(x, dtype=np.float32)
            for start in range(0, x_np.shape[0], max(1, self.batch_size)):
                xb = self._transform_tensor(x_np[start : start + self.batch_size])
                pred = torch.argmax(self.model_(xb), dim=1).detach().cpu().numpy()
                preds.append(pred)
        return self.classes_[np.concatenate(preds)]


class TorchLinearClassifier(TorchMLPClassifier):
    """GPU multinomial logistic regression implemented as a single linear layer."""

    def __init__(
        self,
        lr: float = 0.01,
        epochs: int = 60,
        batch_size: int = 4096,
        weight_decay: float = 0.0001,
        early_stopping_patience: int = 6,
        random_seed: int = 42,
        device: str = "cuda",
    ):
        super().__init__(
            hidden_layers=(),
            activation="relu",
            dropout=0.0,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            random_seed=random_seed,
            device=device,
        )


class CuMLLRClassifier:
    """sklearn-like wrapper for cuML LogisticRegression."""

    def __init__(self, params: dict[str, Any], random_seed: int = 42):
        if CuMLLogisticRegression is None:
            raise RuntimeError("cuML is not installed")
        self.params = params
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.model = CuMLLogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 1000)),
        )

    def fit(self, x, y, sample_weight=None):
        if sample_weight is not None:
            raise TypeError("cuML LogisticRegression wrapper does not support sample_weight")
        x = self.scaler.fit_transform(np.asarray(x, dtype=np.float32)).astype(np.float32)
        self.model.fit(x, y)
        return self

    def predict(self, x):
        x = self.scaler.transform(np.asarray(x, dtype=np.float32)).astype(np.float32)
        pred = self.model.predict(x)
        return np.asarray(pred)


class CuPyXGBClassifier:
    """XGBoost wrapper that keeps fit/predict arrays on CUDA when CuPy exists."""

    def __init__(self, params: dict[str, Any], random_seed: int = 42):
        if XGBClassifier is None:
            raise RuntimeError("XGBoost is not installed")
        if cp is None:
            raise RuntimeError("CuPy is not installed")
        self.model = XGBClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            eval_metric=str(params.get("eval_metric", "mlogloss")),
            random_state=random_seed,
            n_jobs=int(params.get("n_jobs", 4)),
            tree_method="hist",
            device="cuda",
        )

    def fit(self, x, y, sample_weight=None):
        x_gpu = cp.asarray(np.asarray(x, dtype=np.float32))
        y_gpu = cp.asarray(np.asarray(y))
        weight_gpu = None if sample_weight is None else cp.asarray(np.asarray(sample_weight, dtype=np.float32))
        self.model.fit(x_gpu, y_gpu, sample_weight=weight_gpu)
        return self

    def predict(self, x):
        pred = self.model.predict(cp.asarray(np.asarray(x, dtype=np.float32)))
        return cp.asnumpy(pred)


def make_classifier(name: str, params: dict[str, Any], random_seed: int = 42):
    name_upper = name.upper()
    if name_upper == "LR":
        if gpu_enabled(params) and torch is not None:
            return TorchLinearClassifier(
                lr=float(params.get("lr", 0.01)),
                epochs=int(params.get("epochs", 60)),
                batch_size=int(params.get("batch_size", 4096)),
                weight_decay=float(params.get("weight_decay", 0.0001)),
                early_stopping_patience=int(params.get("early_stopping_patience", 6)),
                random_seed=random_seed,
                device="cuda",
            )
        if gpu_enabled(params) and CuMLLogisticRegression is not None:
            return CuMLLRClassifier(params, random_seed=random_seed)
        return make_sklearn_lr(params, random_seed=random_seed)

    if name_upper == "XGBOOST":
        if XGBClassifier is not None:
            if gpu_enabled(params) and cp is not None:
                return CuPyXGBClassifier(params, random_seed=random_seed)
            xgb_kwargs = {}
            if gpu_enabled(params):
                xgb_kwargs.update({"tree_method": "hist", "device": "cuda"})
            return XGBClassifier(
                n_estimators=int(params.get("n_estimators", 200)),
                max_depth=int(params.get("max_depth", 6)),
                learning_rate=float(params.get("learning_rate", 0.1)),
                subsample=float(params.get("subsample", 0.8)),
                colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                eval_metric=str(params.get("eval_metric", "mlogloss")),
                random_state=random_seed,
                n_jobs=int(params.get("n_jobs", 4)),
                **xgb_kwargs,
            )
        return HistGradientBoostingClassifier(
            max_iter=int(params.get("n_estimators", 200)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_leaf_nodes=31,
            random_state=random_seed,
        )

    if name_upper == "MLP":
        hidden = tuple(int(x) for x in params.get("hidden_layers", [256, 128]))
        if gpu_enabled(params) and torch is not None:
            return TorchMLPClassifier(
                hidden_layers=hidden,
                activation=str(params.get("activation", "relu")),
                dropout=float(params.get("dropout", 0.3)),
                lr=float(params.get("lr", 0.001)),
                epochs=int(params.get("epochs", 100)),
                batch_size=int(params.get("batch_size", 256)),
                weight_decay=float(params.get("weight_decay", 0.0001)),
                early_stopping_patience=int(params.get("early_stopping_patience", 10)),
                random_seed=random_seed,
                device="cuda",
            )
        clf = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=str(params.get("activation", "relu")),
            learning_rate_init=float(params.get("lr", 0.001)),
            max_iter=int(params.get("epochs", 100)),
            batch_size=int(params.get("batch_size", 256)),
            alpha=float(params.get("weight_decay", 0.0001)),
            early_stopping=True,
            n_iter_no_change=int(params.get("early_stopping_patience", 10)),
            random_state=random_seed,
        )
        return Pipeline([("scale", StandardScaler(with_mean=False)), ("clf", clf)])

    raise ValueError(f"Unsupported classifier: {name}")


def train_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    classifier: str,
    params: dict[str, Any],
    random_seed: int = 42,
    sample_weight: np.ndarray | None = None,
):
    model = make_classifier(classifier, params, random_seed=random_seed)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        positive = sample_weight > 0
        if not positive.any():
            raise ValueError("sample_weight has no positive entries")
        x_train = x_train[positive]
        y_train = y_train[positive]
        sample_weight = sample_weight[positive]
        sample_weight = sample_weight / sample_weight.mean()
        if isinstance(model, CuMLLRClassifier):
            model = make_sklearn_lr(params, random_seed=random_seed)

    if sample_weight is None:
        model.fit(x_train, y_train)
    elif isinstance(model, Pipeline):
        try:
            model.fit(x_train, y_train, clf__sample_weight=sample_weight)
        except TypeError as exc:
            raise ValueError(f"{classifier} does not support sample_weight") from exc
    else:
        try:
            model.fit(x_train, y_train, sample_weight=sample_weight)
        except TypeError as exc:
            raise ValueError(f"{classifier} does not support sample_weight") from exc
    return model


def encode_labels(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    encoder.fit(np.concatenate([y_train.astype(str), y_test.astype(str)]))
    return encoder.transform(y_train.astype(str)), encoder.transform(y_test.astype(str)), encoder
