"""Takagi-Sugeno Order-1 Fuzzy Inference System with LSE and Hybrid Learning (LSE + GD)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class TakagiSugenoHybrid:
    """ANFIS-style Takagi-Sugeno Order-1 FIS initialized by FCM and tuned via Hybrid Learning."""

    def __init__(
        self,
        n_clusters: int,
        n_features: int,
        ridge_lambda: float = 1.0,
        sigma_min: float = 0.25,
        sigma_max: float = 5.0,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.n_features = int(n_features)
        self.ridge_lambda = float(ridge_lambda)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        self.init_centers_: np.ndarray | None = None
        self.init_sigmas_: np.ndarray | None = None
        self.centers_: np.ndarray | None = None
        self.sigmas_: np.ndarray | None = None
        self.consequents_: np.ndarray | None = None  # shape (c, D + 1): [p_{i0}, p_{i1}, ..., p_{iD}]
        self.history_: list[dict[str, float]] = []

    def initialize_from_fcm(
        self,
        X_train: np.ndarray,
        u_train: np.ndarray,
        centers: np.ndarray,
        m: float = 2.0,
    ) -> "TakagiSugenoHybrid":
        """Initialize Gaussian premise centers and fuzzy-weighted widths from FCM output."""
        X_arr = np.asarray(X_train, dtype=float)
        u_arr = np.asarray(u_train, dtype=float)
        ctr = np.asarray(centers, dtype=float).copy()

        um = u_arr ** m  # (c, N)
        denom = np.maximum(np.sum(um, axis=1, keepdims=True), 1e-12)  # (c, 1)
        sq_diff = (X_arr[np.newaxis, :, :] - ctr[:, np.newaxis, :]) ** 2  # (c, N, D)
        weighted_var = np.sum(um[:, :, np.newaxis] * sq_diff, axis=1) / denom  # (c, D)
        sigmas = np.clip(np.sqrt(np.maximum(weighted_var, 1e-12)), self.sigma_min, self.sigma_max)

        self.init_centers_ = ctr.copy()
        self.init_sigmas_ = sigmas.copy()
        self.centers_ = ctr.copy()
        self.sigmas_ = sigmas.copy()
        return self

    def _forward_premise(
        self, X: np.ndarray, centers: np.ndarray | None = None, sigmas: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Gaussian MFs mu (N, c, D), log_w (N, c), and normalized w_bar (N, c)."""
        ctr = self.centers_ if centers is None else centers
        sig = self.sigmas_ if sigmas is None else sigmas
        if ctr is None or sig is None:
            raise RuntimeError("Premise parameters must be initialized first.")

        X_arr = np.asarray(X, dtype=float)
        diff = X_arr[:, np.newaxis, :] - ctr[np.newaxis, :, :]  # (N, c, D)
        inv_s2 = 1.0 / (sig[np.newaxis, :, :] ** 2)
        log_mu = -0.5 * (diff * diff) * inv_s2  # (N, c, D)
        mu = np.exp(np.clip(log_mu, -700.0, 0.0))

        # Log-sum-exp normalization guarantees zero NaNs even for extreme outliers
        log_w = np.sum(log_mu, axis=2)  # (N, c)
        max_log_w = np.max(log_w, axis=1, keepdims=True)
        shifted_exp = np.exp(log_w - max_log_w)
        w_bar = shifted_exp / np.sum(shifted_exp, axis=1, keepdims=True)
        return mu, log_w, w_bar

    @staticmethod
    def _design_matrix(X: np.ndarray, w_bar: np.ndarray) -> np.ndarray:
        """Build LSE design matrix A of shape (N, c * (D + 1))."""
        n, d = X.shape
        c = w_bar.shape[1]
        X_aug = np.hstack([np.ones((n, 1), dtype=float), X])  # (N, D + 1)
        A = w_bar[:, :, np.newaxis] * X_aug[:, np.newaxis, :]  # (N, c, D + 1)
        return A.reshape(n, c * (d + 1))

    def _solve_lse(
        self, X: np.ndarray, y: np.ndarray, w_bar: np.ndarray, ridge_lambda: float | None = None
    ) -> np.ndarray:
        lam = self.ridge_lambda if ridge_lambda is None else float(ridge_lambda)
        A = self._design_matrix(X, w_bar)
        n_params = A.shape[1]
        AtA = A.T @ A
        reg = lam * np.eye(n_params, dtype=float)
        theta = np.linalg.solve(AtA + reg, A.T @ y)
        return theta.reshape(self.n_clusters, self.n_features + 1)

    def fit_zero_order(self, X_train: np.ndarray, y_train: np.ndarray) -> "TakagiSugenoHybrid":
        """Stage 0 baseline: zero-order Sugeno consequents from normalized firing weights."""
        X_arr = np.asarray(X_train, dtype=float)
        y_arr = np.asarray(y_train, dtype=float)
        _, _, w_bar = self._forward_premise(X_arr)
        denom = np.maximum(np.sum(w_bar, axis=0), 1e-12)
        p0 = (w_bar.T @ y_arr) / denom
        conseq = np.zeros((self.n_clusters, self.n_features + 1), dtype=float)
        conseq[:, 0] = p0
        self.consequents_ = conseq
        return self

    def fit_lse(
        self, X_train: np.ndarray, y_train: np.ndarray, ridge_lambda: float | None = None
    ) -> "TakagiSugenoHybrid":
        """Stage 1: Closed-form Regularized Least Squares Estimation of linear consequents."""
        X_arr = np.asarray(X_train, dtype=float)
        y_arr = np.asarray(y_train, dtype=float)
        if ridge_lambda is not None:
            self.ridge_lambda = float(ridge_lambda)
        _, _, w_bar = self._forward_premise(X_arr)
        self.consequents_ = self._solve_lse(X_arr, y_arr, w_bar, self.ridge_lambda)
        return self

    def _predict_with_components(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.consequents_ is None:
            raise RuntimeError("Consequent parameters must be fitted before prediction.")
        X_arr = np.asarray(X, dtype=float)
        _, _, w_bar = self._forward_premise(X_arr)
        X_aug = np.hstack([np.ones((X_arr.shape[0], 1), dtype=float), X_arr])  # (N, D + 1)
        rule_outputs = X_aug @ self.consequents_.T  # (N, c)
        y_pred = np.sum(w_bar * rule_outputs, axis=1)
        return y_pred, w_bar, rule_outputs

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred, _, _ = self._predict_with_components(X)
        return y_pred

    def _compute_loss_and_grads(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute MSE loss (0.5 * mean((y_pred - y)^2)) and analytical gradients wrt centers and log_sigmas."""
        if self.centers_ is None or self.sigmas_ is None or self.consequents_ is None:
            raise RuntimeError("Model must be initialized and fitted before computing gradients.")

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n = X_arr.shape[0]

        y_pred, w_bar, rule_outputs = self._predict_with_components(X_arr)
        err = y_pred - y_arr  # (N,)
        loss = float(0.5 * np.mean(err * err))

        # dL / dw_bar_{ki} = (err_k / N) * rule_outputs_{ki}
        # dL / d(log_w_{ki}) = (err_k / N) * w_bar_{ki} * (rule_outputs_{ki} - y_pred_k)
        dL_dlog_w = (err[:, np.newaxis] / n) * w_bar * (rule_outputs - y_pred[:, np.newaxis])  # (N, c)

        diff = X_arr[:, np.newaxis, :] - self.centers_[np.newaxis, :, :]  # (N, c, D)
        inv_s2 = 1.0 / (self.sigmas_[np.newaxis, :, :] ** 2)  # (1, c, D)

        # log_w_{ki} = sum_j [ -0.5 * (x_{kj} - v_{ij})^2 * exp(-2 * log_sigma_{ij}) ]
        # d(log_w_{ki}) / dv_{ij} = (x_{kj} - v_{ij}) / sigma_{ij}^2
        # d(log_w_{ki}) / d(log_sigma_{ij}) = (x_{kj} - v_{ij})^2 / sigma_{ij}^2
        grad_v = np.sum(dL_dlog_w[:, :, np.newaxis] * diff * inv_s2, axis=0)
        grad_log_sigma = np.sum(dL_dlog_w[:, :, np.newaxis] * (diff * diff) * inv_s2, axis=0)
        return loss, grad_v, grad_log_sigma

    def fit_hybrid(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 120,
        lr: float = 0.05,
        ridge_lambda: float | None = None,
        patience: int = 25,
    ) -> "TakagiSugenoHybrid":
        """Stage 2: Jang (1993) Hybrid Learning (forward LSE + backward Gradient Descent on premises)."""
        X_tr = np.asarray(X_train, dtype=float)
        y_tr = np.asarray(y_train, dtype=float)
        X_va = np.asarray(X_val, dtype=float)
        y_va = np.asarray(y_val, dtype=float)

        if ridge_lambda is not None:
            self.ridge_lambda = float(ridge_lambda)

        # Initialize best checkpoint from Stage-1 LSE
        self.fit_lse(X_tr, y_tr)
        init_tr_rmse = float(np.sqrt(np.mean((y_tr - self.predict(X_tr)) ** 2)))
        init_va_rmse = float(np.sqrt(np.mean((y_va - self.predict(X_va)) ** 2)))

        best_val_rmse = init_va_rmse
        best_centers = self.centers_.copy()
        best_sigmas = self.sigmas_.copy()
        best_conseq = self.consequents_.copy()

        self.history_ = [
            {"epoch": 0, "train_rmse": init_tr_rmse, "val_rmse": init_va_rmse, "lr": float(lr)}
        ]

        current_lr = float(lr)
        prev_loss = 0.5 * (init_tr_rmse ** 2)
        decreases_streak = 0
        no_improve_epochs = 0

        for ep in range(1, int(epochs) + 1):
            # Backward pass on current LSE consequents
            loss, grad_v, grad_log_sigma = self._compute_loss_and_grads(X_tr, y_tr)
            grad_norm = float(np.sqrt(np.sum(grad_v ** 2) + np.sum(grad_log_sigma ** 2)))
            if grad_norm > 5.0:
                scale = 5.0 / grad_norm
                grad_v = grad_v * scale
                grad_log_sigma = grad_log_sigma * scale

            cand_centers = self.centers_ - current_lr * grad_v
            cand_log_sigmas = np.log(self.sigmas_) - current_lr * grad_log_sigma
            cand_sigmas = np.clip(np.exp(cand_log_sigmas), self.sigma_min, self.sigma_max)

            self.centers_ = cand_centers
            self.sigmas_ = cand_sigmas

            # Forward pass: exact LSE update of consequents for updated premise parameters
            self.fit_lse(X_tr, y_tr)
            tr_rmse = float(np.sqrt(np.mean((y_tr - self.predict(X_tr)) ** 2)))
            va_rmse = float(np.sqrt(np.mean((y_va - self.predict(X_va)) ** 2)))
            new_loss = 0.5 * (tr_rmse ** 2)

            self.history_.append(
                {"epoch": ep, "train_rmse": tr_rmse, "val_rmse": va_rmse, "lr": current_lr}
            )

            if va_rmse < best_val_rmse - 1e-8:
                best_val_rmse = va_rmse
                best_centers = self.centers_.copy()
                best_sigmas = self.sigmas_.copy()
                best_conseq = self.consequents_.copy()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # Jang (1993) adaptive learning rate schedule
            if new_loss < prev_loss - 1e-10:
                decreases_streak += 1
                if decreases_streak >= 4:
                    current_lr = min(current_lr * 1.1, 0.5)
                    decreases_streak = 0
            else:
                current_lr = max(current_lr * 0.5, 1e-5)
                decreases_streak = 0
            prev_loss = new_loss

            if no_improve_epochs >= patience:
                break

        # Restore exact best validation checkpoint
        self.centers_ = best_centers
        self.sigmas_ = best_sigmas
        self.consequents_ = best_conseq
        return self

    def explain_single(self, x_scaled: np.ndarray) -> dict[str, Any]:
        """Return detailed step-by-step Sugeno inference breakdown for one sample."""
        if self.consequents_ is None:
            raise RuntimeError("Model must be fitted before explain_single.")
        x_arr = np.asarray(x_scaled, dtype=float).reshape(1, -1)
        mu, log_w, w_bar = self._forward_premise(x_arr)
        x_aug = np.hstack([np.ones((1, 1), dtype=float), x_arr])
        rule_outputs = (x_aug @ self.consequents_.T)[0]
        w_bar_1d = w_bar[0]
        log_w_1d = log_w[0]
        # Rescaled raw firing strengths (product of Gaussian MFs; clip log_w at -700)
        w_raw = np.exp(np.clip(log_w_1d, -700.0, 0.0))
        weighted = w_bar_1d * rule_outputs
        y_pred = float(np.sum(weighted))
        return {
            "mu": mu[0],  # (c, D)
            "log_w": log_w_1d,
            "w": w_raw,
            "w_bar": w_bar_1d,
            "rule_outputs": rule_outputs,
            "weighted_contributions": weighted,
            "y_pred": y_pred,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "n_features": self.n_features,
            "ridge_lambda": self.ridge_lambda,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "init_centers": None if self.init_centers_ is None else self.init_centers_.tolist(),
            "init_sigmas": None if self.init_sigmas_ is None else self.init_sigmas_.tolist(),
            "centers": None if self.centers_ is None else self.centers_.tolist(),
            "sigmas": None if self.sigmas_ is None else self.sigmas_.tolist(),
            "consequents": None if self.consequents_ is None else self.consequents_.tolist(),
            "history": self.history_,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TakagiSugenoHybrid":
        model = cls(
            n_clusters=data["n_clusters"],
            n_features=data["n_features"],
            ridge_lambda=data["ridge_lambda"],
            sigma_min=data.get("sigma_min", 0.25),
            sigma_max=data.get("sigma_max", 5.0),
        )
        for attr, key in [
            ("init_centers_", "init_centers"),
            ("init_sigmas_", "init_sigmas"),
            ("centers_", "centers"),
            ("sigmas_", "sigmas"),
            ("consequents_", "consequents"),
        ]:
            val = data.get(key)
            setattr(model, attr, None if val is None else np.asarray(val, dtype=float))
        model.history_ = data.get("history", [])
        return model

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TakagiSugenoHybrid":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
