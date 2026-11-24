"""Fuzzy C-Means (FCM) clustering and cluster validity indices (PC, PE, Xie-Beni)."""
from __future__ import annotations

from typing import Iterable
import numpy as np


class FuzzyCMeans:
    """Bezdek (1981) Fuzzy C-Means clustering in pure NumPy."""

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        max_iter: int = 300,
        tol: float = 1e-5,
        random_state: int = 42,
    ) -> None:
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2.")
        if m <= 1.0:
            raise ValueError("Fuzzifier m must be > 1.0.")
        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)

        self.centers_: np.ndarray | None = None
        self.u_: np.ndarray | None = None
        self.jm_history_: list[float] = []
        self.n_iter_: int = 0

    @staticmethod
    def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Compute squared Euclidean distances of shape (c, N)."""
        diff = centers[:, np.newaxis, :] - X[np.newaxis, :, :]
        return np.sum(diff * diff, axis=2)

    def _compute_membership_from_sq_dist(self, sq_dist: np.ndarray) -> np.ndarray:
        """Compute partition matrix U of shape (c, N) from squared distances."""
        c, n = sq_dist.shape
        u = np.zeros((c, n), dtype=float)
        zero_mask = sq_dist <= 1e-24
        has_zero = np.any(zero_mask, axis=0)

        if np.any(has_zero):
            for k in np.where(has_zero)[0]:
                zeros = zero_mask[:, k]
                u[zeros, k] = 1.0 / np.sum(zeros)

        non_zero = ~has_zero
        if np.any(non_zero):
            d_nz = np.sqrt(sq_dist[:, non_zero])  # (c, n_nz)
            power = 2.0 / (self.m - 1.0)
            # u_{ik} = 1 / sum_r (d_{ik} / d_{rk})^power = d_{ik}^{-power} / sum_r d_{rk}^{-power}
            inv_d = np.power(d_nz, -power)
            u[:, non_zero] = inv_d / np.sum(inv_d, axis=0, keepdims=True)
        return u

    def compute_objective(self, X: np.ndarray, u: np.ndarray, centers: np.ndarray | None = None) -> float:
        ctr = self.centers_ if centers is None else centers
        if ctr is None:
            raise RuntimeError("Model is not fitted.")
        sq_dist = self._squared_distances(np.asarray(X, dtype=float), ctr)
        return float(np.sum((u ** self.m) * sq_dist))

    def fit(self, X: np.ndarray) -> "FuzzyCMeans":
        X_arr = np.asarray(X, dtype=float)
        n_samples = X_arr.shape[0]
        if n_samples < self.n_clusters:
            raise ValueError("Number of samples must be >= n_clusters.")

        rng = np.random.default_rng(self.random_state)
        u = rng.dirichlet(np.ones(self.n_clusters), size=n_samples).T  # (c, N)
        self.jm_history_ = []

        for it in range(1, self.max_iter + 1):
            um = u ** self.m  # (c, N)
            denom = np.sum(um, axis=1, keepdims=True)
            centers = (um @ X_arr) / np.maximum(denom, 1e-12)

            sq_dist = self._squared_distances(X_arr, centers)
            jm = float(np.sum(um * sq_dist))
            self.jm_history_.append(jm)

            u_new = self._compute_membership_from_sq_dist(sq_dist)
            delta = float(np.max(np.abs(u_new - u)))
            u = u_new
            self.n_iter_ = it
            if delta < self.tol:
                break

        # Final center update matching converged U, then canonical ascending sort by mean coordinate
        um = u ** self.m
        centers = (um @ X_arr) / np.maximum(np.sum(um, axis=1, keepdims=True), 1e-12)
        order = np.argsort(np.mean(centers, axis=1))
        self.centers_ = centers[order]
        self.u_ = u[order]
        return self

    def predict_membership(self, X: np.ndarray) -> np.ndarray:
        if self.centers_ is None:
            raise RuntimeError("FuzzyCMeans must be fitted before predict_membership.")
        sq_dist = self._squared_distances(np.asarray(X, dtype=float), self.centers_)
        return self._compute_membership_from_sq_dist(sq_dist)

    def validity_metrics(self, X: np.ndarray) -> dict[str, float]:
        """Compute PC (higher better), PE (lower better), Xie-Beni (lower better), and J_m."""
        if self.centers_ is None or self.u_ is None:
            raise RuntimeError("FuzzyCMeans must be fitted before validity_metrics.")
        X_arr = np.asarray(X, dtype=float)
        n = X_arr.shape[0]
        u = self.u_
        pc = float(np.sum(u * u) / n)
        u_safe = np.clip(u, 1e-12, 1.0)
        pe = float(-np.sum(u * np.log(u_safe)) / n)

        sq_dist = self._squared_distances(X_arr, self.centers_)
        jm = float(np.sum((u ** self.m) * sq_dist))

        # Minimum pairwise squared Euclidean distance between distinct cluster centers
        c_sq = self._squared_distances(self.centers_, self.centers_)
        np.fill_diagonal(c_sq, np.inf)
        min_center_sq = float(np.min(c_sq))
        xb = float(jm / (n * max(min_center_sq, 1e-12)))

        return {
            "pc": pc,
            "pe": pe,
            "xb": xb,
            "jm": jm,
            "n_iter": int(self.n_iter_),
        }


def evaluate_cluster_counts(
    X: np.ndarray,
    c_values: Iterable[int] = (2, 3, 4, 5, 6),
    m: float = 2.0,
    max_iter: int = 300,
    tol: float = 1e-5,
    random_state: int = 42,
) -> list[dict[str, float]]:
    """Sweep cluster counts and compute FCM validity metrics for each c."""
    results: list[dict[str, float]] = []
    for c in c_values:
        fcm = FuzzyCMeans(
            n_clusters=int(c), m=m, max_iter=max_iter, tol=tol, random_state=random_state
        ).fit(X)
        metrics = fcm.validity_metrics(X)
        results.append({"c": int(c), **metrics})
    return results
