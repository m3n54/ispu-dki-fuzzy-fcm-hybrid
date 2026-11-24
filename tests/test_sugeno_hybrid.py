import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.fcm import FuzzyCMeans
from src.sugeno_hybrid import TakagiSugenoHybrid


class TestTakagiSugenoHybrid(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.X_train = rng.uniform(-2.0, 2.0, size=(120, 3))
        self.X_val = rng.uniform(-2.0, 2.0, size=(40, 3))

        def target_fn(X: np.ndarray) -> np.ndarray:
            return (
                50.0
                + 15.0 * X[:, 0]
                + 8.0 * np.sin(1.5 * X[:, 1])
                + 4.0 * (X[:, 0] * X[:, 2])
                + 3.0 * (X[:, 2] ** 2)
            )

        self.y_train = target_fn(self.X_train)
        self.y_val = target_fn(self.X_val)

        self.fcm = FuzzyCMeans(n_clusters=3, m=2.0, random_state=7).fit(self.X_train)

    def test_extreme_input_log_space_stability(self):
        model = TakagiSugenoHybrid(n_clusters=3, n_features=3, ridge_lambda=1.0)
        model.initialize_from_fcm(self.X_train, self.fcm.u_, self.fcm.centers_)
        model.fit_lse(self.X_train, self.y_train)

        extreme_x = np.array([[500.0, -500.0, 250.0], [-100.0, 100.0, -100.0]])
        preds = model.predict(extreme_x)
        self.assertEqual(preds.shape, (2,))
        self.assertTrue(np.all(np.isfinite(preds)))

        trace = model.explain_single(extreme_x[0])
        self.assertAlmostEqual(float(np.sum(trace["w_bar"])), 1.0, places=10)
        self.assertAlmostEqual(
            float(np.sum(trace["weighted_contributions"])), trace["y_pred"], places=10
        )

    def test_analytical_gradient_matches_finite_differences(self):
        model = TakagiSugenoHybrid(n_clusters=3, n_features=3, ridge_lambda=0.5)
        model.initialize_from_fcm(self.X_train, self.fcm.u_, self.fcm.centers_)
        model.fit_lse(self.X_train, self.y_train)

        loss0, grad_v, grad_log_s = model._compute_loss_and_grads(self.X_train, self.y_train)
        self.assertTrue(np.isfinite(loss0))

        eps = 1e-6
        num_grad_v = np.zeros_like(model.centers_)
        for i in range(model.n_clusters):
            for j in range(model.n_features):
                orig = model.centers_[i, j]
                model.centers_[i, j] = orig + eps
                lp, _, _ = model._compute_loss_and_grads(self.X_train, self.y_train)
                model.centers_[i, j] = orig - eps
                lm, _, _ = model._compute_loss_and_grads(self.X_train, self.y_train)
                model.centers_[i, j] = orig
                num_grad_v[i, j] = (lp - lm) / (2.0 * eps)

        num_grad_ls = np.zeros_like(model.sigmas_)
        log_s0 = np.log(model.sigmas_)
        for i in range(model.n_clusters):
            for j in range(model.n_features):
                ls = log_s0.copy()
                ls[i, j] += eps
                model.sigmas_ = np.exp(ls)
                lp, _, _ = model._compute_loss_and_grads(self.X_train, self.y_train)
                ls[i, j] -= 2.0 * eps
                model.sigmas_ = np.exp(ls)
                lm, _, _ = model._compute_loss_and_grads(self.X_train, self.y_train)
                model.sigmas_ = np.exp(log_s0)
                num_grad_ls[i, j] = (lp - lm) / (2.0 * eps)

        self.assertLess(np.max(np.abs(grad_v - num_grad_v)), 1e-5)
        self.assertLess(np.max(np.abs(grad_log_s - num_grad_ls)), 1e-5)

    def test_ablation_improvement_and_json_roundtrip(self):
        model = TakagiSugenoHybrid(n_clusters=3, n_features=3, ridge_lambda=0.5)
        model.initialize_from_fcm(self.X_train, self.fcm.u_, self.fcm.centers_)

        model.fit_zero_order(self.X_train, self.y_train)
        rmse0_val = float(np.sqrt(np.mean((self.y_val - model.predict(self.X_val)) ** 2)))

        model.fit_lse(self.X_train, self.y_train)
        rmse1_val = float(np.sqrt(np.mean((self.y_val - model.predict(self.X_val)) ** 2)))
        self.assertLess(rmse1_val, rmse0_val)

        model.fit_hybrid(self.X_train, self.y_train, self.X_val, self.y_val, epochs=60, lr=0.05)
        rmse2_val = float(np.sqrt(np.mean((self.y_val - model.predict(self.X_val)) ** 2)))
        self.assertLessEqual(rmse2_val, rmse1_val + 1e-12)

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "model.json"
            model.save(p)
            loaded = TakagiSugenoHybrid.load(p)
            diff = np.max(np.abs(model.predict(self.X_val) - loaded.predict(self.X_val)))
            self.assertLess(diff, 1e-12)


if __name__ == "__main__":
    unittest.main()
