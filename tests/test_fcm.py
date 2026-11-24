import unittest
import numpy as np

from src.fcm import FuzzyCMeans, evaluate_cluster_counts


class TestFuzzyCMeans(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        c1 = rng.normal(loc=[-4.0, -4.0], scale=0.25, size=(40, 2))
        c2 = rng.normal(loc=[0.0, 0.0], scale=0.25, size=(40, 2))
        c3 = rng.normal(loc=[4.0, 4.0], scale=0.25, size=(40, 2))
        self.X = np.vstack([c1, c2, c3])

    def test_membership_partition_invariants_and_monotonic_jm(self):
        fcm = FuzzyCMeans(n_clusters=3, m=2.0, max_iter=200, tol=1e-6, random_state=42)
        fcm.fit(self.X)

        self.assertEqual(fcm.centers_.shape, (3, 2))
        self.assertEqual(fcm.u_.shape, (3, len(self.X)))
        self.assertTrue(np.all((fcm.u_ >= 0.0) & (fcm.u_ <= 1.0)))
        self.assertTrue(np.allclose(fcm.u_.sum(axis=0), 1.0, atol=1e-10))

        # J_m must be non-increasing across iterations
        diffs = np.diff(fcm.jm_history_)
        self.assertTrue(np.all(diffs <= 1e-8))

        # Centers should be sorted ascending by intensity and match [-4,-4], [0,0], [4,4]
        expected = np.array([[-4.0, -4.0], [0.0, 0.0], [4.0, 4.0]])
        self.assertTrue(np.allclose(fcm.centers_, expected, atol=0.2))

    def test_zero_distance_point_gives_exact_one_hot_membership(self):
        fcm = FuzzyCMeans(n_clusters=3, m=2.0, random_state=42).fit(self.X)
        u_exact = fcm.predict_membership(fcm.centers_)
        self.assertTrue(np.all(np.isfinite(u_exact)))
        self.assertTrue(np.allclose(u_exact, np.eye(3), atol=1e-12))

    def test_validity_metrics_and_cluster_sweep(self):
        fcm = FuzzyCMeans(n_clusters=3, m=2.0, random_state=42).fit(self.X)
        metrics = fcm.validity_metrics(self.X)
        self.assertGreater(metrics["pc"], 0.85)
        self.assertGreater(metrics["pe"], 0.0)
        self.assertGreater(metrics["xb"], 0.0)
        self.assertTrue(np.isfinite(metrics["xb"]))

        sweep = evaluate_cluster_counts(self.X, c_values=(2, 3, 4), random_state=42)
        self.assertEqual([row["c"] for row in sweep], [2, 3, 4])
        # Optimal Xie-Beni (lowest) on 3 well-separated clusters should be at c=3
        best_c = min(sweep, key=lambda r: r["xb"])["c"]
        self.assertEqual(best_c, 3)


if __name__ == "__main__":
    unittest.main()
