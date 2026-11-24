import unittest
import numpy as np
import pandas as pd

from src.evaluation import (
    compute_metrics,
    evaluate_persistence_baseline,
    fit_predict_ols,
    ispu_to_category,
)


class TestEvaluation(unittest.TestCase):
    def test_permen_lhk_category_boundaries(self):
        self.assertEqual(ispu_to_category(0.0), "Baik")
        self.assertEqual(ispu_to_category(50.0), "Baik")
        self.assertEqual(ispu_to_category(50.1), "Sedang")
        self.assertEqual(ispu_to_category(51.0), "Sedang")
        self.assertEqual(ispu_to_category(100.0), "Sedang")
        self.assertEqual(ispu_to_category(101.0), "Tidak Sehat")
        self.assertEqual(ispu_to_category(200.0), "Tidak Sehat")
        self.assertEqual(ispu_to_category(201.0), "Sangat Tidak Sehat")
        self.assertEqual(ispu_to_category(300.0), "Sangat Tidak Sehat")
        self.assertEqual(ispu_to_category(301.0), "Berbahaya")

    def test_compute_metrics_and_baselines(self):
        y_true = np.array([40.0, 80.0, 120.0, 220.0])
        y_pred = np.array([42.0, 78.0, 118.0, 215.0])
        m = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["category_accuracy"], 1.0)
        self.assertGreater(m["r2"], 0.99)
        self.assertGreater(m["rmse"], 0.0)

        df = pd.DataFrame({"max": [40.0, 75.0, 110.0]})
        pers = evaluate_persistence_baseline(df)
        self.assertTrue(np.allclose(pers, [40.0, 75.0, 110.0]))

        X_tr = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
        y_tr = 5.0 + 2.0 * X_tr[:, 0] + 3.0 * X_tr[:, 1]
        pred_tr, pred_ev = fit_predict_ols(X_tr, y_tr, X_tr)
        self.assertTrue(np.allclose(pred_tr, y_tr, atol=1e-8))
        self.assertTrue(np.allclose(pred_ev, y_tr, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
