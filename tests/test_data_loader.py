import unittest
import numpy as np
import pandas as pd

from src.data_loader import (
    FEATURE_COLS,
    StandardScalerNumpy,
    build_next_day_dataset,
    chronological_split,
    clean_ispu_dataframe,
)


class TestDataLoader(unittest.TestCase):
    def _make_sample_df(self) -> pd.DataFrame:
        # Station A: days 1..6 continuous, plus day 8 (gap on day 7)
        # Station B: days 1..4 continuous
        rows = []
        for day in [1, 2, 3, 4, 5, 6, 8]:
            rows.append(
                {
                    "tanggal": f"2023-01-{day:02d}",
                    "lokasi_spku": "DKI1 (Bunderan HI)",
                    "pm_sepuluh": 40 + day,
                    "pm_duakomalima": 60 + day * 2,
                    "so2": 20 + day,
                    "co": 10 + day,
                    "o3": 30 + day,
                    "no2": 15 + day,
                    "max": 60 + day * 2,
                    "kategori": "SEDANG",
                }
            )
        for day in [1, 2, 3, 4]:
            rows.append(
                {
                    "tanggal": f"2023-01-{day:02d}",
                    "lokasi_spku": "DKI2 (Kelapa Gading)",
                    "pm_sepuluh": 35 + day,
                    "pm_duakomalima": 45 + day,
                    "so2": 18 + day,
                    "co": 9 + day,
                    "o3": 25 + day,
                    "no2": 12 + day,
                    "max": 45 + day,
                    "kategori": "BAIK",
                }
            )
        # Add invalid row with '---' marker and duplicate row
        rows.append(
            {
                "tanggal": "2023-01-10",
                "lokasi_spku": "DKI1 (Bunderan HI)",
                "pm_sepuluh": "---",
                "pm_duakomalima": 50,
                "so2": 20,
                "co": 10,
                "o3": 30,
                "no2": 15,
                "max": 50,
                "kategori": "BAIK",
            }
        )
        rows.append(rows[0].copy())
        return pd.DataFrame(rows)

    def test_clean_normalizes_aliases_and_removes_invalid(self):
        raw = self._make_sample_df()
        cleaned, stats = clean_ispu_dataframe(raw)
        for col in ["tanggal", "stasiun", *FEATURE_COLS, "max", "categori"]:
            self.assertIn(col, cleaned.columns)
        # 7 valid Station A + 4 valid Station B = 11 rows (invalid & duplicate dropped)
        self.assertEqual(len(cleaned), 11)
        self.assertEqual(stats["raw_rows"], 13)
        self.assertEqual(stats["clean_rows"], 11)

    def test_next_day_pairing_respects_station_and_exact_1day_gap(self):
        raw = self._make_sample_df()
        cleaned, _ = clean_ispu_dataframe(raw)
        paired = build_next_day_dataset(cleaned)
        # Station A: (1->2, 2->3, 3->4, 4->5, 5->6) = 5 pairs; day 6 does NOT pair to day 8
        # Station B: (1->2, 2->3, 3->4) = 3 pairs
        self.assertEqual(len(paired), 8)
        diffs = (paired["next_tanggal"] - paired["tanggal"]).dt.days
        self.assertTrue((diffs == 1).all())
        # Verify Station A day 1 target is day 2's max (60 + 2*2 = 64)
        row_a1 = paired[
            (paired["stasiun"] == "DKI1 (Bunderan HI)")
            & (paired["tanggal"] == pd.Timestamp("2023-01-01"))
        ].iloc[0]
        self.assertAlmostEqual(row_a1["target_ispu_next"], 64.0)

    def test_chronological_split_has_no_temporal_leakage_and_train_only_scaler(self):
        # Build 40 consecutive days for 1 station
        rows = []
        for i in range(40):
            dt = pd.Timestamp("2023-01-01") + pd.Timedelta(days=i)
            rows.append(
                {
                    "tanggal": dt.strftime("%Y-%m-%d"),
                    "stasiun": "DKI1",
                    "pm10": 30 + i,
                    "pm25": 40 + i * 2,
                    "so2": 15 + (i % 5),
                    "co": 10 + (i % 4),
                    "o3": 25 + (i % 7),
                    "no2": 12 + (i % 3),
                    "max": 40 + i * 2,
                    "categori": "SEDANG",
                }
            )
        cleaned, _ = clean_ispu_dataframe(pd.DataFrame(rows))
        paired = build_next_day_dataset(cleaned)
        split = chronological_split(paired, train_ratio=0.70, val_ratio=0.15)
        self.assertLess(split["train"]["tanggal"].max(), split["val"]["tanggal"].min())
        self.assertLess(split["val"]["tanggal"].max(), split["test"]["tanggal"].min())

        # Scaler must fit ONLY on train: X_train_scaled has mean ~0 and std ~1, while test mean > 0
        self.assertTrue(np.allclose(split["X_train_scaled"].mean(axis=0), 0.0, atol=1e-10))
        self.assertTrue(np.allclose(split["X_train_scaled"].std(axis=0), 1.0, atol=1e-10))
        self.assertGreater(split["X_test_scaled"][:, 0].mean(), 1.0)

        # Round-trip inverse transform
        scaler: StandardScalerNumpy = split["scaler"]
        recovered = scaler.inverse_transform(split["X_train_scaled"])
        self.assertTrue(np.allclose(recovered, split["X_train"], atol=1e-10))


if __name__ == "__main__":
    unittest.main()
