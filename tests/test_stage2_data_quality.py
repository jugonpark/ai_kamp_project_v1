import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing_config import default_preprocessing_config
from stage2.data_quality import DataQualityError, inspect_csv, inspect_dataframe


ROOT = Path(__file__).resolve().parents[1]


def sample_frame():
    return pd.DataFrame({"Unnamed: 0": [0, 1, 2, 3],
        "TimeStamp": ["2026-01-01 00:00:00.000", "2026-01-01 00:00:00.100",
                      "2026-01-01 00:00:00.100", "2026-01-01 00:00:00.300"],
        "AI0_Vibration": [-1., 2., 2., 3.], "AI1_Vibration": [1., -2., -2., 3.],
        "AI2_Current": [-10., -20., -20., 30.], "Equipment_state": [0, 0, 0, 0]})


class Stage2DataQualityTests(unittest.TestCase):
    def test_actual_data_quality_counts(self):
        config = default_preprocessing_config()
        normal = inspect_csv(ROOT / "press_data_normal.csv", config, 0)
        anomaly = inspect_csv(ROOT / "outlier_data.csv", config, 1)
        self.assertEqual((normal.report["rows_before"], normal.report["rows_after"],
                          normal.report["duplicates_removed"], normal.report["segment_count"]),
                         (20000, 19999, 1, 599))
        self.assertEqual((anomaly.report["rows_before"], anomaly.report["rows_after"],
                          anomaly.report["duplicates_removed"], anomaly.report["segment_count"]),
                         (600, 600, 0, 21))
        self.assertEqual(normal.report["segment_length_max"], 50)
        self.assertEqual(anomaly.report["segment_length_max"], 50)
        self.assertEqual(normal.report["ignored_export_index_columns"], ["Unnamed: 0"])
        self.assertEqual(normal.report["nan_count"], 0)
        self.assertEqual(normal.report["inf_count"], 0)

    def test_exact_duplicate_uses_all_data_columns_and_preserves_source_row(self):
        result = inspect_dataframe(sample_frame(), default_preprocessing_config(), 0)
        self.assertEqual(result.report["duplicates_removed"], 1)
        self.assertEqual(result.frame["source_row"].tolist(), [0, 1, 3])
        self.assertEqual(result.frame["segment_id"].tolist(), [0, 0, 1])
        self.assertTrue(np.isnan(result.frame["delta_t_ms"].iloc[0]))
        self.assertEqual(result.frame["delta_t_ms"].iloc[2], 200)
        changed = sample_frame(); changed.loc[2, "AI2_Current"] = -21
        kept = inspect_dataframe(changed, default_preprocessing_config(), 0)
        self.assertEqual(kept.report["duplicates_removed"], 0)
        self.assertEqual(kept.report["timestamp_non_increasing_count"], 1)
        self.assertEqual(kept.frame["segment_id"].tolist(), [0, 0, 1, 2])

    def test_nonfinite_sensor_values_fail_with_counts(self):
        for value, key in ((np.nan, "nan"), (np.inf, "inf")):
            frame = sample_frame(); frame.loc[0, "AI0_Vibration"] = value
            with self.subTest(value=value), self.assertRaises(DataQualityError) as caught:
                inspect_dataframe(frame, default_preprocessing_config(), 0)
            self.assertEqual(caught.exception.counts[key], 1)

    def test_bad_timestamp_and_label_fail(self):
        frame = sample_frame(); frame.loc[0, "TimeStamp"] = "bad"
        with self.assertRaisesRegex(DataQualityError, "TimeStamp parse failed"):
            inspect_dataframe(frame, default_preprocessing_config(), 0)
        frame = sample_frame(); frame.loc[0, "Equipment_state"] = 1
        with self.assertRaisesRegex(DataQualityError, "Expected Equipment_state=0"):
            inspect_dataframe(frame, default_preprocessing_config(), 0)
