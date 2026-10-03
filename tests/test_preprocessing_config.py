import unittest
from pathlib import Path
import shutil
import uuid

from preprocessing_config import (default_preprocessing_config, validate_preprocessing_config,
    save_preprocessing_config, load_preprocessing_config)


class PreprocessingConfigTests(unittest.TestCase):
    def test_stage2_defaults_and_fixed_baseline(self):
        stage2 = default_preprocessing_config()
        self.assertEqual(stage2, {"mode": "STAGE2_SEGMENT_AWARE", "expected_interval_ms": 100,
            "gap_threshold_ms": 150, "remove_exact_duplicates": True, "segment_aware": True,
            "signal_transform": "ABS_ALL", "scaler": "MINMAX", "sequence_length": 20,
            "stride": 1, "use_horizon": False, "prediction_horizon": 100})
        baseline = default_preprocessing_config("KAMP_BASELINE")
        self.assertFalse(baseline["remove_exact_duplicates"])
        self.assertFalse(baseline["segment_aware"])
        self.assertTrue(baseline["use_horizon"])
        with self.assertRaises(ValueError):
            validate_preprocessing_config({**baseline, "sequence_length": 15})

    def test_json_round_trip(self):
        config = validate_preprocessing_config({"sequence_length": 15, "gap_threshold_ms": 200})
        folder = Path.cwd() / "outputs" / ".test_artifacts" / uuid.uuid4().hex
        try:
            path = save_preprocessing_config(folder / "stage2.json", config)
            self.assertEqual(load_preprocessing_config(path), config)
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_invalid_values_rejected(self):
        for change in ({"gap_threshold_ms": 100}, {"sequence_length": 12},
                       {"stride": 0}, {"remove_exact_duplicates": 1},
                       {"scaler": "UNKNOWN"}, {"unknown": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_preprocessing_config(change)
