import unittest
from unittest.mock import patch

import numpy as np

from preprocessing_config import default_preprocessing_config
from stage2.dataset_artifacts import (discover_processed_datasets, load_processed_dataset,
    save_processed_dataset, validate_dataset_id, validate_processed_dataset)
from stage2.preprocessing import run_stage2_preprocessing
from train_lstm_ae import NORMAL_PATH, OUTLIER_PATH, OUTPUT_DIR, FEATURES


class ProcessedDatasetArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_stage2_preprocessing(NORMAL_PATH, OUTLIER_PATH, default_preprocessing_config())
        cls.root = OUTPUT_DIR / ".test_artifacts" / "processed_dataset_tests"
        cls.root.mkdir(parents=True, exist_ok=True)
        cls.dataset_id = "roundtrip_test"
        cls.target = cls.root / cls.dataset_id
        if not cls.target.exists():
            cls.artifact = save_processed_dataset(cls.result, cls.dataset_id, cls.root)
        else:
            cls.artifact = load_processed_dataset(cls.target)

    def test_roundtrip_shapes_metadata_and_scaler(self):
        artifact = load_processed_dataset(self.target)
        self.assertEqual(artifact.config["preprocessing"], self.result.config)
        self.assertEqual(artifact.train["X"].shape, (7411, 20, 3))
        self.assertEqual(artifact.validation["X"].shape, (646, 20, 3))
        self.assertEqual(artifact.test["X"].shape, (2197, 20, 3))
        self.assertTrue(np.array_equal(artifact.train["X"], self.result.train_windows.X))
        self.assertTrue(np.all(artifact.train["y"] == 0))
        for batch in (artifact.train, artifact.validation, artifact.test):
            self.assertEqual({len(v) for v in batch.values()}, {len(batch["X"])})
            self.assertEqual(batch["window_start_timestamp"].dtype, np.dtype("datetime64[ns]"))
        raw = np.array([[1., -2., 3.]])
        np.testing.assert_array_equal(self.result.scaler.transform(raw), artifact.scaler.transform(raw))
        self.assertTrue(validate_processed_dataset(self.target, self.result))

    def test_source_hash_and_cleaned_csv(self):
        import pandas as pd
        for stream in ("normal", "anomaly"):
            source = self.artifact.config["source"][stream]
            self.assertEqual(len(source["sha256"]), 64)
            self.assertGreater(source["size_bytes"], 0)
            frame = pd.read_csv(self.target / f"{stream}_cleaned.csv")
            self.assertTrue({"TimeStamp", *FEATURES, "Equipment_state", "segment_id", "delta_t_ms", "split"}.issubset(frame.columns))
            self.assertEqual(len(frame), self.result.summary[f"{stream}_cleaned_rows"])

    def test_overwrite_invalid_ids_crossgap_and_discovery(self):
        with self.assertRaises(FileExistsError):
            save_processed_dataset(self.result, self.dataset_id, self.root)
        for invalid in ("", "a/b", "a b", ".."):
            with self.assertRaises(ValueError): validate_dataset_id(invalid)
        with patch.dict(self.result.summary, {"cross_gap_windows": 1}):
            with self.assertRaises(ValueError): save_processed_dataset(self.result, "crossgap", self.root)
        incomplete = self.root / "incomplete_test"
        incomplete.mkdir(exist_ok=True)
        self.assertEqual([item["dataset_id"] for item in discover_processed_datasets(self.root)], [self.dataset_id])
        config_path = self.target / "dataset_config.json"
        original = config_path.read_bytes()
        try:
            config_path.write_bytes(original + b" ")
            with self.assertRaises(ValueError): load_processed_dataset(self.target)
            self.assertEqual(discover_processed_datasets(self.root), [])
        finally:
            config_path.write_bytes(original)

    def test_changed_source_and_failed_write_leave_no_discoverable_dataset(self):
        with patch("stage2.dataset_artifacts.fingerprint_csv_source", return_value={"changed": True}):
            with self.assertRaisesRegex(ValueError, "source CSV changed"):
                save_processed_dataset(self.result, "changed_source", self.root)
        with patch("stage2.dataset_artifacts.np.savez_compressed", side_effect=OSError("write failed")):
            with self.assertRaisesRegex(OSError, "write failed"):
                save_processed_dataset(self.result, "failed_write", self.root)
        self.assertFalse((self.root / "failed_write").exists())
        self.assertFalse(any(path.name.startswith(".failed_write.tmp-") for path in self.root.iterdir()))


if __name__ == "__main__": unittest.main()
